"""
src/training/progressive_env.py
===============================
Implements Phase B: Progressive Deployment Curriculum.

This module provides a Gym Wrapper around `VNEOrderingEnv`.
It intercepts calls to generate the substrate and batch, instead providing
a live, continuously-depleting substrate. When the agent achieves a target AR,
the accepted VNRs are permanently committed to the live substrate, and the agent
levels up.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple
from collections import deque
import random
import numpy as np
import networkx as nx
import copy

import gymnasium

from src.scheduler.environment import VNEOrderingEnv
from src.utils.graph_utils import copy_substrate, release_vnr_embedding, substrate_utilisation
from src.generators.vnr_generator import generate_single_vnr
from src.scheduler.rewards import RewardMode


@dataclass
class ProgressiveConfig:
    promote_ar_threshold: float = 0.85
    promote_window: int = 5
    floor_ar: float = 0.20
    vnr_lifetime_episodes: int = 20
    level_up_batch_delta: int = 2
    level_up_demand_scale: float = 1.15
    max_batch_size: int = 30
    max_demand_scale: float = 3.0
    release_fraction: float = 0.3


class ProgressiveDeploymentWrapper(gymnasium.Wrapper):
    """
    Wraps VNEOrderingEnv to implement the progressive deployment curriculum.
    """

    def __init__(self, env: VNEOrderingEnv, cfg: ProgressiveConfig):
        super().__init__(env)
        self.cfg = cfg
        self.inner_env = env

        # Extract initial generators
        self.original_substrate_fn = env.substrate_fn
        self.original_batch_fn = env.batch_fn

        # Overwrite inner env factories to point to our progressive state
        self.inner_env.substrate_fn = self._get_working_substrate
        self.inner_env.batch_fn = self._get_dynamic_batch

        # Curriculum State
        self.live_substrate = self.original_substrate_fn()
        self.committed_vnrs = []  # List of (vnr, mapping, link_paths, commit_episode)
        
        self.episode_ar_history = deque(maxlen=cfg.promote_window)
        self.current_episode = 0
        self.level = 1
        
        # Scaling modifiers
        self.demand_scale = 1.0
        self.current_batch_size = len(self.original_batch_fn())

    def reset(self, seed=None, options=None):
        self.current_episode += 1
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Environment Factory Overrides
    # ------------------------------------------------------------------

    def _get_working_substrate(self) -> nx.Graph:
        """Called by env.reset() — returns a fresh copy of the live (partially filled) substrate."""
        return copy_substrate(self.live_substrate)

    def _get_dynamic_batch(self) -> List[nx.Graph]:
        """Called by env.reset() — returns a new batch possibly scaled by level up."""
        # Use inner env's original base logic if possible, or build custom scaled VNRs
        # For simplicity, we sample from original and scale them up
        batch = self.original_batch_fn()

        # Infer the observed node-count range from the returned batch so that
        # any synthetic extra VNRs stay within the same size distribution.
        observed_sizes = [len(g.nodes) for g in batch]
        min_n = max(2, min(observed_sizes))
        max_n = max(min_n, max(observed_sizes))

        # Adjust batch size
        if len(batch) < self.current_batch_size:
            # Need to synthesize more VNRs; sample size from observed range
            extra_count = self.current_batch_size - len(batch)
            for _ in range(extra_count):
                n = random.randint(min_n, max_n)
                cpu_req = (10, 40)
                bw_req = (10, 50)
                vnr = generate_single_vnr(n, 0.5, cpu_req, bw_req)
                batch.append(vnr)
        elif len(batch) > self.current_batch_size:
            batch = batch[:self.current_batch_size]

        # Apply demand scaling
        if self.demand_scale > 1.0:
            for vnr in batch:
                for n, d in vnr.nodes(data=True):
                    d['cpu'] = float(d.get('cpu', 0)) * self.demand_scale
                for u, v, d in vnr.edges(data=True):
                    d['bw'] = float(d.get('bw', 0)) * self.demand_scale

        return batch

    # ------------------------------------------------------------------
    # Progressive Logic
    # ------------------------------------------------------------------

    def maybe_promote(self, episode_summary: dict) -> dict:
        """
        Called externally (e.g., by train_progressive.py) after each episode finishes.
        Evaluates AR against thresholds and triggers commits, expiry, and level ups.
        
        Returns a dict of curriculum events (for logging).
        """
        ar = episode_summary['acc_rate']
        self.episode_ar_history.append(ar)
        
        mean_ar = sum(self.episode_ar_history) / len(self.episode_ar_history)
        
        events = {
            "levelled_up": False,
            "expired_count": 0,
            "committed_count": 0,
        }

        # 1. Check early release (Floor AR)
        if mean_ar < self.cfg.floor_ar and len(self.episode_ar_history) == self.cfg.promote_window:
            events["expired_count"] = self._force_expire_fraction()
            self.episode_ar_history.clear()

        # 2. Check promotion
        elif mean_ar >= self.cfg.promote_ar_threshold and len(self.episode_ar_history) == self.cfg.promote_window:
            # Commit the accepted VNRs from the LAST episode permanently
            accepted_this_ep = self.inner_env.accepted
            events["committed_count"] = self._commit_to_live(accepted_this_ep)
            
            # Expire natural lifetimes
            events["expired_count"] += self._expire_old_vnrs()
            
            # Level up!
            self._level_up()
            events["levelled_up"] = True
            
            # Reset history so we don't double loop
            self.episode_ar_history.clear()
            
        return events

    def _commit_to_live(self, accepted_vnrs: list) -> int:
        """Commit resources onto live substrate."""
        count = 0
        for vnr, mapping, link_paths in accepted_vnrs:
            # Consume CPU
            for v_node, s_node in mapping.items():
                if s_node in self.live_substrate.nodes:
                    cpu_req = float(vnr.nodes[v_node].get('cpu', 0.0))
                    self.live_substrate.nodes[s_node]['cpu'] -= cpu_req
                    
            # Consume BW
            for (u, v), path in link_paths.items():
                bw_req = float(vnr.edges[u, v].get('bw', 0.0))
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    if self.live_substrate.has_edge(a, b):
                        self.live_substrate.edges[a, b]['bw'] -= bw_req
            
            self.committed_vnrs.append((vnr, mapping, link_paths, self.current_episode))
            count += 1
        return count

    def _expire_old_vnrs(self) -> int:
        """Release VNRs that have exceeded their lifetime."""
        remaining = []
        expired_count = 0
        for record in self.committed_vnrs:
            vnr, mapping, link_paths, start_ep = record
            if (self.current_episode - start_ep) >= self.cfg.vnr_lifetime_episodes:
                release_vnr_embedding(self.live_substrate, vnr, mapping, link_paths)
                expired_count += 1
            else:
                remaining.append(record)
        self.committed_vnrs = remaining
        return expired_count

    def _force_expire_fraction(self) -> int:
        """Used when floor AR is broken (substrate is too jammed). Releases oldest X%."""
        if not self.committed_vnrs:
            return 0
            
        # Sort by oldest first
        self.committed_vnrs.sort(key=lambda x: x[3])
        release_count = max(1, int(len(self.committed_vnrs) * self.cfg.release_fraction))
        
        expired = 0
        for i in range(release_count):
            vnr, mapping, link_paths, _ = self.committed_vnrs[i]
            release_vnr_embedding(self.live_substrate, vnr, mapping, link_paths)
            expired += 1
            
        self.committed_vnrs = self.committed_vnrs[release_count:]
        return expired

    def _level_up(self):
        self.level += 1
        
        # Determine reward mode phase shift (Phase C3)
        if self.level == 3:
            self.inner_env.reward_mode = RewardMode.REVENUE
        elif self.level == 5:
            self.inner_env.reward_mode = RewardMode.CONGESTION_AWARE
        elif self.level == 7:
            self.inner_env.reward_mode = RewardMode.REJECTION_SCALED
        elif self.level == 9:
            self.inner_env.reward_mode = RewardMode.LONGTERM
        
        # Scaling Strategy: Alternate between increasing batch size and increasing demand
        if self.level % 2 == 0 and self.current_batch_size < self.cfg.max_batch_size:
            self.current_batch_size += self.cfg.level_up_batch_delta
            self.current_batch_size = min(self.current_batch_size, self.cfg.max_batch_size)
        else:
            if self.demand_scale < self.cfg.max_demand_scale:
                self.demand_scale *= self.cfg.level_up_demand_scale
                self.demand_scale = min(self.demand_scale, self.cfg.max_demand_scale)
