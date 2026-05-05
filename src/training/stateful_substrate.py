"""
src/training/stateful_substrate.py
====================================
StatefulSubstrateWrapper — resolves train/inference distribution mismatch
by maintaining a persistent, depleting substrate across training episodes.

Problem addressed
-----------------
VNEOrderingEnv.reset() always creates a fresh substrate (0% utilisation).
BatchedVNRSimulator inference operates on a substrate that accumulates
depletion across many time windows. This covariate shift degrades policy
performance at inference time.

Solution
--------
This wrapper intercepts the substrate_fn call inside VNEOrderingEnv.reset()
and replaces it with a closure that returns a copy of a live, persistent
substrate. The wrapper commits accepted VNR embeddings to the live substrate
and expires them after K episodes, simulating the load/unload cycle of
BatchedVNRSimulator.

Usage
-----
Enabled via PPOConfig.stateful_substrate = True.
When disabled (default), behaviour is identical to baseline training.

Constraints
-----------
- Does NOT modify VNEOrderingEnv, hpso_embed, BatchedVNRSimulator, or
  any module outside src/training/.
- Backward-compatible: disabled by default.
- Replicates exact resource physics: consume on accept, release on expiry.

Attribute name contract
-----------------------
Matches graph_utils.py and substrate_utilisation():
  node: 'cpu'     = available CPU,  'max_cpu' = capacity
  edge: 'bw'      = available BW,   'max_bw'  = capacity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class StatefulSubstrateConfig:
    """
    Configuration for StatefulSubstrateWrapper.

    All parameters have safe defaults. When enabled=False the wrapper
    is a no-op and training behaviour is identical to the baseline.
    """
    enabled: bool = False

    # --- Lifetime simulation ---
    vnr_lifetime_episodes: int = 5
    # How many episodes a committed VNR occupies live_substrate before
    # being expired and its resources returned.
    #
    # Calibration formula (derive from dataset stats):
    #   K = round(avg_lifetime / (avg_inter_arrival * window_size))
    #
    # Example: avg_lifetime=25, avg_inter_arrival=1.0, window_size=10
    #   → K = round(25 / (1.0 * 10)) = 3 (for default generate_data.py settings)
    #
    # For fig6 dataset (max_lifetime=300, Pareto α=2.5):
    #   → K ≈ 7

    # --- Overflow protection ---
    overflow_cpu_threshold: float = 0.88
    # When live_substrate CPU utilisation exceeds this fraction, the
    # wrapper resets live_substrate to a fresh substrate.
    # 0.88 leaves a small buffer above the expected ~80% inference peak.
    # Do not set below 0.80 (excessive resets) or above 0.95 (stalled training).

    # --- Warm-up ---
    warmup_episodes: int = 50
    # For the first N episodes, no commits are made to live_substrate,
    # so it stays fresh (0% util). Allows the agent to learn a basic
    # policy before encountering pre-depleted substrates.

    # --- Substrate regeneration on overflow ---
    regenerate_substrate_on_overflow: bool = True
    # If True (default), overflow resets to a newly generated substrate
    # via the original substrate_fn.
    # If False, overflow resets to the substrate captured at wrapper init
    # (same topology, all resources restored).


# ---------------------------------------------------------------------------
# VNR record dataclass
# ---------------------------------------------------------------------------

@dataclass
class VNRRecord:
    """
    Tracks a single committed VNR on live_substrate.

    Fields
    ------
    vnr            : the original VNR graph (needed for cpu/bw demand per vnode/vedge)
    mapping        : {vnode_id: snode_id}  node placement
    link_paths     : {(u, v): [s0, s1, ..., sk]}  path for each virtual link
    commit_episode : episode number when this VNR was committed
    expiry_episode : commit_episode + vnr_lifetime_episodes
    vnr_object_id  : id(vnr) — used to prevent double-commits
    """
    vnr: nx.Graph
    mapping: Dict[int, int]
    link_paths: Dict[Tuple[int, int], List[int]]
    commit_episode: int
    expiry_episode: int
    vnr_object_id: int = field(default=0)

    def __post_init__(self):
        if self.vnr_object_id == 0:
            self.vnr_object_id = id(self.vnr)


# ---------------------------------------------------------------------------
# Module-level resource helpers
# ---------------------------------------------------------------------------

def _check_resources(substrate: nx.Graph, record: VNRRecord) -> bool:
    """
    Pre-validate that substrate has sufficient CPU and BW to honour the
    VNR record WITHOUT deducting anything.

    Returns True if all resources are available, False otherwise.
    Called by _consume_resources as a guard; also useful for debugging.
    """
    # Node check
    for vnode, snode in record.mapping.items():
        if snode not in substrate.nodes:
            return False
        cpu_req = float(record.vnr.nodes[vnode].get('cpu', 0.0))
        current = float(substrate.nodes[snode].get('cpu', 0.0))
        if current < cpu_req:
            logger.warning(
                "_check_resources: CPU insufficient on snode %s "
                "(avail=%.2f, req=%.2f) for VNR obj_id=%d",
                snode, current, cpu_req, record.vnr_object_id,
            )
            return False

    # Edge check
    for u, v in record.vnr.edges():
        if (u, v) in record.link_paths:
            path = record.link_paths[(u, v)]
        elif (v, u) in record.link_paths:
            path = record.link_paths[(v, u)]
        else:
            continue

        bw_req = float(record.vnr.edges[u, v].get('bw', 0.0))
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if substrate.has_edge(a, b):
                current_bw = float(substrate.edges[a, b].get('bw', 0.0))
            elif substrate.has_edge(b, a):
                current_bw = float(substrate.edges[b, a].get('bw', 0.0))
            else:
                logger.warning(
                    "_check_resources: substrate edge (%s,%s) missing for VNR obj_id=%d",
                    a, b, record.vnr_object_id,
                )
                return False
            if current_bw < bw_req:
                logger.warning(
                    "_check_resources: BW insufficient on edge (%s,%s) "
                    "(avail=%.2f, req=%.2f) for VNR obj_id=%d",
                    a, b, current_bw, bw_req, record.vnr_object_id,
                )
                return False
    return True


def _consume_resources(substrate: nx.Graph, record: VNRRecord) -> bool:
    """
    Deduct CPU and BW from substrate nodes/edges per the VNR mapping.

    Attribute names match graph_utils.py:
      node: 'cpu' (available)
      edge: 'bw'  (available)

    Pre-validates resource availability before ANY deduction so the
    substrate is never left in a partially-committed inconsistent state.

    Returns
    -------
    True  — resources successfully deducted.
    False — insufficient resources; substrate is NOT modified.
    """
    # --- Pre-flight check: bail out early if resources are insufficient ---
    if not _check_resources(substrate, record):
        logger.warning(
            "_consume_resources: pre-flight FAILED for VNR obj_id=%d; "
            "skipping commit to keep substrate consistent.",
            record.vnr_object_id,
        )
        return False

    # --- Safe to deduct (all checks passed) ---
    # Node resources
    for vnode, snode in record.mapping.items():
        cpu_req = float(record.vnr.nodes[vnode].get('cpu', 0.0))
        substrate.nodes[snode]['cpu'] = float(substrate.nodes[snode].get('cpu', 0.0)) - cpu_req

    # Link resources
    for u, v in record.vnr.edges():
        if (u, v) in record.link_paths:
            path = record.link_paths[(u, v)]
        elif (v, u) in record.link_paths:
            path = record.link_paths[(v, u)]
        else:
            continue

        bw_req = float(record.vnr.edges[u, v].get('bw', 0.0))
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if substrate.has_edge(a, b):
                substrate.edges[a, b]['bw'] = float(substrate.edges[a, b].get('bw', 0.0)) - bw_req
            elif substrate.has_edge(b, a):
                substrate.edges[b, a]['bw'] = float(substrate.edges[b, a].get('bw', 0.0)) - bw_req

    return True


def _release_resources(substrate: nx.Graph, record: VNRRecord) -> None:
    """
    Restore CPU and BW to substrate nodes/edges when a VNR expires.

    Clamps restored values to 'max_cpu' / 'max_bw' capacity to prevent
    floating-point accumulation from exceeding physical capacity.

    NOTE: Attribute names follow graph_utils.py / substrate_utilisation():
      capacity = 'max_cpu' (nodes), 'max_bw' (edges)
      NOT 'cpu_total' / 'bw_total' as stated in the original plan spec.
    """
    # Node resources
    for vnode, snode in record.mapping.items():
        if snode not in substrate.nodes:
            continue
        cpu_req = float(record.vnr.nodes[vnode].get('cpu', 0.0))
        current = float(substrate.nodes[snode].get('cpu', 0.0))
        # Use max_cpu or cpu_total as capacity cap; fall back to current + req if not set
        cpu_cap = float(substrate.nodes[snode].get(
            'max_cpu', substrate.nodes[snode].get('cpu_total', current + cpu_req)
        ))
        substrate.nodes[snode]['cpu'] = min(cpu_cap, current + cpu_req)

    # Link resources
    for u, v in record.vnr.edges():
        if (u, v) in record.link_paths:
            path = record.link_paths[(u, v)]
        elif (v, u) in record.link_paths:
            path = record.link_paths[(v, u)]
        else:
            continue

        bw_req = float(record.vnr.edges[u, v].get('bw', 0.0))
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if substrate.has_edge(a, b):
                current_bw = float(substrate.edges[a, b].get('bw', 0.0))
                bw_cap = float(substrate.edges[a, b].get(
                    'max_bw', substrate.edges[a, b].get('bw_total', current_bw + bw_req)
                ))
                substrate.edges[a, b]['bw'] = min(bw_cap, current_bw + bw_req)
            elif substrate.has_edge(b, a):
                current_bw = float(substrate.edges[b, a].get('bw', 0.0))
                bw_cap = float(substrate.edges[b, a].get(
                    'max_bw', substrate.edges[b, a].get('bw_total', current_bw + bw_req)
                ))
                substrate.edges[b, a]['bw'] = min(bw_cap, current_bw + bw_req)


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class StatefulSubstrateWrapper:
    """
    Intercepts substrate_fn inside VNEOrderingEnv to maintain a persistent,
    depleting substrate across training episodes.

    Key mechanisms
    --------------
    1. Persistent substrate — live_substrate never resets unless overflow fires.
    2. Post-step commit — on_step() reads env.accepted[-1] and applies
       resource deductions to live_substrate after each successful embedding.
    3. Lifetime expiry — on_episode_start() releases VNRs whose
       expiry_episode <= episode_count, mirroring BatchedVNRSimulator.
    4. Overflow protection — if CPU util > overflow_cpu_threshold, reset
       live_substrate to prevent the agent from being stuck in an
       unembeddable state.
    5. Warm-up — no commits for the first warmup_episodes episodes,
       so live_substrate stays at 0% util while the agent builds a
       basic policy.

    Parameters
    ----------
    env          : VNEOrderingEnv instance (already constructed)
    substrate_fn : the original substrate_fn callable (kept for overflow reset)
    cfg          : StatefulSubstrateConfig
    """

    def __init__(self, env: Any, substrate_fn: Any, cfg: StatefulSubstrateConfig):
        self.env = env
        self.original_substrate_fn = substrate_fn
        self.cfg = cfg

        # Initialise live_substrate from a fresh generation
        from src.utils.graph_utils import copy_substrate
        self._copy_substrate = copy_substrate

        self.live_substrate: nx.Graph = substrate_fn()
        # Fallback snapshot for non-regenerative overflow reset
        self.initial_substrate: nx.Graph = copy_substrate(self.live_substrate)

        # State
        self.committed_vnrs: List[VNRRecord] = []
        self.episode_count: int = 0
        self._committed_ids_this_episode: Set[int] = set()  # prevent double-commits

        # Telemetry
        self.live_cpu_util_history: List[float] = []

        # ── Patch the inner env ──────────────────────────────────────────────
        # VNEOrderingEnv.reset() calls self.substrate_fn() to get a new
        # substrate each episode. By replacing it here, every subsequent
        # env.reset() receives copy_substrate(live_substrate) instead of a
        # freshly generated graph.
        env.substrate_fn = self._patched_substrate_fn

    # ------------------------------------------------------------------
    # Patched substrate_fn
    # ------------------------------------------------------------------

    def _patched_substrate_fn(self) -> nx.Graph:
        """
        Returns a copy of the current live_substrate.
        This is what VNEOrderingEnv.reset() will call when constructing
        each episode. The copy ensures hpso_embed can mutate env.substrate
        without corrupting live_substrate.
        """
        return self._copy_substrate(self.live_substrate)

    # ------------------------------------------------------------------
    # Episode start hook
    # ------------------------------------------------------------------

    def on_episode_start(self) -> None:
        """
        Called BEFORE env.reset() at the start of each episode.

        Actions
        -------
        1. Increment episode counter
        2. If still in warm-up: return (live_substrate stays fresh, no commits yet)
        3. Expire committed VNRs whose expiry_episode <= episode_count
        4. Check overflow; reset live_substrate if threshold exceeded
        """
        self.episode_count += 1
        # Reset per-episode double-commit guard
        self._committed_ids_this_episode = set()

        cfg = self.cfg

        if self.episode_count <= cfg.warmup_episodes:
            # During warm-up, live_substrate has no commits yet → stays 0% util.
            # The patched substrate_fn returns its copy correctly; no action needed.
            return

        # --- 1. Expire committed VNRs ---
        remaining: List[VNRRecord] = []
        for record in self.committed_vnrs:
            if self.episode_count >= record.expiry_episode:
                _release_resources(self.live_substrate, record)
                logger.debug(
                    "Expired VNR (obj_id=%d) committed at episode %d",
                    record.vnr_object_id, record.commit_episode,
                )
            else:
                remaining.append(record)
        self.committed_vnrs = remaining

        # --- 2. Overflow check ---
        from src.utils.graph_utils import substrate_utilisation
        util = substrate_utilisation(self.live_substrate)
        cpu_util = util['cpu_util']
        self.live_cpu_util_history.append(cpu_util)

        if cpu_util > cfg.overflow_cpu_threshold:
            logger.warning(
                "OVERFLOW RESET at episode %d — live CPU util was %.1f%% "
                "(threshold=%.1f%%)",
                self.episode_count, cpu_util * 100, cfg.overflow_cpu_threshold * 100,
            )
            if cfg.regenerate_substrate_on_overflow:
                self.live_substrate = self.original_substrate_fn()
            else:
                self.live_substrate = self._copy_substrate(self.initial_substrate)
            self.committed_vnrs = []

    # ------------------------------------------------------------------
    # Step hook
    # ------------------------------------------------------------------

    def on_step(
        self,
        action: int,
        next_obs: dict,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        """
        Called IMMEDIATELY AFTER env.step() returns.

        If the step resulted in a successful embedding (env.last_success=True),
        reads env.accepted[-1] and commits the VNR's resource consumption
        to live_substrate.

        Guards
        ------
        - No-op during warm-up period.
        - Prevents double-commit if on_step is called twice for the same VNR
          (tracked via id(vnr) in _committed_ids_this_episode).
        """
        if self.episode_count <= self.cfg.warmup_episodes:
            return  # passive during warm-up

        if not self.env.last_success:
            return  # VNR was rejected; nothing to commit

        if not self.env.accepted:
            return  # safety guard

        vnr, mapping, link_paths = self.env.accepted[-1]

        # Prevent double-commit within the same episode
        obj_id = id(vnr)
        if obj_id in self._committed_ids_this_episode:
            logger.debug(
                "on_step: VNR obj_id=%d already committed this episode; skipping",
                obj_id,
            )
            return

        record = VNRRecord(
            vnr            = vnr,
            mapping        = mapping,
            link_paths     = link_paths,
            commit_episode = self.episode_count,
            expiry_episode = self.episode_count + self.cfg.vnr_lifetime_episodes,
            vnr_object_id  = obj_id,
        )

        committed = _consume_resources(self.live_substrate, record)
        if committed:
            self.committed_vnrs.append(record)
            self._committed_ids_this_episode.add(obj_id)
            logger.debug(
                "Committed VNR obj_id=%d at episode %d (expires ep %d); "
                "total committed=%d",
                obj_id, record.commit_episode, record.expiry_episode,
                len(self.committed_vnrs),
            )
        else:
            # Resource pre-flight failed — do NOT track this VNR so its
            # resources are never double-released on expiry either.
            logger.warning(
                "on_step: VNR obj_id=%d SKIPPED (infeasible on live_substrate "
                "at episode %d). live_substrate may be more depleted than "
                "env.substrate; this VNR will not be tracked for expiry.",
                obj_id, self.episode_count,
            )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_live_util(self) -> dict:
        """
        Returns current utilisation stats for TensorBoard logging.

        Returns
        -------
        dict with keys:
            live_cpu_util  : float  [0, 1]
            live_bw_util   : float  [0, 1]
            committed_vnrs : int
            episode        : int
        """
        from src.utils.graph_utils import substrate_utilisation
        util = substrate_utilisation(self.live_substrate)
        return {
            'live_cpu_util':  util['cpu_util'],
            'live_bw_util':   util.get('bw_util', 0.0),
            'committed_vnrs': len(self.committed_vnrs),
            'episode':        self.episode_count,
        }
