"""
src/rl/environment_v2.py
========================
PPO v2 Training Environment — Stateful, Window-Based VNE Simulation.

Episode = 1 full pass through all N VNRs on a SINGLE persistent substrate.
The substrate resets to fresh ONLY between episodes (epochs), not between windows.
Leave events (VNR expiry) are processed at the START of each time window — exactly
matching hpso_batch_scheduler.py inference behaviour.

Data mode: PREGENERATED only.
  - substrate_path / vnr_path are loaded ONCE in __init__.
  - reset() creates a fresh copy of substrate and replays the same VNR stream.

Usage:
    env = VNEEnvironmentV2(
        substrate_path="dataset/rl_training/train/substrate.json",
        vnr_path="dataset/rl_training/train/vnr_stream.json",
        window_size=50,
    )
    obs, _ = env.reset()
    while True:
        action = agent.act(obs)           # index into obs["vnr_list"]
        obs, reward, done, _, info = env.step(action)
        if done:
            summary = env.episode_summary()
            break
"""

from __future__ import annotations

import copy
import enum
from typing import Dict, List, Optional, Tuple

import gymnasium
import networkx as nx
import torch

from src.algorithms.fast_hpso import hpso_embed
from src.evaluation.eval import cost_of_embedding, revenue_of_vnr
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
from src.utils.graph_utils import copy_substrate, release_vnr_embedding
from src.utils.io_utils import load_substrate_from_json, load_vnr_stream_from_json


# ---------------------------------------------------------------------------
# DataMode
# ---------------------------------------------------------------------------

class DataMode(str, enum.Enum):
    PREGENERATED = "pregenerated"


# ---------------------------------------------------------------------------
# VNEEnvironmentV2
# ---------------------------------------------------------------------------

class VNEEnvironmentV2(gymnasium.Env):
    """
    PPO environment for VNR ordering (v2 pipeline).

    Episode = 1 pass through all N VNRs on a single substrate.
    Substrate resets to fresh after each episode (epoch).
    Leave events are processed at the start of each time window (same as inference).

    Data modes:
      - PREGENERATED: load from substrate_path / vnr_path (list of nx.Graph)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        # --- Data ---
        substrate_path: str,
        vnr_path: str,
        # --- Simulation params ---
        window_size: int = 10,
        max_queue_delay: int = 100,
        # --- HPSO params ---
        hpso_params: Optional[dict] = None,
    ):
        super().__init__()

        assert substrate_path is not None, "substrate_path is required"
        assert vnr_path is not None, "vnr_path is required"

        # Load once at startup (can be changed via load_dataset)
        self.load_dataset(substrate_path, vnr_path)

        self.window_size = window_size
        self.max_queue_delay = max_queue_delay
        self.hpso_params: dict = hpso_params or {}

        # Episode state — initialised properly in reset()
        self.substrate: Optional[nx.Graph] = None
        self.substrate_original: Optional[nx.Graph] = None
        self.active_embeddings: List[Tuple] = []      # (vnr, mapping, link_paths, expiry)
        self.vnr_queue: List[nx.Graph] = []
        self.episode_accepted: List[Tuple] = []       # (vnr, mapping, link_paths)
        self.episode_rejected: List[nx.Graph] = []
        self.episode_accepted_costs: List[float] = []
        self.episode_rewards: List[float] = []

        self._all_windows: List[List[nx.Graph]] = []
        self.window_idx: int = 0
        self.total_windows: int = 0

        # Action / observation spaces — set lazily; gymnasium requires them
        # but our obs is a dict of PyG Data objects (non-standard).
        self.action_space = gymnasium.spaces.Discrete(1)       # placeholder
        self.observation_space = gymnasium.spaces.Discrete(1)  # placeholder

    # ------------------------------------------------------------------
    # Core gymnasium interface
    # ------------------------------------------------------------------

    def load_dataset(self, substrate_path: str, vnr_path: str):
        """Hot-swap the dataset to train on a different replica."""
        self._substrate_original = load_substrate_from_json(substrate_path)
        self._vnr_stream_raw = load_vnr_stream_from_json(vnr_path)

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        Always: fresh substrate copy + full (same) VNR stream.
        Should only be called after done=True from a previous episode.
        """
        super().reset(seed=seed)

        # Fresh working copy — original untouched
        self.substrate = copy_substrate(self._substrate_original)
        self.substrate_original = self._substrate_original

        # Reset episode accumulators
        self.active_embeddings = []
        self.episode_accepted = []
        self.episode_rejected = []
        self.episode_accepted_costs = []
        self.episode_rewards = []

        # Partition the (immutable) VNR stream into time windows
        self._all_windows = self._partition_into_windows(self._vnr_stream_raw)
        self.window_idx = 0
        self.total_windows = len(self._all_windows)

        # Advance to window 0 (processes leave events + loads VNR queue)
        self._advance_to_window(0)

        # Handle edge case: first window may be empty — skip ahead
        while not self.vnr_queue and self.window_idx + 1 < self.total_windows:
            self.window_idx += 1
            self._advance_to_window(self.window_idx)

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Action = index into self.vnr_queue (local, not global).

        Selects the VNR at position `action`, runs HPSO to attempt embedding,
        then optionally advances the window if the queue is empty.

        done=True only when ALL windows across ALL N VNRs are exhausted.
        """
        assert self.vnr_queue, "step() called with an empty vnr_queue — bug in env or trainer"
        assert 0 <= action < len(self.vnr_queue), (
            f"action {action} out of range [0, {len(self.vnr_queue)})"
        )

        # Pop chosen VNR
        vnr: nx.Graph = self.vnr_queue.pop(action)

        # --- HPSO Embedding ---
        # result = hpso_embed(
        #     substrate_graph=self.substrate,
        #     vnr_graph=vnr,
        #     **self.hpso_params,
        # )

        _embed_fn = self.hpso_params.pop('embed_fn', None)

        if _embed_fn is not None:
            # batch-style: wrap single VNR as a batch of 1
            accepted, _ = _embed_fn(self.substrate, [(vnr, None)])
            result = (accepted[0][1], accepted[0][2]) if accepted else None
        else:
            from src.algorithms.fast_hpso import hpso_embed
            result = hpso_embed(
                substrate_graph=self.substrate,
                vnr_graph=vnr,
                **self.hpso_params,
            )

        if result is not None:
            mapping, link_paths = result
            expiry = vnr.graph["arrival_time"] + vnr.graph["lifetime"]
            self.active_embeddings.append((vnr, mapping, link_paths, expiry))
            self.episode_accepted.append((vnr, mapping, link_paths))
            cost = cost_of_embedding(mapping, link_paths, vnr, self.substrate)
            self.episode_accepted_costs.append(cost)
            success = True
        else:
            self.episode_rejected.append(vnr)
            cost = None
            success = False

        # --- Check/advance window ---
        done = self._check_done_and_advance()

        # --- Reward ---
        reward = self._compute_reward(success, vnr, done, cost)
        self.episode_rewards.append(reward)

        return self._get_obs(), reward, done, False, self._get_info(success)

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _partition_into_windows(self, vnr_stream: List[nx.Graph]) -> List[List[nx.Graph]]:
        """
        Partition the VNR stream into time windows.

        Returns list[list[nx.Graph]], window i contains VNRs with
        arrival_time ∈ [i * window_size, (i+1) * window_size).

        Empty windows are kept (for leave-event processing) but trailing
        empty windows are pruned.
        """
        if not vnr_stream:
            return [[]]

        max_time = max(v.graph["arrival_time"] for v in vnr_stream)
        n_windows = int(max_time / self.window_size) + 1

        windows: List[List[nx.Graph]] = [[] for _ in range(n_windows)]
        for vnr in vnr_stream:
            w_idx = int(vnr.graph["arrival_time"] / self.window_size)
            windows[w_idx].append(vnr)

        # Prune trailing empty windows (keep at least 1)
        while len(windows) > 1 and not windows[-1]:
            windows.pop()

        return windows

    def _advance_to_window(self, window_idx: int):
        """
        Transition to window `window_idx`.

        Order (CRITICAL — must match inference):
          1. Expire VNRs whose lifetime ended before this window starts.
          2. Load VNRs for this window.
          3. Drop any VNR that has exceeded max_queue_delay.
        """
        window_start = window_idx * self.window_size
        window_end   = (window_idx + 1) * self.window_size

        # --- Step 1: Expire embeddings ---
        still_active = []
        for emb in self.active_embeddings:
            vnr, mapping, link_paths, expiry = emb
            if expiry < window_start:
                release_vnr_embedding(self.substrate, vnr, mapping, link_paths)
            else:
                still_active.append(emb)
        self.active_embeddings = still_active

        # --- Step 2: Load new VNRs for this window ---
        if window_idx < len(self._all_windows):
            new_vnrs = self._all_windows[window_idx]
        else:
            new_vnrs = []

        # --- Step 3: Build vnr_queue, drop stale VNRs ---
        self.vnr_queue = []
        for vnr in new_vnrs:
            wait_time = window_start - vnr.graph["arrival_time"]
            if wait_time <= self.max_queue_delay:
                self.vnr_queue.append(vnr)
            # else: VNR waited too long in queue → drop silently

        self.window_expiry_time = window_end

    def _check_done_and_advance(self) -> bool:
        """
        If the current vnr_queue is empty, advance to the next window.
        Keeps advancing through empty windows until VNRs are found or
        all windows are exhausted (done=True).
        """
        if self.vnr_queue:
            return False  # VNRs still remain in this window

        # Current window exhausted — try to advance
        self.window_idx += 1

        if self.window_idx >= self.total_windows:
            return True  # All windows done

        self._advance_to_window(self.window_idx)

        # Skip empty windows
        while not self.vnr_queue:
            self.window_idx += 1
            if self.window_idx >= self.total_windows:
                return True
            self._advance_to_window(self.window_idx)

        return False

    # ------------------------------------------------------------------
    # Observation / reward / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        """
        Returns a dict:
          "substrate" : PyG Data  [1 graph]
          "vnr_list"  : list of PyG Data  [remaining VNRs in current window]
        """
        sub_pyg  = substrate_to_pyg(self.substrate)
        vnr_pygs = [vnr_to_pyg(v) for v in self.vnr_queue]
        return {"substrate": sub_pyg, "vnr_list": vnr_pygs}

    def _compute_reward(
        self,
        success: bool,
        vnr: nx.Graph,
        done: bool,
        cost: Optional[float],
        alpha: float = 0.5,
        lambda_ar: float = 0.4,
        lambda_rc: float = 0.6,
    ) -> float:
        """
        Per-step reward (dense) + terminal bonus (when done=True).

        Per-step:
          - Accept: α * rc_norm + (1-α) * inline_ar
          - Reject: -(1-α) * (1 - inline_ar)

        Terminal (added on done):
          λ_ar * AR + λ_rc * RC_clipped
        """
        n_accepted = len(self.episode_accepted)
        n_rejected = len(self.episode_rejected)
        n_total    = n_accepted + n_rejected
        inline_ar  = n_accepted / max(n_total, 1)

        if success and cost is not None and cost > 0:
            rev      = revenue_of_vnr(vnr)
            rc       = rev / cost
            rc_norm  = rc / (1.0 + rc)          # squash to [0, 1)
            r_step   = alpha * rc_norm + (1.0 - alpha) * inline_ar
        else:
            r_step   = -(1.0 - alpha) * (1.0 - inline_ar)

        r_terminal = 0.0
        if done:
            total_rev  = sum(revenue_of_vnr(v) for v, _, _ in self.episode_accepted)
            total_cost = sum(self.episode_accepted_costs) if self.episode_accepted_costs else 1e-6
            ar  = n_accepted / max(n_total, 1)
            rc  = min(total_rev / max(total_cost, 1e-6), 1.0)
            r_terminal = lambda_ar * ar + lambda_rc * rc

        return r_step + r_terminal

    def _get_info(self, success: bool) -> Dict:
        n_acc = len(self.episode_accepted)
        n_rej = len(self.episode_rejected)
        return {
            "success":    success,
            "n_accepted": n_acc,
            "n_rejected": n_rej,
            "window_idx": self.window_idx,
        }

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def episode_summary(self) -> Dict:
        """Return end-of-episode statistics."""
        n_accepted = len(self.episode_accepted)
        n_rejected = len(self.episode_rejected)
        n_total    = n_accepted + n_rejected

        total_rev  = sum(revenue_of_vnr(v) for v, _, _ in self.episode_accepted)
        total_cost = sum(self.episode_accepted_costs) if self.episode_accepted_costs else 1e-6

        return {
            "acceptance_rate":    n_accepted / max(n_total, 1),
            "revenue_cost_ratio": min(total_rev / max(total_cost, 1e-6), 2.0),
            "n_accepted":         n_accepted,
            "n_rejected":         n_rejected,
            "n_total":            n_total,
            "total_reward":       sum(self.episode_rewards),
        }

    # ------------------------------------------------------------------
    # Substrate statistics (for TensorBoard)
    # ------------------------------------------------------------------

    def substrate_utilisation(self) -> Dict[str, float]:
        """CPU and BW utilisation of the working substrate."""
        sub = self.substrate
        if sub is None:
            return {"cpu_util": 0.0, "bw_util": 0.0}

        orig = self._substrate_original

        total_cpu  = sum(orig.nodes[n].get("cpu_total", orig.nodes[n].get("cpu", 1e-9))
                         for n in orig.nodes)
        avail_cpu  = sum(sub.nodes[n].get("cpu", 0.0) for n in sub.nodes)

        total_bw   = sum(orig.edges[u, v].get("bw_total", orig.edges[u, v].get("bw", 1e-9))
                         for u, v in orig.edges)
        avail_bw   = sum(sub.edges[u, v].get("bw", 0.0) for u, v in sub.edges)

        return {
            "cpu_util": 1.0 - avail_cpu / max(total_cpu, 1e-9),
            "bw_util":  1.0 - avail_bw  / max(total_bw,  1e-9),
        }
