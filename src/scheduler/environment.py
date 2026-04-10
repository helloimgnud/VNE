"""
src/scheduler/environment.py
============================
Gymnasium-compatible RL environment for VNR ordering.

MDP formulation (see network_encoder_rl.md §6):
  State  s_t  = (substrate_t, remaining_vnr_list)
  Action a_t  = index i ∈ [0, |remaining|) → pick VNR_i as next to embed
  Transition  = run hpso_embed(substrate_t, VNR_i)
                → update substrate_t (resources consumed in-place)
                → remove VNR_i from remaining list
  Reward r_t  = pluggable (see src/scheduler/rewards.py)
  Terminal    = remaining list is empty

The environment supports three reward modes:
  "simple"   — Phase 1 (acceptance ±)
  "revenue"  — Phase 2 (R/C per step)
  "longterm" — Phase 3 (R/C + terminal AR bonus)

Integration with existing code
-------------------------------
hpso_embed is imported from src.algorithms.fast_hpso (unchanged).
copy_substrate is imported from src.utils.graph_utils (unchanged).
Feature converters come from src.scheduler.features.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium
import numpy as np

from src.algorithms.fast_hpso import hpso_embed
from src.utils.graph_utils import copy_substrate
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
from src.scheduler.rewards import RewardMode, compute_reward


# ---------------------------------------------------------------------------
# Default HPSO hyper-parameters
# ---------------------------------------------------------------------------

_DEFAULT_HPSO = dict(
    particles    = 20,
    iterations   = 30,
    w_max        = 0.9,
    w_min        = 0.5,
    beta         = 0.3,
    gamma        = 0.3,
    T0           = 100,
    cooling_rate = 0.95,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class VNEOrderingEnv(gymnasium.Env):
    """
    Gymnasium environment for VNR ordering.

    Each episode processes one batch of VNRs on one substrate. The agent
    repeatedly picks which VNR to embed next; HPSO performs the actual
    node/link mapping and updates substrate resources in-place.

    Parameters
    ----------
    substrate_fn : callable() → networkx.Graph
        Returns a **fresh** substrate graph for each new episode.
        The env calls ``copy_substrate`` internally so the original is
        never mutated.
    batch_fn : callable() → list[networkx.Graph]
        Returns a fresh list of VNR graphs for each episode.
    hpso_params : dict, optional
        Override default HPSO hyper-parameters.
    reward_mode : str | RewardMode
        One of "simple", "revenue", "longterm".
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        substrate_fn: Callable,
        batch_fn:     Callable,
        hpso_params:  Optional[dict]             = None,
        reward_mode:  RewardMode | str           = RewardMode.SIMPLE,
    ):
        super().__init__()
        self.substrate_fn = substrate_fn
        self.batch_fn     = batch_fn
        self.hpso_params  = {**_DEFAULT_HPSO, **(hpso_params or {})}
        self.reward_mode  = RewardMode(reward_mode) if isinstance(reward_mode, str) \
                            else reward_mode

        # --- Episode state (set in reset) ---
        self.substrate:    Any           = None
        self.vnr_list:     List          = []
        self.remaining:    List[int]     = []  # indices into vnr_list
        self.accepted:     List[Tuple]   = []  # (vnr, mapping, link_paths)
        self.rejected:     List          = []  # [vnr]
        self.last_success: bool          = False

        # --- Gymnasium spaces (symbolic; actual obs are PyG dicts) ---
        # SB3 requires concrete spaces; we use minimal placeholders.
        # The custom REINFORCE loop bypasses these entirely.
        self.observation_space = gymnasium.spaces.Dict({
            "n_remaining": gymnasium.spaces.Discrete(1024),
        })
        self.action_space = gymnasium.spaces.Discrete(1)  # Updated dynamically in step

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Start a new episode.

        Returns
        -------
        obs  : dict with keys "substrate", "vnr_list", "n_remaining"
        info : {}
        """
        super().reset(seed=seed)

        raw_substrate   = self.substrate_fn()
        self.substrate  = copy_substrate(raw_substrate)

        self.vnr_list   = self.batch_fn()
        self.remaining  = list(range(len(self.vnr_list)))
        self.accepted   = []
        self.rejected   = []
        self.accepted_costs = []  # track real costs
        self.last_success = False
        self.last_step_cost = None

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        Process the VNR at ``self.remaining[action]``.

        Parameters
        ----------
        action : int
            Index into `self.remaining` (not into `self.vnr_list` directly).

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if not self.remaining:
            raise RuntimeError("step() called on a finished episode. Call reset() first.")

        vnr_idx = self.remaining[action]
        vnr     = self.vnr_list[vnr_idx]

        result = hpso_embed(
            substrate_graph = self.substrate,
            vnr_graph       = vnr,
            **self.hpso_params,
        )

        if result is not None:
            mapping, link_paths = result
            self.accepted.append((vnr, mapping, link_paths))
            self.last_success = True
            
            from src.evaluation.eval import cost_of_embedding
            cost = cost_of_embedding(mapping, link_paths, vnr, self.substrate)
            self.accepted_costs.append(cost)
            self.last_step_cost = cost
        else:
            self.rejected.append(vnr)
            self.last_success = False
            self.last_step_cost = None

        self.remaining.pop(action)
        done = (len(self.remaining) == 0)
        
        from src.utils.graph_utils import substrate_utilisation
        sub_util = substrate_utilisation(self.substrate)

        reward = compute_reward(
            mode     = self.reward_mode,
            success  = self.last_success,
            vnr      = vnr,
            done     = done,
            accepted = self.accepted,
            rejected = self.rejected,
            step_cost=self.last_step_cost,
            accepted_costs=self.accepted_costs,
            substrate_util=sub_util,
        )

        obs  = self._get_obs()
        info = {
            "accepted":  len(self.accepted),
            "rejected":  len(self.rejected),
            "n_remaining": len(self.remaining),
            "substrate_util": sub_util,
        }
        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        """
        Build the observation dict.

        Returns a dict with three keys:
          "substrate"  → PyG Data  (single substrate graph)
          "vnr_list"   → list of PyG Data  (remaining VNRs only)
          "n_remaining"→ int
        """
        return {
            "substrate":   substrate_to_pyg(self.substrate),
            "vnr_list":    [vnr_to_pyg(self.vnr_list[i]) for i in self.remaining],
            "n_remaining": len(self.remaining),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def episode_summary(self) -> dict:
        """Return summary statistics for the current (or last) episode."""
        n_total = len(self.accepted) + len(self.rejected)
        ar  = len(self.accepted) / (n_total + 1e-9)

        from src.evaluation.eval import revenue_of_vnr
        total_rev  = sum(revenue_of_vnr(v) for v, _, _ in self.accepted)
        total_cost = sum(self.accepted_costs) if self.accepted_costs else 1e-6
        rc  = total_rev / total_cost

        from src.utils.graph_utils import substrate_utilisation
        u = substrate_utilisation(self.substrate)

        return dict(
            n_total    = n_total,
            n_accepted = len(self.accepted),
            n_rejected = len(self.rejected),
            acc_rate   = ar,
            total_rev  = total_rev,
            total_cost = total_cost,
            rc_ratio   = rc,
            cpu_util   = u['cpu_util'],
            bw_util    = u['bw_util'],
        )
