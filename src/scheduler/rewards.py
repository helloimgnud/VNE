"""
src/scheduler/rewards.py
========================
Pluggable reward functions for the VNEOrderingEnv.

Reward modes supported (see network_encoder_rl.md §6.2):

  RewardMode.SIMPLE          — +1.0 on accept, -0.5 on reject.
                               Used in Phase 1 to validate that the network
                               can learn basic acceptance maximisation.

  RewardMode.R2C_AC          — Revenue/Cost per step **plus** a terminal
                               acceptance-rate bonus (ar * 3.0 + rc * 2.0).
                               Balances efficiency AND throughput from Level 2.

  RewardMode.REVENUE         — revenue / cost on accept; -revenue*0.1 on
                               reject. Pure efficiency signal (legacy Phase 2).

  RewardMode.LONGTERM        — per-step revenue (if accepted) + terminal bonus:
                                   ar * 5.0 + rc * 2.0
                               Propagates credit assignment for the whole batch
                               using PPO's GAE (Phase 3).

  RewardMode.CONGESTION_AWARE — R/C scaled by substrate CPU fill.

  RewardMode.REJECTION_SCALED — R/C on accept; heavy, fill-proportional
                                rejection penalty.

Plugin design
-------------
``compute_reward(mode, last_result, vnr, done, accepted, rejected)``
accepts one of the modes and dispatches to the appropriate function.
New reward modes can be added by:
  1. Adding a value to ``RewardMode``
  2. Implementing a ``_reward_<name>`` function below
  3. Adding the new entry to ``_REWARD_FNS``
"""

from __future__ import annotations
from src.evaluation.eval import cost_of_vnr

import enum
from typing import List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class RewardMode(str, enum.Enum):
    SIMPLE           = "simple"
    R2C_AC           = "r2c_ac"          # Revenue/Cost + Acceptance-Rate blend
    REVENUE          = "revenue"
    LONGTERM         = "longterm"
    CONGESTION_AWARE = "congestion_aware"
    REJECTION_SCALED = "rejection_scaled"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _revenue(vnr: nx.Graph) -> float:
    """Total CPU + BW demand of a VNR (used as revenue proxy)."""
    cpu = sum(float(vnr.nodes[n].get("cpu", 0.0)) for n in vnr.nodes())
    bw  = sum(float(vnr.edges[e].get("bw",  0.0)) for e in vnr.edges())
    return cpu + bw


def _demand_cost(vnr: nx.Graph) -> float:
    """Lightweight cost proxy (same as revenue — used only as fallback)."""
    return cost_of_vnr(vnr) + 1e-6


def _real_rc(vnr: nx.Graph, real_step_cost: Optional[float]) -> float:
    """Compute per-step R/C ratio using real embedding cost."""
    rev = _revenue(vnr)
    if real_step_cost is not None and real_step_cost > 1e-9:
        return rev / real_step_cost
    return rev / _demand_cost(vnr)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _reward_simple(
    success: bool, vnr: nx.Graph, done: bool, accepted: list, rejected: list,
    step_cost: Optional[float] = None, accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None
) -> float:
    return 1.0 if success else -0.5


def _reward_r2c_ac(
    success: bool, vnr: nx.Graph, done: bool, accepted: list, rejected: list,
    step_cost: Optional[float] = None, accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None
) -> float:
    """Convex blend of normalised R/C and running Acceptance Rate.

    Formulation
    -----------
      Accept:  r = α · r2c_norm  +  (1-α) · inline_ar
      Reject:  r = -(1-α) · (1 - inline_ar)

    where
      r2c_norm  = r2c / (1 + r2c)   squashes [0, ∞) → [0, 1)   monotone-preserving
      inline_ar = n_accepted / n_total  running AR at this step  ∈ [0, 1]
      α         = 0.5               equal weight; tune toward 1.0 to prioritise efficiency

    Reward range (α = 0.5)
    ----------------------
      Accept: [0,  1)    always non-negative, bounded
      Reject: [-0.5, 0]  penalty grows as current AR worsens
      Full:   [-0.5, 1)

    Why no terminal bonus?
    ----------------------
    inline_ar already injects AR pressure at every step, so a terminal spike
    is redundant and would reintroduce the return-scale explosion.
    GAE (γ=0.99, λ=0.95) propagates the accumulated per-step AR signal
    backward through time naturally.
    """
    _ALPHA = 0.5   # tune: higher → care more about efficiency, lower → care more about AR

    # Note: accepted / rejected already include the current VNR —
    # env.step() appends before calling compute_reward.
    n_accepted = len(accepted)
    n_total    = n_accepted + len(rejected)
    inline_ar  = n_accepted / (n_total + 1e-9)

    if success:
        rc      = _real_rc(vnr, step_cost)   # raw R/C ∈ [0, ∞)
        rc_norm = rc / (1.0 + rc)            # squashed ∈ [0, 1)
        reward  = _ALPHA * rc_norm + (1.0 - _ALPHA) * inline_ar
    else:
        # Penalty proportional to how much AR is currently suffering
        reward = -(1.0 - _ALPHA) * (1.0 - inline_ar)

    return reward




def _reward_revenue(
    success: bool, vnr: nx.Graph, done: bool, accepted: list, rejected: list,
    step_cost: Optional[float] = None, accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None
) -> float:
    if success:
        return _real_rc(vnr, step_cost)
    else:
        return -_revenue(vnr) * 0.1


def _reward_longterm(
    success: bool, vnr: nx.Graph, done: bool, accepted: list, rejected: list,
    step_cost: Optional[float] = None, accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None
) -> float:
    step_r = _revenue(vnr) if success else 0.0

    if done:
        total_rev = sum(_revenue(v) for v, _, _ in accepted)
        if accepted_costs and len(accepted_costs) == len(accepted):
            total_cost = sum(accepted_costs)
        else:
            total_cost = sum(_demand_cost(v) for v, _, _ in accepted)
            
        n_total = len(accepted) + len(rejected)
        ar = len(accepted) / (n_total + 1e-9)
        rc = total_rev / (total_cost + 1e-6)
        step_r += ar * 5.0 + rc * 2.0

    return step_r


def _reward_congestion_aware(
    success: bool, vnr: nx.Graph, done: bool, accepted: list, rejected: list,
    step_cost: Optional[float] = None, accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None
) -> float:
    """Scales R/C reward up when the substrate is congested."""
    cpu_fill = substrate_util.get("cpu_util", 0.0) if substrate_util else 0.0
    if success:
        rc = _real_rc(vnr, step_cost)
        return rc * (1.0 + cpu_fill)
    else:
        return -_revenue(vnr) * 0.2 * (1.0 + cpu_fill)


def _reward_rejection_scaled(
    success: bool, vnr: nx.Graph, done: bool, accepted: list, rejected: list,
    step_cost: Optional[float] = None, accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None
) -> float:
    """Heavy rejection penalty proportional to substrate fill."""
    cpu_fill = substrate_util.get("cpu_util", 0.0) if substrate_util else 0.0
    if success:
        return _real_rc(vnr, step_cost)
    else:
        return -_revenue(vnr) * (0.1 + 0.5 * cpu_fill)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_REWARD_FNS = {
    RewardMode.SIMPLE:           _reward_simple,
    RewardMode.R2C_AC:           _reward_r2c_ac,
    RewardMode.REVENUE:          _reward_revenue,
    RewardMode.LONGTERM:         _reward_longterm,
    RewardMode.CONGESTION_AWARE: _reward_congestion_aware,
    RewardMode.REJECTION_SCALED: _reward_rejection_scaled,
}


def compute_reward(
    mode:     RewardMode | str,
    success:  bool,
    vnr:      nx.Graph,
    done:     bool,
    accepted: list,
    rejected: list,
    step_cost: Optional[float] = None,
    accepted_costs: Optional[List[float]] = None,
    substrate_util: Optional[dict] = None,
) -> float:
    """
    Compute step reward using the selected reward mode.

    Parameters
    ----------
    mode           : one of RewardMode.SIMPLE / REVENUE / LONGTERM  (or str value)
    success        : whether the latest hpso_embed call succeeded
    vnr            : the VNR that was just processed
    done           : True if no VNRs remain in the episode
    accepted       : list of (vnr, mapping, link_paths) accumulated so far
    rejected       : list of VNR graphs that were rejected so far
    step_cost      : real embedding cost of this VNR
    accepted_costs : list of real embedding costs for all accepted VNRs
    substrate_util : dictionary with resource utilization stats

    Returns
    -------
    float : step reward
    """
    key = RewardMode(mode) if isinstance(mode, str) else mode
    fn  = _REWARD_FNS.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown reward mode '{mode}'. "
            f"Choose from {[m.value for m in RewardMode]}."
        )
    return fn(success, vnr, done, accepted, rejected, 
              step_cost=step_cost, accepted_costs=accepted_costs, 
              substrate_util=substrate_util)
