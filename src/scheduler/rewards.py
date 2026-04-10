"""
src/scheduler/rewards.py
========================
Pluggable reward functions for the VNEOrderingEnv.

Three reward modes are supported (see network_encoder_rl.md §6.2):

  RewardMode.SIMPLE    — +1.0 on accept, -0.5 on reject.
                         Used in Phase 1 to validate that the network can
                         learn basic acceptance maximisation.

  RewardMode.REVENUE   — revenue / cost on accept; -revenue*0.1 on reject.
                         Adds resource-efficiency signal (Phase 2).

  RewardMode.LONGTERM  — per-step revenue (if accepted) + terminal bonus:
                             ar * 5.0 + rc * 2.0
                         Propagates credit assignment for the whole batch
                         using PPO's GAE (Phase 3).

Plugin design
-------------
``compute_reward(mode, last_result, vnr, done, accepted, rejected)``
accepts one of the three modes and dispatches to the appropriate function.
New reward modes can be added by:
  1. Adding a value to ``RewardMode``
  2. Implementing a ``_reward_<name>`` function below
  3. Adding the new entry to ``_REWARD_FNS``
"""

from __future__ import annotations

import enum
from typing import List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class RewardMode(str, enum.Enum):
    SIMPLE   = "simple"
    REVENUE  = "revenue"
    LONGTERM = "longterm"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _revenue(vnr: nx.Graph) -> float:
    """Total CPU + BW demand of a VNR (used as revenue proxy)."""
    cpu = sum(float(vnr.nodes[n].get("cpu", 0.0)) for n in vnr.nodes())
    bw  = sum(float(vnr.edges[e].get("bw",  0.0)) for e in vnr.edges())
    return cpu + bw


def _cost(vnr: nx.Graph) -> float:
    """Lightweight cost proxy: same as revenue (demand-based)."""
    return _revenue(vnr) + 1e-6


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _reward_simple(
    success:  bool,
    vnr:      nx.Graph,
    done:     bool,
    accepted: list,
    rejected: list,
) -> float:
    return 1.0 if success else -0.5


def _reward_revenue(
    success:  bool,
    vnr:      nx.Graph,
    done:     bool,
    accepted: list,
    rejected: list,
) -> float:
    if success:
        rev  = _revenue(vnr)
        cost = _cost(vnr)
        return rev / cost
    else:
        return -_revenue(vnr) * 0.1


def _reward_longterm(
    success:  bool,
    vnr:      nx.Graph,
    done:     bool,
    accepted: list,
    rejected: list,
) -> float:
    step_r = _revenue(vnr) if success else 0.0

    if done:
        total_rev  = sum(_revenue(v) for v, _, _ in accepted)
        total_cost = sum(_cost(v)    for v, _, _ in accepted)
        n_total    = len(accepted) + len(rejected)
        ar = len(accepted) / (n_total + 1e-9)
        rc = total_rev / (total_cost + 1e-6)
        step_r += ar * 5.0 + rc * 2.0

    return step_r


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_REWARD_FNS = {
    RewardMode.SIMPLE:   _reward_simple,
    RewardMode.REVENUE:  _reward_revenue,
    RewardMode.LONGTERM: _reward_longterm,
}


def compute_reward(
    mode:     RewardMode | str,
    success:  bool,
    vnr:      nx.Graph,
    done:     bool,
    accepted: list,
    rejected: list,
) -> float:
    """
    Compute step reward using the selected reward mode.

    Parameters
    ----------
    mode     : one of RewardMode.SIMPLE / REVENUE / LONGTERM  (or str value)
    success  : whether the latest hpso_embed call succeeded
    vnr      : the VNR that was just processed
    done     : True if no VNRs remain in the episode
    accepted : list of (vnr, mapping, link_paths) accumulated so far
    rejected : list of VNR graphs that were rejected so far

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
    return fn(success, vnr, done, accepted, rejected)
