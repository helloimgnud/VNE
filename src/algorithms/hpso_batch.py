"""
src/algorithms/hpso_batch.py
============================
Batch HPSO embedding with optional RL-based VNR ordering.

Original behaviour
------------------
Sort VNRs by revenue in descending order (greedy heuristic), then embed
sequentially with HPSO.

Extended behaviour (rl_agent provided)
---------------------------------------
Instead of the revenue-sort heuristic, a trained VNRSchedulerAgent
(src/rl.VNRSchedulerAgent) decides the processing order.  The agent uses
a Graph Pointer Network (GAT-based, adapted from GPRL) to select VNRs
based on the current substrate state and the pending queue.

Usage
-----
# Original (no RL, backward-compatible):
    accepted, rejected = hpso_embed_batch(substrate, batch)

# With RL agent:
    from src.rl import VNRSchedulerAgent
    agent = VNRSchedulerAgent.load_from_checkpoint('result/scheduler_model.pt')
    accepted, rejected = hpso_embed_batch(substrate, batch, rl_agent=agent)

The function signature intentionally keeps `rl_agent=None` first so all
existing call-sites continue to work without modification.
"""

import copy
from typing import List, Optional, Tuple

from src.algorithms.fast_hpso import hpso_embed
from src.evaluation.eval import revenue_of_vnr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch) -> list:
    """Accept both plain VNR lists and [(vnr, info), ...] tuples."""
    if len(batch) > 0 and isinstance(batch[0], tuple):
        return [vnr for vnr, _ in batch]
    return list(batch)


def _sort_by_revenue(vnr_list: list) -> List[int]:
    """Original heuristic: revenue-descending order. Returns index list."""
    indexed = sorted(
        range(len(vnr_list)),
        key=lambda i: revenue_of_vnr(vnr_list[i]),
        reverse=True,
    )
    return indexed


def _rl_order(
    vnr_list: list,
    substrate,
    agent,
    build_sub_fn=None,
    build_vnr_fn=None,
) -> List[int]:
    """
    Ask the trained RL agent for a processing order.
    Falls back to revenue-sort if agent raises any exception.
    """
    try:
        return agent.forward_rl_order(
            substrate, vnr_list,
            build_sub_fn=build_sub_fn,
            build_vnr_fn=build_vnr_fn,
        )
    except Exception as e:
        print(f"[HPSO Batch] RL agent ordering failed ({e}); "
              f"falling back to revenue-sort.")
        return _sort_by_revenue(vnr_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hpso_embed_batch(
    substrate,
    batch,
    rl_agent            = None,
    build_sub_fn        = None,
    build_vnr_fn        = None,
    particles:    int   = 20,
    iterations:   int   = 30,
    w_max:        float = 0.9,
    w_min:        float = 0.5,
    beta:         float = 0.3,
    gamma:        float = 0.3,
    T0:           float = 100,
    cooling_rate: float = 0.95,
    verbose:      bool  = False,
) -> Tuple[list, list]:
    """
    Embed a batch of VNRs using HPSO, with RL-based or heuristic ordering.

    Parameters
    ----------
    substrate   : networkx substrate graph (resources are mutated in-place on success)
    batch       : list of VNR graphs  **or**  list of (vnr, info) tuples
    rl_agent    : optional VNRSchedulerAgent; if None, uses revenue-sort heuristic
    build_sub_fn: optional DGL builder for substrate (passed to rl_agent)
    build_vnr_fn: optional DGL builder for VNRs      (passed to rl_agent)
    particles   : HPSO particle count
    iterations  : HPSO iteration count
    w_max, w_min, beta, gamma, T0, cooling_rate : HPSO hyper-parameters
    verbose     : print per-VNR progress

    Returns
    -------
    accepted : list of (vnr, mapping, link_paths)
    rejected : list of vnr
    """
    vnr_list = _unpack_batch(batch)

    if len(vnr_list) == 0:
        return [], []

    # --- Determine processing order ---
    if rl_agent is not None:
        order = _rl_order(
            vnr_list, substrate, rl_agent,
            build_sub_fn=build_sub_fn,
            build_vnr_fn=build_vnr_fn,
        )
        if verbose:
            print(f"[HPSO Batch] RL ordering: {order}")
    else:
        # Original behaviour: sort by revenue descending
        order = _sort_by_revenue(vnr_list)
        if verbose:
            print(f"[HPSO Batch] Revenue-sort ordering: {order}")

    # --- Embed in chosen order ---
    accepted: list = []
    rejected: list = []

    for step, idx in enumerate(order):
        vnr = vnr_list[idx]
        if verbose:
            print(f"[HPSO Batch] Step {step + 1}/{len(order)}: "
                  f"VNR[{idx}] (nodes={len(vnr.nodes())})")

        result = hpso_embed(
            substrate_graph=substrate,
            vnr_graph=vnr,
            particles=particles,
            iterations=iterations,
            w_max=w_max,
            w_min=w_min,
            beta=beta,
            gamma=gamma,
            T0=T0,
            cooling_rate=cooling_rate,
        )

        if result is not None:
            mapping, link_paths = result
            accepted.append((vnr, mapping, link_paths))
            if verbose:
                print(f"   → Accepted")
        else:
            rejected.append(vnr)
            if verbose:
                print(f"   → Rejected")

    return accepted, rejected


# ---------------------------------------------------------------------------
# Backward-compatible alias
#   Old code:  from src.algorithms.hpso_batch import hpso_embed_batch
#   Kept alias so no existing callsite breaks.
# ---------------------------------------------------------------------------

def hpso_batch(substrate, batch, **kwargs):
    """Alias for hpso_embed_batch (backward compatible)."""
    return hpso_embed_batch(substrate, batch, **kwargs)