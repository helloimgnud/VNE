"""
src/algorithms/hpso_batch_scheduler.py
========================================
Drop-in replacement / extension of hpso_batch.py that uses the GCN-RL
VNRScheduler for VNR ordering.

Design principles
-----------------
- The ORIGINAL ``hpso_batch.py`` is NOT modified.
- This module is a NEW file with a new function name
  ``hpso_embed_batch_scheduled`` that adds the scheduler plugin point.
- Backward-compatible: calling with ``scheduler=None`` falls back to the
  original revenue-sort heuristic (exactly replicating hpso_batch.py).

API
---
# Original behaviour (revenue sort, no scheduler):
    accepted, rejected = hpso_embed_batch_scheduled(substrate, batch)

# With trained scheduler (learned ordering):
    from src.scheduler import VNRScheduler
    from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
    scheduler = VNRScheduler.load("checkpoints/reinforce_phase1_final.pt")
    accepted, rejected = hpso_embed_batch_scheduled(
        substrate, batch, scheduler=scheduler
    )

Integration with existing workflows
------------------------------------
If you have code that calls ``hpso_embed_batch`` from hpso_batch.py, you can
transparently switch by replacing the import line:

    # Before
    from src.algorithms.hpso_batch import hpso_embed_batch

    # After (with scheduler)
    from src.algorithms.hpso_batch_scheduler import hpso_embed_batch_scheduled as hpso_embed_batch
    # Then pass scheduler=scheduler_instance to each call.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.algorithms.fast_hpso import hpso_embed
from src.evaluation.eval import revenue_of_vnr


# ---------------------------------------------------------------------------
# Internal helpers (mirrored from hpso_batch.py for self-containment)
# ---------------------------------------------------------------------------

def _unpack_batch(batch) -> list:
    """Accept both plain VNR lists and [(vnr, info), ...] tuples."""
    if len(batch) > 0 and isinstance(batch[0], tuple):
        return [vnr for vnr, _ in batch]
    return list(batch)


def _revenue_sort_order(vnr_list: list) -> List[int]:
    """Original heuristic: sort by revenue descending, return index list."""
    return sorted(
        range(len(vnr_list)),
        key=lambda i: revenue_of_vnr(vnr_list[i]),
        reverse=True,
    )


def _scheduler_order(
    vnr_list:  list,
    substrate,
    scheduler,
) -> List[int]:
    """
    Ask the trained VNRScheduler for a processing order.

    Falls back to revenue-sort if scoring fails for any reason
    (e.g. PyG not installed, encoder error).

    Returns
    -------
    list[int] : indices into vnr_list, highest-priority first
    """
    try:
        from src.scheduler.features import substrate_to_pyg, vnr_to_pyg

        sub_data  = substrate_to_pyg(substrate)
        vnr_datas = [vnr_to_pyg(v) for v in vnr_list]

        scores = scheduler.predict(sub_data, vnr_datas)   # [B], no grad

        # argsort descending → highest score first
        order = scores.argsort(descending=True).tolist()
        return order

    except Exception as exc:
        print(
            f"[hpso_batch_scheduler] Scheduler ordering failed ({exc}); "
            f"falling back to revenue-sort."
        )
        return _revenue_sort_order(vnr_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hpso_embed_batch_scheduled(
    substrate,
    batch,
    scheduler           = None,
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
    Embed a batch of VNRs using HPSO with optional GCN-RL ordering.

    Parameters
    ----------
    substrate     : networkx substrate graph (mutated in-place on success)
    batch         : list of VNR graphs  **or**  list of (vnr, info) tuples
    scheduler     : VNRScheduler instance (optional).
                    If None, falls back to static revenue-sort ordering.
    particles, iterations, w_max, w_min, beta, gamma, T0, cooling_rate
                  : HPSO hyper-parameters (same defaults as hpso_batch.py)
    verbose       : print per-VNR embedding progress

    Returns
    -------
    accepted : list of (vnr, mapping, link_paths)
    rejected : list of vnr
    """
    vnr_list = _unpack_batch(batch)

    if len(vnr_list) == 0:
        return [], []

    accepted: list = []
    rejected: list = []

    if scheduler is not None:
        # -----------------------------------------------------
        # Autoregressive MDP Inference (Matches PPO Training)
        # -----------------------------------------------------
        # The agent must observe the degraded substrate after EVERY decision
        # to pick the next optimum VNR, just like it does in VNEOrderingEnv.
        from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
        remaining_indices = list(range(len(vnr_list)))
        step = 0

        if verbose:
            print(f"[HPSO Scheduler] Starting autoregressive RL inference for {len(vnr_list)} VNRs.")

        while remaining_indices:
            try:
                sub_data = substrate_to_pyg(substrate)
                rem_vnrs = [vnr_to_pyg(vnr_list[i]) for i in remaining_indices]
                scores = scheduler.predict(sub_data, rem_vnrs)  # [n_remaining]
                local_idx = scores.argmax().item()
                global_idx = remaining_indices.pop(local_idx)
            except Exception as exc:
                print(f"[hpso_batch_scheduler] RL inference failed ({exc}). Falling back to static revenue-sort.")
                remaining_indices.sort(key=lambda i: revenue_of_vnr(vnr_list[i]), reverse=True)
                global_idx = remaining_indices.pop(0)

            vnr = vnr_list[global_idx]
            if verbose:
                print(f"[HPSO Scheduler] RL Step {step + 1}/{len(vnr_list)}: VNR[{global_idx}]")

            result = hpso_embed(
                substrate_graph=substrate,
                vnr_graph=vnr,
                particles=particles, iterations=iterations,
                w_max=w_max, w_min=w_min, beta=beta, gamma=gamma,
                T0=T0, cooling_rate=cooling_rate,
            )

            if result is not None:
                mapping, link_paths = result
                accepted.append((vnr, mapping, link_paths))
                if verbose:
                    print("   → Accepted")
            else:
                rejected.append(vnr)
                if verbose:
                    print("   → Rejected")
            
            step += 1

    else:
        # -----------------------------------------------------
        # Original Static Inference (Revenue descending)
        # -----------------------------------------------------
        order = _revenue_sort_order(vnr_list)
        if verbose:
            print(f"[HPSO Scheduler] Revenue-sort ordering: {order}")

        for step, idx in enumerate(order):
            vnr = vnr_list[idx]

            if verbose:
                print(f"[HPSO Scheduler] Step {step + 1}/{len(order)}: VNR[{idx}]")

            result = hpso_embed(
                substrate_graph=substrate,
                vnr_graph=vnr,
                particles=particles, iterations=iterations,
                w_max=w_max, w_min=w_min, beta=beta, gamma=gamma,
                T0=T0, cooling_rate=cooling_rate,
            )

            if result is not None:
                mapping, link_paths = result
                accepted.append((vnr, mapping, link_paths))
                if verbose:
                    print("   → Accepted")
            else:
                rejected.append(vnr)
                if verbose:
                    print("   → Rejected")

    return accepted, rejected


# ---------------------------------------------------------------------------
# Convenience alias matching original function name
# Allows zero-change drop-in replacement when scheduler is not passed.
# ---------------------------------------------------------------------------

def hpso_embed_batch(substrate, batch, **kwargs):
    """
    Alias for hpso_embed_batch_scheduled.

    Calling ``hpso_embed_batch(substrate, batch)``  ← no scheduler ← exact same
    behaviour as the original ``hpso_batch.hpso_embed_batch``.

    Calling ``hpso_embed_batch(substrate, batch, scheduler=model)`` uses the
    learned ordering.
    """
    return hpso_embed_batch_scheduled(substrate, batch, **kwargs)
