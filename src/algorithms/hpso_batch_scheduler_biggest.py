"""
src/algorithms/hpso_batch_scheduler_biggest.py
==============================================
Greedy version of the batch scheduler.
Sorts VNRs in descending order based on their revenue
(biggest revenue first).
"""

from typing import List, Tuple
from src.algorithms.fast_hpso import hpso_embed
from src.evaluation.eval import revenue_of_vnr

def _unpack_batch(batch) -> list:
    """Accept both plain VNR lists and [(vnr, info), ...] tuples."""
    if len(batch) > 0 and isinstance(batch[0], tuple):
        return [vnr for vnr, _ in batch]
    return list(batch)

def _revenue_sort_biggest_first(vnr_list: list) -> List[int]:
    """Greedy heuristic: sort by revenue descending (biggest to smallest)."""
    return sorted(
        range(len(vnr_list)),
        key=lambda i: revenue_of_vnr(vnr_list[i]),
        reverse=True,
    )

def hpso_embed_batch_biggest(
    substrate,
    batch,
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
    Embed a batch of VNRs using a pure greedy strategy: scheduling strictly
    from biggest revenue to smallest revenue.
    """
    vnr_list = _unpack_batch(batch)

    if len(vnr_list) == 0:
        return [], []

    accepted: list = []
    rejected: list = []

    order = _revenue_sort_biggest_first(vnr_list)
    if verbose:
        print(f"[HPSO Biggest] Biggest-revenue-first ordering: {order}")

    for step, idx in enumerate(order):
        vnr = vnr_list[idx]

        if verbose:
            print(f"[HPSO Biggest] Step {step + 1}/{len(order)}: VNR[{idx}]")

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

# Alias for backward compatibility if replacing standard embed_batch signatures
def hpso_embed_batch(substrate, batch, **kwargs):
    return hpso_embed_batch_biggest(substrate, batch, **kwargs)
