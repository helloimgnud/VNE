import copy
from src.algorithms.fast_hpso import hpso_embed
from src.evaluation.eval import revenue_of_vnr

def hpso_embed_batch(
    substrate,
    batch,
    particles=20,
    iterations=30,
    w_max=0.9,
    w_min=0.5,
    beta=0.3,
    gamma=0.3,
    T0=100,
    cooling_rate=0.95,
    verbose=False,
):
    """
    Embed a batch of VNRs using HPSO.
    Processes the VNRs sequentially in the received ordering.
    (Later, this ordering will be optimized by an external ordering network).

    Returns
    -------
    accepted : list of (vnr, mapping, link_paths)
    rejected : list of vnr
    """
    accepted = []
    rejected = []

    # Parse batch: can be a list of VNRs or list of (vnr, info) tuples
    # Matches the structure of embed_batch in parallel_mt_vne.py
    if len(batch) > 0 and isinstance(batch[0], tuple):
        vnr_list = [vnr for vnr, _ in batch]
    else:
        vnr_list = batch

    # Sort VNRs by revenue in descending order (bigger revenue first)
    vnr_list.sort(key=lambda x: revenue_of_vnr(x), reverse=True)

    for i, vnr in enumerate(vnr_list):
        if verbose:
            print(f"[HPSO Batch] Processing VNR {i+1}/{len(vnr_list)} (nodes: {len(vnr.nodes())})")
            
        # hpso_embed evaluates and implicitly reserves resources on the substrate graph
        # upon successful embedding via build_and_reserve
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
            cooling_rate=cooling_rate
        )

        if result is not None:
            mapping, link_paths = result
            accepted.append((vnr, mapping, link_paths))
            if verbose:
                print(f" -> Accepted")
        else:
            rejected.append(vnr)
            if verbose:
                print(f" -> Rejected")

    return accepted, rejected