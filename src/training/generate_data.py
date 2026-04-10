"""
src/training/generate_data.py
=============================
Data-generation factory functions for the VNR ordering scheduler training.

These are thin wrappers around the existing generators in
``src/generators/substrate_generator.py`` and
``src/generators/vnr_generator.py``.

They return **callables** (closures) suitable for use as ``substrate_fn``
and ``batch_fn`` parameters to ``VNEOrderingEnv.__init__``.

Functions
---------
make_substrate_fn(...)  → callable() → networkx.Graph
make_batch_fn(...)      → callable() → list[networkx.Graph]
make_env_fns(...)       → (substrate_fn, batch_fn)
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

import networkx as nx

from src.generators.substrate_generator import generate_substrate
from src.generators.vnr_generator import generate_single_vnr


# ---------------------------------------------------------------------------
# Substrate factory
# ---------------------------------------------------------------------------

def make_substrate_fn(
    num_nodes:    int   = 50,
    num_domains:  int   = 4,
    p_intra:      float = 0.5,
    p_inter:      float = 0.05,
    cpu_range:    Tuple = (50, 100),
    bw_range:     Tuple = (50, 200),
    fixed_seed:   Optional[int] = None,
) -> Callable[[], nx.Graph]:
    """
    Return a zero-argument callable that generates a substrate graph.

    Parameters
    ----------
    num_nodes    : total substrate nodes (default 50)
    num_domains  : number of domains     (default 4)
    p_intra      : intra-domain edge probability
    p_inter      : inter-domain edge probability
    cpu_range    : (min, max) CPU capacity
    bw_range     : (min, max) BW capacity
    fixed_seed   : if not None, every call returns the SAME substrate
                   (useful for deterministic evaluation); if None, the seed
                   increments on each call to get diverse topologies.

    Returns
    -------
    Callable[[], nx.Graph]
    """
    call_counter = [0]

    def _fn() -> nx.Graph:
        if fixed_seed is not None:
            seed = fixed_seed
        else:
            seed = random.randint(0, 2 ** 31 - 1)
        call_counter[0] += 1
        return generate_substrate(
            num_nodes_total = num_nodes,
            num_domains     = num_domains,
            p_intra         = p_intra,
            p_inter         = p_inter,
            cpu_range       = cpu_range,
            bw_range        = bw_range,
            seed            = seed,
        )

    return _fn


# ---------------------------------------------------------------------------
# VNR batch factory
# ---------------------------------------------------------------------------

def make_batch_fn(
    batch_size:   int   = 10,
    num_nodes:    int   = 4,          # mean VNR size; actual size sampled ± 2
    variable_size: bool = True,
    edge_prob:    float = 0.5,
    cpu_range:    Tuple = (5, 30),
    bw_range:     Tuple = (5, 50),
    fixed_seed:   Optional[int] = None,
) -> Callable[[], list]:
    """
    Return a zero-argument callable that generates a batch of VNR graphs.

    Parameters
    ----------
    batch_size    : number of VNRs per batch
    num_nodes     : base number of virtual nodes per VNR
    variable_size : if True, VNR size is sampled uniformly in
                    [max(2, num_nodes-2), num_nodes+2]
    edge_prob     : Erdős–Rényi edge probability
    cpu_range     : (min, max) CPU demand per virtual node
    bw_range      : (min, max) BW demand per virtual link
    fixed_seed    : if not None, returns the same batch every call

    Returns
    -------
    Callable[[], list[nx.Graph]]
    """
    def _fn() -> list:
        if fixed_seed is not None:
            random.seed(fixed_seed)

        batch = []
        for _ in range(batch_size):
            if variable_size:
                n = random.randint(max(2, num_nodes - 2), num_nodes + 2)
            else:
                n = num_nodes
            vnr = generate_single_vnr(
                num_nodes  = n,
                edge_prob  = edge_prob,
                cpu_range  = cpu_range,
                bw_range   = bw_range,
            )
            batch.append(vnr)
        return batch

    return _fn


# ---------------------------------------------------------------------------
# Convenient combined factory
# ---------------------------------------------------------------------------

def make_env_fns(
    substrate_nodes: int   = 50,
    batch_size:      int   = 10,
    vnr_nodes:       int   = 4,
    sub_cpu_range:   Tuple = (50, 100),
    sub_bw_range:    Tuple = (50, 200),
    vnr_cpu_range:   Tuple = (5, 30),
    vnr_bw_range:    Tuple = (5, 50),
    fixed_substrate: bool  = False,
    substrate_seed:  Optional[int] = 42,
) -> Tuple[Callable, Callable]:
    """
    Convenience function to build both substrate_fn and batch_fn together.

    Parameters
    ----------
    substrate_nodes : size of the substrate network
    batch_size      : number of VNRs per episode
    vnr_nodes       : average VNR size
    sub_cpu_range   : substrate CPU capacity range
    sub_bw_range    : substrate BW capacity range
    vnr_cpu_range   : VNR CPU demand range
    vnr_bw_range    : VNR BW demand range
    fixed_substrate : if True, all episodes use the same substrate topology
    substrate_seed  : seed for fixed substrate (ignored if fixed_substrate=False)

    Returns
    -------
    substrate_fn, batch_fn
    """
    substrate_fn = make_substrate_fn(
        num_nodes  = substrate_nodes,
        cpu_range  = sub_cpu_range,
        bw_range   = sub_bw_range,
        fixed_seed = substrate_seed if fixed_substrate else None,
    )
    batch_fn = make_batch_fn(
        batch_size = batch_size,
        num_nodes  = vnr_nodes,
        cpu_range  = vnr_cpu_range,
        bw_range   = vnr_bw_range,
    )
    return substrate_fn, batch_fn
