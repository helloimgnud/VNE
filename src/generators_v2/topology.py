# src/generators_v2/topology.py
"""
Topology helpers for generators_v2.

Exposes:
  - TopologyType   : string constants for all supported topology types
  - build_topology : low-level factory that returns a plain nx.Graph
"""

import networkx as nx


class TopologyType:
    """
    Namespace of supported topology type strings.

    Used as the `topology_type` argument in substrate / VNR generators.
    """
    # ── Erdős–Rényi random graph (default for VNRs) ──────────────────────────
    RANDOM = "random"

    # ── Multi-domain random (intra+inter probabilities) ──────────────────────
    MULTI_DOMAIN = "multi_domain"

    # ── Waxman geographic model (default for substrate in virne) ─────────────
    WAXMAN = "waxman"

    # ── Simple path (chain) ──────────────────────────────────────────────────
    PATH = "path"

    # ── Star ─────────────────────────────────────────────────────────────────
    STAR = "star"

    # ── 2-D grid ─────────────────────────────────────────────────────────────
    GRID_2D = "grid_2d"

    # ── Scale-free (Barabási–Albert) ─────────────────────────────────────────
    BARABASI_ALBERT = "barabasi_albert"

    ALL = [RANDOM, MULTI_DOMAIN, WAXMAN, PATH, STAR, GRID_2D, BARABASI_ALBERT]


def build_topology(
    topology_type: str,
    num_nodes: int,
    *,
    # Erdős–Rényi / random
    random_prob: float = 0.5,
    # Waxman
    wm_alpha: float = 0.5,
    wm_beta: float = 0.2,
    # Grid 2-D
    grid_m: int = None,
    grid_n: int = None,
    # Barabási–Albert
    ba_m: int = 2,
    seed: int = None,
) -> nx.Graph:
    """
    Build and return a connected nx.Graph with the requested topology.

    Parameters
    ----------
    topology_type : str
        One of TopologyType.* constants.
    num_nodes : int
        Number of nodes.
    random_prob : float
        Edge probability for RANDOM topology.
    wm_alpha, wm_beta : float
        Waxman model parameters.
    grid_m, grid_n : int
        Grid dimensions for GRID_2D (you must also set num_nodes == grid_m*grid_n).
    ba_m : int
        Number of edges to attach from a new node (Barabási–Albert).
    seed : int or None
        Random seed.

    Returns
    -------
    nx.Graph – always connected.
    """
    if topology_type == TopologyType.PATH:
        return nx.path_graph(num_nodes)

    if topology_type == TopologyType.STAR:
        return nx.star_graph(num_nodes - 1)

    if topology_type == TopologyType.GRID_2D:
        if grid_m is None or grid_n is None:
            raise ValueError("GRID_2D requires grid_m and grid_n.")
        return nx.grid_2d_graph(grid_m, grid_n)

    if topology_type == TopologyType.WAXMAN:
        while True:
            G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta, seed=seed)
            if nx.is_connected(G):
                return G

    if topology_type == TopologyType.BARABASI_ALBERT:
        return nx.barabasi_albert_graph(num_nodes, ba_m, seed=seed)

    if topology_type in (TopologyType.RANDOM, TopologyType.MULTI_DOMAIN):
        while True:
            G = nx.erdos_renyi_graph(num_nodes, random_prob, seed=seed, directed=False)
            if nx.is_connected(G):
                return G

    raise NotImplementedError(f"Topology '{topology_type}' is not implemented.")
