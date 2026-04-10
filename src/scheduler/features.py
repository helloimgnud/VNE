"""
src/scheduler/features.py
=========================
NetworkX graph → PyTorch Geometric Data conversion.

Functions
---------
substrate_to_pyg(G)  : convert a substrate NetworkX graph to a PyG Data object
vnr_to_pyg(G)        : convert a VNR NetworkX graph to a PyG Data object

Feature specs (see network_encoder_rl.md §4):
  Substrate node  : [cpu_avail, cpu_ratio, degree, avg_bw_neighbors, clustering_coeff]  → shape [N_s, 5]
  Substrate edge  : [bw_avail, bw_ratio]                                                 → shape [E_s, 2]
  VNR node        : [cpu_demand, degree, sum_bw_incident, cpu_demand/total_cpu]          → shape [N_v, 4]
  VNR edge        : [bw_demand]                                                          → shape [E_v, 1]

Both functions work safely when the graph has no edges (returns empty tensors).

The converters are intentionally graph-feature-only — they do NOT call any
random generator or modify the graph state.
"""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Substrate graph → PyG
# ---------------------------------------------------------------------------

def substrate_to_pyg(G: nx.Graph) -> Data:
    """
    Convert a substrate NetworkX graph to a PyG Data object.

    Node attributes expected on G:
        ``cpu``       — current available CPU   (float/int)
        ``cpu_total`` — original total CPU      (float/int, optional; defaults to cpu)

    Edge attributes expected on G:
        ``bw``        — current available BW    (float/int)
        ``bw_total``  — original total BW       (float/int, optional; defaults to bw)

    Returns
    -------
    torch_geometric.data.Data
        x          : [N_s, 5]  node features
        edge_index : [2, 2*E_s] (undirected → both directions)
        edge_attr  : [2*E_s, 2] edge features
    """
    nodes = sorted(G.nodes())
    idx   = {n: i for i, n in enumerate(nodes)}

    # --- Node features ---
    # Precompute clustering coefficients (expensive; only for undirected graphs)
    try:
        cluster = nx.clustering(G) if not G.is_directed() else {n: 0.0 for n in nodes}
    except Exception:
        cluster = {n: 0.0 for n in nodes}

    x_rows = []
    for n in nodes:
        nd       = G.nodes[n]
        cpu_av   = float(nd.get("cpu", 0.0))
        cpu_tot  = float(nd.get("cpu_total", cpu_av)) + 1e-6
        cpu_ratio= cpu_av / cpu_tot
        deg      = float(G.degree(n))

        # Average BW of incident edges
        nbrs     = list(G.neighbors(n))
        avg_bw   = 0.0
        if nbrs:
            total_bw = sum(float(G.edges[n, nb].get("bw", 0.0)) for nb in nbrs)
            avg_bw   = total_bw / len(nbrs)

        clust    = float(cluster.get(n, 0.0))
        x_rows.append([cpu_av, cpu_ratio, deg, avg_bw, clust])

    x = torch.tensor(x_rows, dtype=torch.float)  # [N_s, 5]

    # --- Edge index + features ---
    edges = list(G.edges(data=True))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.float)
    else:
        forward  = [[idx[u], idx[v]] for u, v, _ in edges]
        backward = [[idx[v], idx[u]] for u, v, _ in edges]
        edge_index = torch.tensor(forward + backward, dtype=torch.long).t().contiguous()

        ea_rows = []
        for _, _, d in edges:
            bw_av  = float(d.get("bw", 0.0))
            bw_tot = float(d.get("bw_total", bw_av)) + 1e-6
            ea_rows.append([bw_av, bw_av / bw_tot])
        # Duplicate for both directions
        ea_all = ea_rows + ea_rows
        edge_attr = torch.tensor(ea_all, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# VNR graph → PyG
# ---------------------------------------------------------------------------

def vnr_to_pyg(G: nx.Graph) -> Data:
    """
    Convert a VNR NetworkX graph to a PyG Data object.

    Node attributes expected on G:
        ``cpu`` — CPU demand  (float/int)

    Edge attributes expected on G:
        ``bw``  — bandwidth demand  (float/int)

    Returns
    -------
    torch_geometric.data.Data
        x          : [N_v, 4]  node features
        edge_index : [2, 2*E_v]
        edge_attr  : [2*E_v, 1] edge features
    """
    nodes     = sorted(G.nodes())
    idx       = {n: i for i, n in enumerate(nodes)}
    total_cpu = sum(float(G.nodes[n].get("cpu", 0.0)) for n in nodes) + 1e-6

    x_rows = []
    for n in nodes:
        cpu_d      = float(G.nodes[n].get("cpu", 0.0))
        deg        = float(G.degree(n))
        nbrs       = list(G.neighbors(n))
        sum_bw     = sum(float(G.edges[n, nb].get("bw", 0.0)) for nb in nbrs)
        cpu_rel    = cpu_d / total_cpu
        x_rows.append([cpu_d, deg, sum_bw, cpu_rel])

    x = torch.tensor(x_rows, dtype=torch.float)  # [N_v, 4]

    edges = list(G.edges(data=True))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float)
    else:
        forward  = [[idx[u], idx[v]] for u, v, _ in edges]
        backward = [[idx[v], idx[u]] for u, v, _ in edges]
        edge_index = torch.tensor(forward + backward, dtype=torch.long).t().contiguous()

        ea_rows = [[float(d.get("bw", 0.0))] for _, _, d in edges]
        edge_attr = torch.tensor(ea_rows + ea_rows, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
