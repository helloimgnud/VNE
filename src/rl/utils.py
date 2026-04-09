"""
src/rl/utils.py
===============
Shared configuration defaults and DGL graph builders.

The build_vnr_dgl / build_substrate_dgl functions convert networkx graphs
(produced by src/generators/) into DGLGraphs ready for the GATEncoder.
These are the same feature extraction functions previously in vnr_scheduler.py,
moved here so they are importable by the training script, hpso_batch.py, and
the RL env without creating circular imports.
"""

import torch
import dgl
import networkx as nx
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Default hyper-parameter configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict = dict(
    # --- Substrate GAT encoder ---
    substrate_node_feat_dim=4,    # [cpu_res_ratio, mem_res_ratio, avg_bw_ratio, utilisation]
    substrate_hidden=128,
    substrate_gat_heads=[4, 4, 1],

    # --- VNR GAT encoder (shared weights across VNRs in queue) ---
    vnr_node_feat_dim=3,          # [cpu_demand_norm, mem_demand_norm, vnf_type_norm]
    vnr_hidden=64,
    vnr_gat_heads=[4, 4, 1],
    use_summary_feats=False,      # Set True to append [log(|Nv|), max_cpu, max_bw]

    # --- Context MLP & Pointer Decoder ---
    context_dim=256,
    gru_hidden=256,

    # --- Critic head ---
    critic_hidden=128,

    # --- Scheduling ---
    K_max=20,                     # max VNRs per time window (padding target)

    # --- PPO hyper-parameters (idea.md §6 / §10.1) ---
    gamma=0.98,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    lr=3e-4,
    k_epochs=4,
    batch_size=64,

    # --- HPSO defaults passed through to hpso_embed ---
    hpso_particles=20,
    hpso_iterations=30,
    hpso_w_max=0.9,
    hpso_w_min=0.5,
    hpso_beta=0.3,
    hpso_gamma=0.3,
    hpso_T0=100,
    hpso_cooling_rate=0.95,
)


# ---------------------------------------------------------------------------
# DGL graph builders
# ---------------------------------------------------------------------------

def build_vnr_dgl(
    vnr_nx: nx.Graph,
    max_cpu: float = 100.0,
    max_mem: float = 100.0,
    num_vnf_types: int = 10,
) -> dgl.DGLGraph:
    """
    Convert a networkx VNR graph → DGLGraph with ndata['nfeat'].

    Feature columns (3 dims, matching DEFAULT_CFG['vnr_node_feat_dim']):
      [0] cpu_demand / max_cpu             (normalised CPU requirement)
      [1] mem_demand / max_mem             (normalised MEM requirement)
      [2] vnf_type   / num_vnf_types       (normalised VNF type id)

    The function is tolerant of missing node attributes (defaults to 0).
    """
    nodes = list(vnr_nx.nodes())
    edges = list(vnr_nx.edges())

    if len(edges) == 0:
        # GAT requires at least one edge; add self-loops to avoid isolated node issues
        u_t = torch.zeros(len(nodes), dtype=torch.long)
        v_t = torch.zeros(len(nodes), dtype=torch.long)
    else:
        u_t = torch.tensor([e[0] for e in edges], dtype=torch.long)
        v_t = torch.tensor([e[1] for e in edges], dtype=torch.long)

    g = dgl.graph((u_t, v_t), num_nodes=len(nodes))
    g = dgl.add_self_loop(g)   # ensure no zero-in-degree issues

    feats = []
    for n in nodes:
        nd   = vnr_nx.nodes[n]
        cpu  = float(nd.get('cpu', nd.get('cpu_demand', 0.0))) / max_cpu
        mem  = float(nd.get('mem', nd.get('mem_demand', nd.get('memory', 0.0)))) / max_mem
        vnft = float(nd.get('vnf_type', nd.get('type', 0))) / num_vnf_types
        feats.append([cpu, mem, vnft])

    g.ndata['nfeat'] = torch.tensor(feats, dtype=torch.float32)
    return g


def build_substrate_dgl(
    sub_nx: nx.Graph,
    total_cpu: float = 50.0,
    total_mem: float = 50.0,
    total_bw: float  = 100.0,
) -> dgl.DGLGraph:
    """
    Convert a networkx substrate graph → DGLGraph with ndata['nfeat'].

    Feature columns (4 dims, matching DEFAULT_CFG['substrate_node_feat_dim']):
      [0] cpu_res / total_cpu       (residual CPU fraction)
      [1] mem_res / total_mem       (residual MEM fraction)
      [2] avg_bw_res / total_bw     (mean residual BW over incident edges)
      [3] 1 - cpu_res               (utilisation)

    Attribute name heuristics:
      CPU  : 'cpu_res' → 'cpu' (fallback)
      MEM  : 'mem_res' → 'mem' → 'memory' (fallback)
      BW   : 'bw_res'  → 'bw'  (fallback) per edge, neighbour average
    """
    nodes = list(sub_nx.nodes())
    edges = list(sub_nx.edges())

    if len(edges) == 0:
        u_t = torch.zeros(len(nodes), dtype=torch.long)
        v_t = torch.zeros(len(nodes), dtype=torch.long)
    else:
        u_t = torch.tensor([e[0] for e in edges], dtype=torch.long)
        v_t = torch.tensor([e[1] for e in edges], dtype=torch.long)

    g = dgl.graph((u_t, v_t), num_nodes=len(nodes))
    g = dgl.add_self_loop(g)

    feats = []
    for n in nodes:
        nd = sub_nx.nodes[n]
        cpu_res = float(nd.get('cpu_res', nd.get('cpu', total_cpu))) / total_cpu
        mem_res = float(nd.get('mem_res', nd.get('mem', nd.get('memory', total_mem)))) / total_mem
        util    = 1.0 - cpu_res

        nbrs = list(sub_nx.neighbors(n))
        if nbrs:
            bw_vals = [
                float(sub_nx[n][nb].get('bw_res', sub_nx[n][nb].get('bw', total_bw))) / total_bw
                for nb in nbrs
            ]
            bw_avg = sum(bw_vals) / len(bw_vals)
        else:
            bw_avg = 1.0

        feats.append([cpu_res, mem_res, bw_avg, util])

    g.ndata['nfeat'] = torch.tensor(feats, dtype=torch.float32)
    return g
