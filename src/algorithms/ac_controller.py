# src/algorithms/ac_controller.py
"""
RL-based Admission Controller for VNE.

Adapted from HRL-ACRA's upper-level agent (hrl_ac).
Uses a GAT-based encoder to produce graph-level embeddings of
the substrate and VNR, then an MLP outputs accept/reject logits.

**Pure PyTorch implementation** -- no torch_geometric dependency.
Works with plain NetworkX graphs (cpu/bw attributes).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Feature dimensions (must match HRL-ACRA's training config)
#   p_net: 1(cpu) + 3(degree, max_link_bw, sum_link_bw) = 4
#   v_net: 2(cpu + lifetime) + 3(degree, max_link_bw, sum_link_bw) = 5
#   edge dims = 1 (bw)
# =========================================================================
P_NET_FEATURE_DIM = 4
V_NET_FEATURE_DIM = 5
P_NET_EDGE_DIM = 1
V_NET_EDGE_DIM = 1


# =========================================================================
#  Pure-PyTorch GAT layer  (replaces torch_geometric.nn.GATConv)
# =========================================================================

class PureGATConv(nn.Module):
    """
    Single-head GAT convolution with optional edge features.
    Mirrors the interface of PyG's GATConv(in, out, edge_dim=...).
    """
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False) if edge_dim else None
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.lin_src.weight)
        nn.init.orthogonal_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att)
        if self.lin_edge is not None:
            nn.init.orthogonal_(self.lin_edge.weight)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x:          (N, in_channels)
            edge_index: (2, E)  LongTensor
            edge_attr:  (E, edge_dim) or None
        Returns:
            out: (N, out_channels)
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        N = x.size(0)

        x_src = self.lin_src(x)   # (N, out)
        x_dst = self.lin_dst(x)   # (N, out)

        # --- attention coefficients ---
        alpha_src = x_src[src_idx]  # (E, out)
        alpha_dst = x_dst[dst_idx]  # (E, out)

        if self.lin_edge is not None and edge_attr is not None:
            edge_emb = self.lin_edge(edge_attr)  # (E, out)
            alpha_src = alpha_src + edge_emb

        alpha = torch.cat([alpha_src, alpha_dst], dim=-1)  # (E, 2*out)
        alpha = (alpha * self.att).sum(dim=-1)              # (E,)
        alpha = F.leaky_relu(alpha, 0.2)

        # softmax per destination
        alpha = alpha - alpha.max()  # numerical stability
        alpha_exp = alpha.exp()
        denom = torch.zeros(N, device=x.device)
        denom.scatter_add_(0, dst_idx, alpha_exp)
        alpha_norm = alpha_exp / (denom[dst_idx] + 1e-12)  # (E,)

        # --- message aggregation ---
        msg = x_src[src_idx] * alpha_norm.unsqueeze(-1)  # (E, out)
        out = torch.zeros(N, x_src.size(1), device=x.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)
        return out + self.bias


# =========================================================================
#  Graph Pooling  (replaces torch_scatter / torch_geometric pooling)
# =========================================================================

def scatter_reduce(x, batch, num_graphs, reduce='mean'):
    """Pure-PyTorch scatter reduce: mean, sum, or max."""
    out = torch.zeros(num_graphs, x.size(1), device=x.device)
    if reduce == 'sum' or reduce == 'add':
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    elif reduce == 'mean':
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts = torch.zeros(num_graphs, device=x.device)
        counts.scatter_add_(0, batch, torch.ones(x.size(0), device=x.device))
        out = out / counts.clamp(min=1).unsqueeze(-1)
    return out


class GraphAttentionPooling(nn.Module):
    """Attention pooling -> single graph-level vector."""
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.orthogonal_(self.weight)

    def forward(self, x, batch, num_graphs=None):
        if num_graphs is None:
            num_graphs = batch[-1].item() + 1
        mean = scatter_reduce(x, batch, num_graphs, 'mean')
        transformed = torch.tanh(torch.mm(mean, self.weight))
        coefs = torch.sigmoid((x * transformed[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x
        return scatter_reduce(weighted, batch, num_graphs, 'sum')


class GraphPooling(nn.Module):
    def __init__(self, aggr='mean'):
        super().__init__()
        self.aggr = aggr

    def forward(self, x, batch, num_graphs=None):
        if num_graphs is None:
            num_graphs = batch[-1].item() + 1
        return scatter_reduce(x, batch, num_graphs, self.aggr)


# =========================================================================
#  DeepEdgeFeatureGAT  (mirrors HRL-ACRA net.py exactly)
# =========================================================================

class DeepEdgeFeatureGAT(nn.Module):
    """
    N-layer GAT with edge features, initial residual, identity mapping.
    """
    def __init__(self, input_dim, output_dim, edge_dim,
                 num_layers=5, alpha=0.2, theta=0.2,
                 embedding_dim=128, dropout_prob=1.0,
                 batch_norm=False):
        super().__init__()
        assert num_layers >= 2
        self.alpha = alpha
        self.theta = theta
        self.num_mid = num_layers - 2

        self.conv_s = PureGATConv(input_dim, embedding_dim, edge_dim=edge_dim)
        for i in range(self.num_mid):
            self.add_module(f'conv_{i}',
                PureGATConv(embedding_dim, embedding_dim, edge_dim=edge_dim))
            self.add_module(f'norm_{i}',
                nn.BatchNorm1d(embedding_dim) if batch_norm else nn.Identity())
            self.add_module(f'dout_{i}',
                nn.Dropout(dropout_prob) if dropout_prob < 1.0 else nn.Identity())
            self.register_parameter(f'weight_{i}',
                nn.Parameter(torch.Tensor(embedding_dim, embedding_dim)))
        self.conv_e = PureGATConv(embedding_dim, output_dim, edge_dim=edge_dim)
        self._init_weight_params()

    def _init_weight_params(self):
        for i in range(self.num_mid):
            nn.init.orthogonal_(getattr(self, f'weight_{i}'))

    def forward(self, x, edge_index, edge_attr):
        x0 = self.conv_s(x, edge_index, edge_attr)
        x = x0
        for i in range(self.num_mid):
            conv_x = getattr(self, f'conv_{i}')(x, edge_index, edge_attr)
            norm   = getattr(self, f'norm_{i}')
            dout   = getattr(self, f'dout_{i}')
            w      = getattr(self, f'weight_{i}')
            beta   = math.log(self.theta / (i + 1) + 1)
            conv_x.mul_(1 - self.alpha)
            res_x  = self.alpha * x0
            x = conv_x.add_(res_x)
            x = torch.addmm(x, x, w, beta=1.0 - beta, alpha=beta)
            x = F.leaky_relu(dout(norm(x)))
        x = self.conv_e(x, edge_index, edge_attr)
        return x


class MLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3,
                 embedding_dims=None, batch_norm=False):
        super().__init__()
        if embedding_dims is None:
            embedding_dims = [input_dim * 2] * (num_layers - 1)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers += [nn.Linear(input_dim, embedding_dims[0]),
                           nn.BatchNorm1d(embedding_dims[0]) if batch_norm else nn.Identity()]
            elif i == num_layers - 1:
                layers += [nn.Linear(embedding_dims[-1], output_dim), nn.Identity()]
            else:
                layers += [nn.Linear(embedding_dims[i-1], embedding_dims[i]),
                           nn.BatchNorm1d(embedding_dims[i]) if batch_norm else nn.Identity()]
            if i != num_layers - 1:
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):
        return self.net(x)


# =========================================================================
#  Encoder  (mirrors hrl_ac/net.py)
# =========================================================================

class ACEncoder(nn.Module):
    """Encode substrate + VNR into a fused embedding vector."""
    def __init__(self, p_feat_dim, p_edge_dim, v_feat_dim, v_edge_dim,
                 emb_dim=128, dropout=1.0, batch_norm=False):
        super().__init__()
        self.p_gnn = DeepEdgeFeatureGAT(
            p_feat_dim, emb_dim, p_edge_dim,
            num_layers=5, embedding_dim=emb_dim,
            dropout_prob=dropout, batch_norm=batch_norm)
        self.v_gnn = DeepEdgeFeatureGAT(
            v_feat_dim, emb_dim, v_edge_dim,
            num_layers=3, embedding_dim=emb_dim,
            dropout_prob=dropout, batch_norm=batch_norm)
        self.p_gap  = GraphAttentionPooling(emb_dim)
        self.v_gap  = GraphAttentionPooling(emb_dim)
        self.p_mean = GraphPooling('mean')
        self.v_mean = GraphPooling('mean')
        self.p_sum  = GraphPooling('sum')
        self.v_sum  = GraphPooling('sum')

    def forward(self, p_x, p_ei, p_ea, p_batch,
                      v_x, v_ei, v_ea, v_batch):
        p_emb = self.p_gnn(p_x, p_ei, p_ea)
        v_emb = self.v_gnn(v_x, v_ei, v_ea)
        p_g = (self.p_gap(p_emb, p_batch)
             + self.p_mean(p_emb, p_batch)
             + self.p_sum(p_emb, p_batch))
        v_g = (self.v_gap(v_emb, v_batch)
             + self.v_mean(v_emb, v_batch)
             + self.v_sum(v_emb, v_batch))
        return torch.cat([p_g, v_g], dim=-1)


class ACActorCritic(nn.Module):
    """
    Full ActorCritic matching HRL-ACRA's hrl_ac/net.py.
    Actor  -> 2-class logits (reject, accept)
    Critic -> scalar value
    """
    def __init__(self, p_feat_dim=P_NET_FEATURE_DIM,
                       p_edge_dim=P_NET_EDGE_DIM,
                       v_feat_dim=V_NET_FEATURE_DIM,
                       v_edge_dim=V_NET_EDGE_DIM,
                       emb_dim=128, dropout=0.5, batch_norm=False):
        super().__init__()
        self.actor_enc = ACEncoder(p_feat_dim, p_edge_dim,
                                   v_feat_dim, v_edge_dim,
                                   emb_dim, dropout, batch_norm)
        self.actor_mlp = MLPNet(emb_dim * 2, 2, num_layers=3,
                                embedding_dims=[emb_dim * 2, emb_dim],
                                batch_norm=False)
        self.critic_enc = ACEncoder(p_feat_dim, p_edge_dim,
                                    v_feat_dim, v_edge_dim,
                                    emb_dim, dropout, batch_norm)
        self.critic_mlp = MLPNet(emb_dim * 2, 1, num_layers=3,
                                 embedding_dims=[emb_dim * 2, emb_dim],
                                 batch_norm=False)

    def act(self, obs):
        """Return action logits (batch, 2)."""
        fused = self.actor_enc(
            obs['p_x'], obs['p_ei'], obs['p_ea'], obs['p_batch'],
            obs['v_x'], obs['v_ei'], obs['v_ea'], obs['v_batch'])
        return self.actor_mlp(fused)

    def evaluate(self, obs):
        """Return state value (batch, 1)."""
        fused = self.critic_enc(
            obs['p_x'], obs['p_ei'], obs['p_ea'], obs['p_batch'],
            obs['v_x'], obs['v_ei'], obs['v_ea'], obs['v_batch'])
        return self.critic_mlp(fused)


# =========================================================================
#  Feature Extraction  (NetworkX -> tensors)
# =========================================================================

def _edge_index_and_attr(G, bw_benchmark):
    """Convert NetworkX graph edges to COO edge_index + edge_attr."""
    edges = list(G.edges())
    if not edges:
        ei = np.zeros((2, 0), dtype=np.int64)
        ea = np.zeros((0, 1), dtype=np.float32)
        return ei, ea
    # Undirected -> both directions
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    ei = np.array([src, dst], dtype=np.int64)
    bw_vals = [G.edges[u, v].get('bw', 0) for u, v in edges]
    bw_vals = bw_vals + bw_vals
    ea = (np.array(bw_vals, dtype=np.float32) / max(bw_benchmark, 1e-6)).reshape(-1, 1)
    return ei, ea


def extract_features(substrate, vnr, vnr_lifetime_norm=1.0):
    """
    Build observation from plain NetworkX graphs.

    Returns dict with numpy arrays:
        p_net_x, p_net_edge_index, p_net_edge_attr,
        v_net_x, v_net_edge_index, v_net_edge_attr,
        v_net_attrs
    """
    nodes = list(substrate.nodes())
    cpus  = [substrate.nodes[n].get('cpu', 0) for n in nodes]
    cpu_bm = max(cpus) if cpus else 1.0

    degrees = dict(substrate.degree())
    deg_bm  = max(degrees.values()) if degrees else 1.0

    bw_list = [d.get('bw', 0) for _, _, d in substrate.edges(data=True)]
    bw_bm   = max(bw_list) if bw_list else 1.0

    def _node_link_stats(G, bm):
        stats_max, stats_sum = [], []
        for n in G.nodes():
            nbrs = list(G.edges(n, data=True))
            bws = [d.get('bw', 0) for _, _, d in nbrs]
            stats_max.append(max(bws) / max(bm, 1e-6) if bws else 0.0)
            stats_sum.append(sum(bws) / max(bm * len(G.nodes()), 1e-6) if bws else 0.0)
        return np.array(stats_max, dtype=np.float32), np.array(stats_sum, dtype=np.float32)

    # substrate node features (N, 4): [cpu, degree, max_link_bw, sum_link_bw]
    p_cpu  = np.array([substrate.nodes[n].get('cpu', 0) / max(cpu_bm, 1e-6)
                       for n in nodes], dtype=np.float32)
    p_deg  = np.array([degrees[n] / max(deg_bm, 1e-6) for n in nodes], dtype=np.float32)
    p_lmax, p_lsum = _node_link_stats(substrate, bw_bm)
    p_x = np.stack([p_cpu, p_deg, p_lmax, p_lsum], axis=1)

    p_ei, p_ea = _edge_index_and_attr(substrate, bw_bm)

    # VNR node features (M, 5): [cpu, degree, max_link_bw, sum_link_bw, lifetime]
    vnodes = list(vnr.nodes())
    v_cpu  = np.array([vnr.nodes[n].get('cpu', 0) / max(cpu_bm, 1e-6)
                       for n in vnodes], dtype=np.float32)
    vdeg   = dict(vnr.degree())
    v_dg   = np.array([vdeg[n] / max(deg_bm, 1e-6) for n in vnodes], dtype=np.float32)
    v_lmax, v_lsum = _node_link_stats(vnr, bw_bm)
    lifetime_col = np.full(len(vnodes), vnr_lifetime_norm, dtype=np.float32)

    v_base = np.stack([v_cpu, v_dg, v_lmax, v_lsum], axis=1)
    v_x = np.concatenate([v_base, lifetime_col.reshape(-1, 1)], axis=1)

    v_ei, v_ea = _edge_index_and_attr(vnr, bw_bm)

    return {
        'p_net_x': p_x,
        'p_net_edge_index': p_ei,
        'p_net_edge_attr': p_ea,
        'v_net_x': v_x,
        'v_net_edge_index': v_ei,
        'v_net_edge_attr': v_ea,
    }


def obs_to_tensor(obs, device):
    """Convert numpy observation dict to model-ready tensors."""
    p_x  = torch.tensor(obs['p_net_x'], dtype=torch.float32).to(device)
    p_ei = torch.tensor(obs['p_net_edge_index'], dtype=torch.long).to(device)
    p_ea = torch.tensor(obs['p_net_edge_attr'], dtype=torch.float32).to(device)
    v_x  = torch.tensor(obs['v_net_x'], dtype=torch.float32).to(device)
    v_ei = torch.tensor(obs['v_net_edge_index'], dtype=torch.long).to(device)
    v_ea = torch.tensor(obs['v_net_edge_attr'], dtype=torch.float32).to(device)
    # batch vectors (single graph in batch)
    p_batch = torch.zeros(p_x.size(0), dtype=torch.long, device=device)
    v_batch = torch.zeros(v_x.size(0), dtype=torch.long, device=device)
    return {
        'p_x': p_x, 'p_ei': p_ei, 'p_ea': p_ea, 'p_batch': p_batch,
        'v_x': v_x, 'v_ei': v_ei, 'v_ea': v_ea, 'v_batch': v_batch,
    }


# =========================================================================
#  Public API
# =========================================================================

class AdmissionController:
    """
    Wraps ACActorCritic for easy inference.

    Usage
    -----
    >>> ac = AdmissionController('path/to/model.pkl')
    >>> ac.should_accept(substrate_nx, vnr_nx)   # -> True / False
    >>> ac.score(substrate_nx, vnr_nx)            # -> float in [0, 1]
    """

    def __init__(self, model_path=None, device='cpu', embedding_dim=64):
        self.device = torch.device(device)
        self.policy = ACActorCritic(
            emb_dim=embedding_dim,
            dropout=0.5,
            batch_norm=False,
        ).to(self.device)

        if model_path is not None:
            self._load(model_path)
        self.policy.eval()

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if 'policy' in ckpt:
            self.policy.load_state_dict(ckpt['policy'])
        else:
            self.policy.load_state_dict(ckpt)
        print(f'[AC] Loaded pretrained model from {path}')

    @torch.no_grad()
    def score(self, substrate, vnr):
        """Return P(accept) in [0, 1]."""
        obs = extract_features(substrate, vnr)
        t_obs = obs_to_tensor(obs, self.device)
        logits = self.policy.act(t_obs)
        probs  = F.softmax(logits, dim=-1)
        return probs[0, 1].item()

    @torch.no_grad()
    def should_accept(self, substrate, vnr, threshold=0.5):
        """Binary decision: accept if P(accept) >= threshold."""
        return self.score(substrate, vnr) >= threshold
