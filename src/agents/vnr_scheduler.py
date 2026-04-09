"""
vnr_scheduler.py
================
VNR Scheduling Agent based on Graph Pointer Network + PPO.

Replaces the simple revenue-sort heuristic in hpso_batch.py with a
learned policy that decides *which* VNR to feed to the HPSO solver next,
given the current substrate state and the full pending queue.

Architecture (mirrors GPRL paper but at request-level):
  SubstrateEncoder  : 3-layer GAT  → mean-pool → h_p  (R^128)
  VNREncoder        : 3-layer GAT  → max-pool  → h_vi (R^64)  [shared weights]
  ContextMLP        : [h_p || mean(h_vi)] → context (R^256)
  PointerDecoder    : GRU + attention → logits over active queue
  CriticHead        : [h_p || mean(h_vi)] → scalar value

Usage (standalone PPO training loop – no stable_baselines3 dependency):
    See README.md and train_scheduler.py for full training script.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch.conv as dglnn
import networkx as nx
from typing import List, Optional, Tuple, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyper-parameters (can be overridden via config dict)
# ---------------------------------------------------------------------------
DEFAULT_CFG = dict(
    # Substrate encoder
    substrate_node_feat_dim=4,   # [cpu_res_ratio, mem_res_ratio, bw_avg_ratio, util]
    substrate_hidden=128,
    substrate_gat_heads=[4, 4, 1],

    # VNR encoder (shared across all VNRs in queue)
    vnr_node_feat_dim=3,         # [cpu_demand_norm, mem_demand_norm, vnf_type_id_norm]
    vnr_hidden=64,
    vnr_gat_heads=[4, 4, 1],

    # Decoder
    context_dim=256,
    gru_hidden=256,

    # Critic
    critic_hidden=128,

    # Training
    K_max=20,                    # max VNRs per time window (padding target)
    gamma=0.98,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    lr=3e-4,
    k_epochs=4,
    batch_size=64,
)


# ---------------------------------------------------------------------------
# GAT Encoder (3-layer, for both substrate and VNR)
# ---------------------------------------------------------------------------
class GATEncoder(nn.Module):
    """
    3-layer Graph Attention Network encoder.
    Returns per-node embeddings of shape (total_nodes, out_dim).
    """

    def __init__(self, in_feat: int, hidden: int, heads: List[int]):
        """
        heads: list of 3 ints, e.g. [4, 4, 1].
               First two layers concatenate heads → hidden*heads[i] output.
               Last layer averages heads → hidden output.
        """
        super().__init__()
        assert len(heads) == 3
        h0, h1, h2 = heads

        self.gat1 = dglnn.GATConv(in_feat,       hidden,        h0,
                                   allow_zero_in_degree=True, activation=F.elu)
        self.gat2 = dglnn.GATConv(hidden * h0,   hidden,        h1,
                                   allow_zero_in_degree=True, activation=F.elu)
        self.gat3 = dglnn.GATConv(hidden * h1,   hidden,        h2,
                                   allow_zero_in_degree=True, activation=None)
        self.out_dim = hidden  # final dim (h2 heads averaged)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        g    : DGLGraph (may be batched)
        feat : (total_nodes, in_feat)
        returns: (total_nodes, hidden)
        """
        h = self.gat1(g, feat)           # (N, h0, hidden)
        h = h.flatten(1)                  # (N, h0*hidden)
        h = self.gat2(g, h)              # (N, h1, hidden)
        h = h.flatten(1)                  # (N, h1*hidden)
        h = self.gat3(g, h)              # (N, h2=1, hidden)
        h = h.mean(1)                     # (N, hidden)
        return h


# ---------------------------------------------------------------------------
# Substrate Encoder  →  single global vector h_p
# ---------------------------------------------------------------------------
class SubstrateEncoder(nn.Module):
    """
    Encodes the substrate graph into a global embedding h_p ∈ R^hidden.
    Mean-pool over all physical nodes.
    """

    def __init__(self, node_feat_dim: int, hidden: int, heads: List[int]):
        super().__init__()
        self.gat = GATEncoder(node_feat_dim, hidden, heads)
        self.out_dim = hidden

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        g  : single substrate DGLGraph (batch_size=1 during scheduling)
        returns: (1, hidden)  — global substrate embedding
        """
        node_feat = g.ndata['nfeat']          # (N_p, node_feat_dim)
        h = self.gat(g, node_feat)             # (N_p, hidden)
        h_p = h.mean(dim=0, keepdim=True)     # (1, hidden) — mean pool
        return h_p


# ---------------------------------------------------------------------------
# VNR Encoder  →  per-VNR embedding h_vi
# ---------------------------------------------------------------------------
class VNREncoder(nn.Module):
    """
    Encodes a single VNR DAG into a fixed-size embedding h_vi ∈ R^hidden.
    Max-pool over virtual nodes to capture the "hardest" node constraint.

    Weights are shared across all VNRs in the queue.
    """

    def __init__(self, node_feat_dim: int, hidden: int, heads: List[int]):
        super().__init__()
        self.gat = GATEncoder(node_feat_dim, hidden, heads)
        self.out_dim = hidden

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        g  : single VNR DGLGraph
        returns: (1, hidden)  — VNR embedding (max-pool)
        """
        node_feat = g.ndata['nfeat']           # (N_v, node_feat_dim)
        h = self.gat(g, node_feat)              # (N_v, hidden)
        h_vi, _ = h.max(dim=0, keepdim=True)  # (1, hidden) — max pool
        return h_vi


# ---------------------------------------------------------------------------
# Pointer Decoder (GRU + attention)
# ---------------------------------------------------------------------------
class PointerDecoder(nn.Module):
    """
    Given:
      - static_keys : (K_max, key_dim)  — per-VNR embedding projected to key space
      - context     : (1, gru_hidden)   — initial GRU hidden state
      - mask        : (K_max,) bool     — True = available

    For each scheduling step t:
      1. GRU(input=last_selected_key, hidden=h_{t-1}) → d_t
      2. Attention score u^t_i = v^T tanh(W1*h_vi + W2*d_t)
      3. Mask illegal slots, softmax → p_i
    """

    def __init__(self, key_dim: int, gru_hidden: int):
        super().__init__()
        self.key_dim = key_dim
        self.gru_hidden = gru_hidden

        self.gru = nn.GRUCell(key_dim, gru_hidden)

        # Pointer attention parameters (GPRL eq. 18)
        self.W1 = nn.Linear(key_dim,    gru_hidden, bias=False)
        self.W2 = nn.Linear(gru_hidden, gru_hidden, bias=False)
        self.v  = nn.Linear(gru_hidden, 1,          bias=False)

    def forward(
        self,
        keys: torch.Tensor,        # (K_max, key_dim)
        h_prev: torch.Tensor,      # (1, gru_hidden)
        last_key: torch.Tensor,    # (1, key_dim)  — embedding of last chosen VNR
        mask: torch.Tensor,        # (K_max,) bool — True = selectable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits  : (K_max,)  — raw (un-softmaxed, pre-masked) scores
            h_next  : (1, gru_hidden)
        """
        # GRU step
        h_next = self.gru(last_key, h_prev)           # (1, gru_hidden)

        # Attention: u^t_i = v^T * tanh(W1*h_vi + W2*d_t)
        keys_proj = self.W1(keys)                      # (K_max, gru_hidden)
        dec_proj  = self.W2(h_next)                    # (1,     gru_hidden)
        energy    = self.v(torch.tanh(keys_proj + dec_proj))  # (K_max, 1)
        logits    = energy.squeeze(-1)                 # (K_max,)

        # Apply mask: set unavailable entries to -inf
        logits = logits.masked_fill(~mask, float('-inf'))

        return logits, h_next


# ---------------------------------------------------------------------------
# Context MLP  (substrate + queue summary → initial GRU hidden)
# ---------------------------------------------------------------------------
class ContextMLP(nn.Module):
    def __init__(self, substrate_dim: int, vnr_dim: int, context_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(substrate_dim + vnr_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

    def forward(self, h_p: torch.Tensor, h_queue_mean: torch.Tensor) -> torch.Tensor:
        """
        h_p          : (1, substrate_dim)
        h_queue_mean : (1, vnr_dim)
        returns      : (1, context_dim)
        """
        x = torch.cat([h_p, h_queue_mean], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Key Projection (VNR embeddings → pointer key space)
# ---------------------------------------------------------------------------
class KeyProjection(nn.Module):
    """Projects VNR embedding to pointer key space (same dim as GRU hidden)."""
    def __init__(self, vnr_dim: int, key_dim: int):
        super().__init__()
        self.proj = nn.Linear(vnr_dim, key_dim)

    def forward(self, h_vi: torch.Tensor) -> torch.Tensor:
        return self.proj(h_vi)


# ---------------------------------------------------------------------------
# Critic Head
# ---------------------------------------------------------------------------
class CriticHead(nn.Module):
    def __init__(self, substrate_dim: int, vnr_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(substrate_dim + vnr_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_p: torch.Tensor, h_queue_mean: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar value estimate V(s_t) — shape (1,)
        """
        x = torch.cat([h_p, h_queue_mean], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Full VNR Scheduling Agent
# ---------------------------------------------------------------------------
class VNRSchedulerAgent(nn.Module):
    """
    Complete actor-critic agent for VNR scheduling.

    At each step t the agent:
      1. Encodes current substrate state → h_p
      2. Encodes each pending VNR → {h_vi}
      3. Builds context from h_p + mean({h_vi})
      4. Pointer-decoder selects next VNR index from active queue
      5. Critic estimates V(s_t)

    An 'episode' processes one time window: K VNRs are scheduled one by one
    until the queue is empty.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        c = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg = c

        self.K_max      = c['K_max']
        self.vnr_dim    = c['vnr_hidden']
        self.sub_dim    = c['substrate_hidden']
        self.gru_hidden = c['gru_hidden']

        # --- Shared encoders ---
        self.substrate_enc = SubstrateEncoder(
            c['substrate_node_feat_dim'], c['substrate_hidden'], c['substrate_gat_heads']
        )
        self.vnr_enc = VNREncoder(
            c['vnr_node_feat_dim'], c['vnr_hidden'], c['vnr_gat_heads']
        )

        # --- Context and key projection ---
        self.context_mlp = ContextMLP(c['substrate_hidden'], c['vnr_hidden'], c['context_dim'])
        self.key_proj    = KeyProjection(c['vnr_hidden'], c['gru_hidden'])

        # --- Pointer decoder ---
        self.decoder = PointerDecoder(c['gru_hidden'], c['gru_hidden'])

        # Learnable "start token" for first GRU input (no previous selection)
        self.start_token = nn.Parameter(torch.zeros(1, c['gru_hidden']))

        # --- Critic head (shares substrate_enc + vnr_enc) ---
        self.critic = CriticHead(c['substrate_hidden'], c['vnr_hidden'], c['critic_hidden'])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    def encode_substrate(self, g_sub: dgl.DGLGraph) -> torch.Tensor:
        """Returns h_p ∈ (1, substrate_hidden)."""
        return self.substrate_enc(g_sub)

    def encode_vnr_queue(
        self,
        vnr_graphs: List[Optional[dgl.DGLGraph]],
    ) -> torch.Tensor:
        """
        Encode K_max VNR slots.
        vnr_graphs: list of length K_max; None entries = padding (dummy).
        Returns keys : (K_max, gru_hidden)
        """
        embeddings = []
        for g in vnr_graphs:
            if g is not None:
                h = self.vnr_enc(g.to(device))          # (1, vnr_hidden)
            else:
                h = torch.zeros(1, self.vnr_dim, device=device)
            embeddings.append(h)

        stack = torch.cat(embeddings, dim=0)             # (K_max, vnr_hidden)
        keys  = self.key_proj(stack)                     # (K_max, gru_hidden)
        return keys, stack                               # keys for decoder, stack for critic/context

    # ------------------------------------------------------------------
    # Single scheduling step (used during rollout)
    # ------------------------------------------------------------------

    def step(
        self,
        h_p:      torch.Tensor,   # (1, sub_dim)
        keys:     torch.Tensor,   # (K_max, gru_hidden)
        h_stack:  torch.Tensor,   # (K_max, vnr_dim)  — raw VNR embeddings
        mask:     torch.Tensor,   # (K_max,) bool
        h_gru:    torch.Tensor,   # (1, gru_hidden)  GRU hidden state
        last_key: torch.Tensor,   # (1, gru_hidden)  last chosen key
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action    : int  (chosen VNR index in 0..K_max-1)
            log_prob  : (1,) tensor
            value     : (1,) tensor
            h_gru_new : (1, gru_hidden)
        """
        # Pointer decoder
        logits, h_gru_new = self.decoder(keys, h_gru, last_key, mask)

        # Distribution
        probs = F.softmax(logits, dim=0)
        dist  = torch.distributions.Categorical(probs)

        if deterministic:
            action = logits.argmax().item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action, device=device))

        # Critic value
        h_queue_mean = h_stack[mask].mean(dim=0, keepdim=True) if mask.any() \
                       else torch.zeros(1, self.vnr_dim, device=device)
        value = self.critic(h_p, h_queue_mean)

        return action, log_prob, value, h_gru_new

    # ------------------------------------------------------------------
    # Full episode rollout (K steps)
    # ------------------------------------------------------------------

    def rollout(
        self,
        g_sub:      dgl.DGLGraph,
        vnr_graphs: List[Optional[dgl.DGLGraph]],
        K_real:     int,
        deterministic: bool = False,
    ) -> Dict:
        """
        Run one episode (one time window) and collect trajectory.

        Parameters
        ----------
        g_sub       : substrate DGLGraph (current state at start of window)
        vnr_graphs  : list of K_real DGLGraphs (the actual VNRs)
                      padded to K_max with None
        K_real      : actual number of VNRs (≤ K_max)
        deterministic : greedy if True, sample otherwise

        Returns dict with keys:
            actions, log_probs, values, mask_history
            (rewards and next-state update are added by the training loop)
        """
        h_p   = self.encode_substrate(g_sub.to(device))
        keys, h_stack = self.encode_vnr_queue(vnr_graphs)

        # Boolean mask: True = not yet selected AND is a real VNR
        mask = torch.zeros(self.K_max, dtype=torch.bool, device=device)
        mask[:K_real] = True

        # Initial GRU state from context
        h_queue_mean = h_stack[:K_real].mean(dim=0, keepdim=True)
        h_gru = self.context_mlp(h_p, h_queue_mean)   # (1, gru_hidden)
        last_key = self.start_token                     # (1, gru_hidden)

        trajectory = dict(actions=[], log_probs=[], values=[], masks=[])

        for _ in range(K_real):
            action, lp, val, h_gru = self.step(
                h_p, keys, h_stack, mask, h_gru, last_key, deterministic
            )

            trajectory['actions'].append(action)
            trajectory['log_probs'].append(lp)
            trajectory['values'].append(val)
            trajectory['masks'].append(mask.clone())

            # Update state: mark chosen VNR as done
            mask[action] = False
            last_key = keys[action:action+1]            # (1, gru_hidden)

        return trajectory

    # ------------------------------------------------------------------
    # Evaluate actions for PPO update (re-compute log_prob & entropy)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        g_sub:      dgl.DGLGraph,
        vnr_graphs: List[Optional[dgl.DGLGraph]],
        K_real:     int,
        actions:    List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-run episode with fixed actions to get new log_probs, values, entropy.
        Used in PPO update phase.

        Returns:
            log_probs : (K_real,)
            values    : (K_real,)
            entropies : (K_real,)
        """
        h_p   = self.encode_substrate(g_sub.to(device))
        keys, h_stack = self.encode_vnr_queue(vnr_graphs)

        mask = torch.zeros(self.K_max, dtype=torch.bool, device=device)
        mask[:K_real] = True

        h_queue_mean = h_stack[:K_real].mean(dim=0, keepdim=True)
        h_gru = self.context_mlp(h_p, h_queue_mean)
        last_key = self.start_token

        log_probs, values, entropies = [], [], []

        for t in range(K_real):
            logits, h_gru = self.decoder(keys, h_gru, last_key, mask)
            probs = F.softmax(logits, dim=0)
            dist  = torch.distributions.Categorical(probs)

            a = torch.tensor(actions[t], device=device)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

            h_queue_mean = h_stack[mask].mean(dim=0, keepdim=True) if mask.any() \
                           else torch.zeros(1, self.vnr_dim, device=device)
            values.append(self.critic(h_p, h_queue_mean))

            # Update for next step
            mask[actions[t]] = False
            last_key = keys[actions[t]:actions[t]+1]

        return (
            torch.stack(log_probs),
            torch.cat(values),
            torch.stack(entropies),
        )


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """
    Standalone PPO trainer for VNRSchedulerAgent.
    Integrates directly with hpso_batch.py as the reward oracle.

    Typical usage:
        trainer = PPOTrainer(agent, cfg)
        for episode in range(N):
            window = env.get_window()          # list of VNR graphs + substrate
            ordering = trainer.collect_and_update(window, substrate, hpso_fn)
    """

    def __init__(self, agent: VNRSchedulerAgent, cfg: dict = None):
        self.agent = agent.to(device)
        c = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg = c
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=c['lr'])

    # ------------------------------------------------------------------
    # GAE advantage computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_gae(rewards, values, gamma=0.98, lam=0.95):
        """
        rewards : list of floats length T
        values  : list of tensors length T  (detached scalars)
        Returns advantages and returns-to-go as tensors.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device)
        returns    = torch.zeros(T, device=device)

        gae = 0.0
        v_next = 0.0
        for t in reversed(range(T)):
            v_t   = values[t].item() if isinstance(values[t], torch.Tensor) else values[t]
            delta = rewards[t] + gamma * v_next - v_t
            gae   = delta + gamma * lam * gae
            advantages[t] = gae
            v_next = v_t

        returns = advantages + torch.tensor(
            [v.item() if isinstance(v, torch.Tensor) else v for v in values],
            device=device
        )
        return advantages, returns

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(result, vnr, penalty=0.1):
        """
        result : None (fail) or (mapping, link_paths) from hpso_embed
        Returns scalar float reward.
        """
        if result is None:
            return -penalty
        # R/C ratio as reward (same as GPRL paper eq. 21)
        from src.evaluation.eval import revenue_of_vnr, cost_of_vnr
        try:
            rev  = revenue_of_vnr(vnr)
            cost = cost_of_vnr(vnr, result[0], result[1])
            if cost <= 0:
                return 0.0
            return rev / cost
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Collect one episode + run PPO update
    # ------------------------------------------------------------------

    def collect_and_update(
        self,
        vnr_list:    list,        # list of VNR networkx graphs
        substrate,                # substrate networkx graph (will be mutated by HPSO)
        hpso_embed_fn,            # callable(substrate, vnr) → (mapping, paths) or None
        build_vnr_dgl_fn,         # callable(vnr_nx) → dgl.DGLGraph
        build_sub_dgl_fn,         # callable(substrate_nx) → dgl.DGLGraph
    ) -> Tuple[list, list, dict]:
        """
        Full collect-and-update cycle for one time window.

        Parameters
        ----------
        vnr_list        : list of K real VNR graphs (networkx)
        substrate       : mutable substrate graph (resources updated after each embed)
        hpso_embed_fn   : function(substrate_graph, vnr_graph) → result or None
        build_vnr_dgl_fn: converts a VNR nx graph to DGLGraph with ndata['nfeat']
        build_sub_dgl_fn: converts substrate nx graph to DGLGraph with ndata['nfeat']

        Returns
        -------
        accepted_order  : list of VNR objects that were accepted
        rejected        : list of rejected VNR objects
        metrics         : dict with acc_rate, avg_revenue, avg_cost, rc_ratio
        """
        K_real = len(vnr_list)
        K_max  = self.cfg['K_max']
        assert K_real <= K_max, f"K_real={K_real} exceeds K_max={K_max}"

        # --- Build DGL graphs for substrate and VNR queue ---
        g_sub_initial  = build_sub_dgl_fn(substrate).to(device)
        vnr_dgl_list   = [build_vnr_dgl_fn(v) for v in vnr_list]
        vnr_dgl_padded = vnr_dgl_list + [None] * (K_max - K_real)

        # --- Rollout (no grad) ---
        self.agent.eval()
        with torch.no_grad():
            traj = self.agent.rollout(g_sub_initial, vnr_dgl_padded, K_real)

        actions = traj['actions']        # list of K_real ints

        # --- Execute ordering via HPSO, collect rewards ---
        substrate_copy = copy.deepcopy(substrate)   # don't mutate original here
        rewards, accepted, rejected = [], [], []
        rev_list, cost_list = [], []

        for t, vnr_idx in enumerate(actions):
            vnr = vnr_list[vnr_idx]
            result = hpso_embed_fn(substrate_copy, vnr)

            r = self.compute_reward(result, vnr)
            rewards.append(r)

            if result is not None:
                mapping, link_paths = result
                accepted.append((vnr, mapping, link_paths))
                # resources already reserved by hpso_embed_fn on substrate_copy
                try:
                    from src.evaluation.eval import revenue_of_vnr, cost_of_vnr
                    rev_list.append(revenue_of_vnr(vnr))
                    cost_list.append(cost_of_vnr(vnr, mapping, link_paths))
                except Exception:
                    pass
            else:
                rejected.append(vnr)

        # --- PPO update ---
        advantages, returns = self.compute_gae(
            rewards,
            [v.detach() for v in traj['values']],
            self.cfg['gamma'], self.cfg['gae_lambda']
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Re-evaluate with grad for K_epochs
        self.agent.train()
        old_log_probs = torch.stack(traj['log_probs']).detach()

        for _ in range(self.cfg['k_epochs']):
            log_probs_new, values_new, entropies = self.agent.evaluate_actions(
                g_sub_initial, vnr_dgl_padded, K_real, actions
            )

            # PPO ratio
            ratio = torch.exp(log_probs_new - old_log_probs)

            # Clipped objective
            eps = self.cfg['clip_eps']
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values_new, returns)

            # Entropy bonus
            entropy_loss = -entropies.mean()

            total_loss = (policy_loss
                          + self.cfg['value_coef']  * value_loss
                          + self.cfg['entropy_coef'] * entropy_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()

        # --- Metrics ---
        acc_rate = len(accepted) / K_real if K_real > 0 else 0.0
        avg_rev  = sum(rev_list)  / len(rev_list)  if rev_list  else 0.0
        avg_cost = sum(cost_list) / len(cost_list) if cost_list else 0.0
        rc_ratio = sum(rev_list)  / sum(cost_list) if cost_list and sum(cost_list) > 0 else 0.0

        metrics = dict(
            acc_rate=acc_rate,
            avg_revenue=avg_rev,
            avg_cost=avg_cost,
            rc_ratio=rc_ratio,
            total_loss=total_loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
        )

        return accepted, rejected, metrics


# ---------------------------------------------------------------------------
# Utility: build DGL graph from networkx VNR/substrate
# ---------------------------------------------------------------------------

def build_vnr_dgl(vnr_nx, node_feat_keys=('cpu', 'mem'), vnf_type_key='vnf_type',
                  max_demand=100.0) -> dgl.DGLGraph:
    """
    Convert a networkx VNR DAG to a DGLGraph with ndata['nfeat'].

    nfeat columns:
      [0] cpu_demand_norm
      [1] mem_demand_norm
      [2] vnf_type_norm   (type_id / num_types)

    Adjust column extraction to match your actual VNR graph attributes.
    """
    nodes = list(vnr_nx.nodes())
    edges = list(vnr_nx.edges())

    if len(edges) == 0:
        # Self-loop to avoid isolated node issues with GAT
        u = torch.zeros(len(nodes), dtype=torch.long)
        v = torch.zeros(len(nodes), dtype=torch.long)
    else:
        u = torch.tensor([e[0] for e in edges], dtype=torch.long)
        v = torch.tensor([e[1] for e in edges], dtype=torch.long)

    g = dgl.graph((u, v), num_nodes=len(nodes))

    feats = []
    for n in nodes:
        nd = vnr_nx.nodes[n]
        cpu  = nd.get('cpu',      nd.get('cpu_demand',  0.0)) / max_demand
        mem  = nd.get('mem',      nd.get('mem_demand',  0.0)) / max_demand
        vnft = nd.get('vnf_type', nd.get('type',        0))   / 10.0  # normalise
        feats.append([cpu, mem, vnft])

    g.ndata['nfeat'] = torch.tensor(feats, dtype=torch.float32)
    return g


def build_substrate_dgl(sub_nx, total_cpu=50.0, total_mem=50.0,
                        total_bw=100.0) -> dgl.DGLGraph:
    """
    Convert substrate networkx graph to DGLGraph with ndata['nfeat'].

    nfeat columns:
      [0] cpu_residual_ratio
      [1] mem_residual_ratio
      [2] avg_bw_residual_ratio  (mean over incident edges)
      [3] utilisation            (1 - cpu_res/total)

    Adjust attribute names to match your substrate graph.
    """
    nodes = list(sub_nx.nodes())
    edges = list(sub_nx.edges())

    if len(edges) == 0:
        u = torch.zeros(len(nodes), dtype=torch.long)
        v = torch.zeros(len(nodes), dtype=torch.long)
    else:
        u = torch.tensor([e[0] for e in edges], dtype=torch.long)
        v = torch.tensor([e[1] for e in edges], dtype=torch.long)

    g = dgl.graph((u, v), num_nodes=len(nodes))

    feats = []
    for n in nodes:
        nd = sub_nx.nodes[n]
        cpu_res = nd.get('cpu_res', nd.get('cpu', total_cpu)) / total_cpu
        mem_res = nd.get('mem_res', nd.get('mem', total_mem)) / total_mem
        util    = 1.0 - cpu_res

        # Average BW ratio over incident edges
        nbrs = list(sub_nx.neighbors(n))
        if nbrs:
            bw_vals = [sub_nx[n][nb].get('bw_res', sub_nx[n][nb].get('bw', total_bw))
                       / total_bw for nb in nbrs]
            bw_avg = sum(bw_vals) / len(bw_vals)
        else:
            bw_avg = 1.0

        feats.append([cpu_res, mem_res, bw_avg, util])

    g.ndata['nfeat'] = torch.tensor(feats, dtype=torch.float32)
    return g


# ---------------------------------------------------------------------------
# Drop-in replacement for hpso_batch.py
# ---------------------------------------------------------------------------

def hpso_batch_rl(
    substrate,
    batch,
    agent: VNRSchedulerAgent,
    build_sub_dgl_fn=None,
    build_vnr_dgl_fn=None,
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
    Drop-in replacement for hpso_batch() in hpso_batch.py.

    Instead of sorting VNRs by revenue, the RL agent decides the ordering.
    If agent=None, falls back to revenue-descending sort (original behaviour).

    Parameters
    ----------
    substrate   : substrate networkx graph
    batch       : list of VNR graphs or list of (vnr, info) tuples
    agent       : trained VNRSchedulerAgent (or None for baseline fallback)
    build_*_fn  : optional custom DGL builders; defaults use build_vnr_dgl / build_substrate_dgl
    """
    from src.algorithms.fast_hpso import hpso_embed
    from src.evaluation.eval import revenue_of_vnr

    # Unpack batch
    if len(batch) > 0 and isinstance(batch[0], tuple):
        vnr_list = [vnr for vnr, _ in batch]
    else:
        vnr_list = list(batch)

    if agent is None or len(vnr_list) == 0:
        # Fallback: original revenue-sort
        vnr_list.sort(key=lambda x: revenue_of_vnr(x), reverse=True)
        order = list(range(len(vnr_list)))
    else:
        # --- RL agent determines ordering ---
        _build_sub = build_sub_dgl_fn or build_substrate_dgl
        _build_vnr = build_vnr_dgl_fn or build_vnr_dgl

        K_max  = agent.K_max
        K_real = min(len(vnr_list), K_max)

        g_sub   = _build_sub(substrate).to(device)
        vnr_dgl = [_build_vnr(v) for v in vnr_list[:K_real]]
        padded  = vnr_dgl + [None] * (K_max - K_real)

        agent.eval()
        with torch.no_grad():
            traj = agent.rollout(g_sub, padded, K_real, deterministic=True)

        order = traj['actions']   # list of K_real indices in chosen order
        # Any VNRs beyond K_max that couldn't be scheduled → append at end
        remaining = [i for i in range(len(vnr_list)) if i not in order]
        order = order + remaining

    # --- Execute HPSO in RL-determined order ---
    accepted, rejected = [], []

    def _embed(vnr):
        return hpso_embed(
            substrate_graph=substrate,
            vnr_graph=vnr,
            particles=particles,
            iterations=iterations,
            w_max=w_max, w_min=w_min,
            beta=beta, gamma=gamma,
            T0=T0, cooling_rate=cooling_rate,
        )

    for i, idx in enumerate(order):
        vnr = vnr_list[idx]
        if verbose:
            print(f"[HPSO-RL] Step {i+1}/{len(order)}, VNR idx={idx} "
                  f"(nodes={len(vnr.nodes())})")
        result = _embed(vnr)
        if result is not None:
            mapping, link_paths = result
            accepted.append((vnr, mapping, link_paths))
            if verbose:
                print(" -> Accepted")
        else:
            rejected.append(vnr)
            if verbose:
                print(" -> Rejected")

    return accepted, rejected
