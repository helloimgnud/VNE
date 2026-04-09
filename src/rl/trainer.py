"""
src/rl/trainer.py
=================
PPO trainer for VNRSchedulerAgent.

Training loop (idea.md §6.4):
  1. Collect rollout  → trajectory (s, a, r, log_π, V)
  2. Execute HPSO in RL-chosen order → rewards
  3. Compute GAE advantages
  4. PPO update for k_epochs with clipped surrogate + value + entropy losses

The trainer is designed to be used as a module:
    trainer = PPOTrainer(agent, cfg)
    metrics = trainer.collect_and_update(vnr_list, substrate, hpso_fn, ...)

It is the single place that owns the optimizer and gradient updates, keeping
the agent class pure (network + rollout logic only).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

from src.rl.agent import VNRSchedulerAgent
from src.rl.utils import DEFAULT_CFG, build_vnr_dgl, build_substrate_dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOTrainer:
    """
    Standalone PPO trainer.

    Parameters
    ----------
    agent : VNRSchedulerAgent
    cfg   : dict — overrides for DEFAULT_CFG (training hyper-parameters)
    """

    def __init__(self, agent: VNRSchedulerAgent, cfg: Optional[dict] = None):
        self.agent = agent.to(device)
        self.cfg   = {**DEFAULT_CFG, **(cfg or {})}
        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=self.cfg['lr'],
        )

    # ------------------------------------------------------------------
    # GAE advantage computation  (idea.md §5.2 / §6.2)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_gae(
        rewards: List[float],
        values:  List[torch.Tensor],
        gamma:   float = 0.98,
        lam:     float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation.

        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        Â_t = Σ_{k≥0} (γλ)^k · δ_{t+k}

        Returns
        -------
        advantages : (T,)  — Â_t
        returns    : (T,)  — Â_t + V(s_t)   [targets for value loss]
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device)

        gae    = 0.0
        v_next = 0.0
        for t in reversed(range(T)):
            v_t   = float(values[t].item()) if isinstance(values[t], torch.Tensor) \
                    else float(values[t])
            delta  = rewards[t] + gamma * v_next - v_t
            gae    = delta + gamma * lam * gae
            advantages[t] = gae
            v_next = v_t

        v_tensor = torch.tensor(
            [float(v.item()) if isinstance(v, torch.Tensor) else float(v)
             for v in values],
            device=device,
        )
        returns = advantages + v_tensor
        return advantages, returns

    # ------------------------------------------------------------------
    # Reward computation  (idea.md §3.3)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(result, vnr, penalty: float = 0.1) -> float:
        """
        r_t = Revenue/Cost  if HPSO succeeds
        r_t = -penalty      if HPSO fails
        """
        if result is None:
            return -penalty
        try:
            from src.evaluation.eval import revenue_of_vnr
            rev = revenue_of_vnr(vnr)
            # Simple cost proxy (full substrate cost needs substrate graph;
            # if cost_of_embedding is needed import from eval.py)
            cost = (sum(vnr.nodes[n].get('cpu', 1.0) for n in vnr.nodes()) +
                    sum(vnr.edges[e].get('bw', 1.0)  for e in vnr.edges()))
            return rev / max(cost, 1e-6)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main collect-and-update cycle
    # ------------------------------------------------------------------

    def collect_and_update(
        self,
        vnr_list:        list,
        substrate,
        hpso_embed_fn:   Callable,
        build_vnr_fn:    Optional[Callable] = None,
        build_sub_fn:    Optional[Callable] = None,
    ) -> Tuple[list, list, dict]:
        """
        One complete collect-and-update cycle for a single time window.

        Parameters
        ----------
        vnr_list       : list of K networkx VNR graphs
        substrate      : substrate networkx graph (a deep copy is used internally)
        hpso_embed_fn  : callable(substrate, vnr) → (mapping, paths) | None
        build_vnr_fn   : optional VNR → DGL builder (default: build_vnr_dgl)
        build_sub_fn   : optional substrate → DGL builder (default: build_substrate_dgl)

        Returns
        -------
        accepted   : list of (vnr, mapping, link_paths)
        rejected   : list of vnr
        metrics    : dict with acc_rate, rc_ratio, total_loss, ...
        """
        K_real = len(vnr_list)
        K_max  = self.cfg['K_max']
        assert K_real <= K_max, (
            f"K_real={K_real} exceeds K_max={K_max}. "
            f"Increase K_max in cfg or reduce window size."
        )

        _build_vnr = build_vnr_fn or build_vnr_dgl
        _build_sub = build_sub_fn or build_substrate_dgl

        # --- Build DGL graphs ---
        g_sub_init   = _build_sub(substrate).to(device)
        vnr_dgl_list = [_build_vnr(v) for v in vnr_list]
        vnr_padded   = vnr_dgl_list + [None] * (K_max - K_real)

        # --- Collect trajectory (no grad) ---
        self.agent.eval()
        with torch.no_grad():
            traj = self.agent.rollout(g_sub_init, vnr_padded, K_real,
                                       deterministic=False)

        actions = traj['actions']   # list[int]

        # --- Execute HPSO in RL order, collect rewards ---
        sub_copy              = copy.deepcopy(substrate)
        rewards: List[float]  = []
        accepted: list        = []
        rejected: list        = []
        rev_list: List[float] = []
        cost_list: List[float]= []

        for vnr_idx in actions:
            vnr    = vnr_list[vnr_idx]
            result = hpso_embed_fn(sub_copy, vnr)

            r = self.compute_reward(result, vnr, self.cfg.get('penalty', 0.1))
            rewards.append(r)

            if result is not None:
                mapping, link_paths = result
                accepted.append((vnr, mapping, link_paths))
                try:
                    from src.evaluation.eval import revenue_of_vnr
                    rev = revenue_of_vnr(vnr)
                    cost = (sum(vnr.nodes[n].get('cpu', 1.0) for n in vnr.nodes()) +
                            sum(vnr.edges[e].get('bw',  1.0) for e in vnr.edges()))
                    rev_list.append(rev)
                    cost_list.append(cost)
                except Exception:
                    pass
            else:
                rejected.append(vnr)

        # --- GAE ---
        advantages, returns = self.compute_gae(
            rewards,
            [v.detach() for v in traj['values']],
            self.cfg['gamma'],
            self.cfg['gae_lambda'],
        )
        # Normalise advantages (reduce variance)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO update (k_epochs) ---
        self.agent.train()
        old_log_probs = torch.stack(traj['log_probs']).detach()

        total_loss_val = 0.0
        policy_loss_val = 0.0
        value_loss_val  = 0.0

        for _ in range(self.cfg['k_epochs']):
            log_probs_new, values_new, entropies = self.agent.evaluate_actions(
                g_sub_init, vnr_padded, K_real, actions
            )

            # PPO ratio
            ratio = torch.exp(log_probs_new - old_log_probs)

            # Clipped surrogate (idea.md §6.1)
            eps   = self.cfg['clip_eps']
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (idea.md §5.2)
            value_loss = F.mse_loss(values_new, returns)

            # Entropy bonus (idea.md §6.1)
            entropy_loss = -entropies.mean()

            loss = (policy_loss
                    + self.cfg['value_coef']   * value_loss
                    + self.cfg['entropy_coef'] * entropy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss_val  = loss.item()
            policy_loss_val = policy_loss.item()
            value_loss_val  = value_loss.item()

        # --- Metrics ---
        n_acc    = len(accepted)
        acc_rate = n_acc / K_real if K_real > 0 else 0.0
        rc_ratio = (sum(rev_list) / sum(cost_list)
                    if cost_list and sum(cost_list) > 0 else 0.0)

        metrics = dict(
            acc_rate     = acc_rate,
            rc_ratio     = rc_ratio,
            avg_revenue  = sum(rev_list)  / len(rev_list)  if rev_list  else 0.0,
            avg_cost     = sum(cost_list) / len(cost_list) if cost_list else 0.0,
            total_loss   = total_loss_val,
            policy_loss  = policy_loss_val,
            value_loss   = value_loss_val,
            n_accepted   = n_acc,
            n_rejected   = K_real - n_acc,
        )

        return accepted, rejected, metrics

    # ------------------------------------------------------------------
    # Model checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str, history: Optional[list] = None):
        """Save agent weights + config (and optional training history)."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state': self.agent.state_dict(),
            'cfg':         self.cfg,
            'history':     history or [],
        }, path)
        print(f"[PPOTrainer] Model saved → {path}")

    def load(self, path: str):
        """Load agent weights from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        self.agent.load_state_dict(ckpt['model_state'])
        print(f"[PPOTrainer] Model loaded ← {path}")
        return ckpt.get('history', [])
