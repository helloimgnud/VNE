"""
src/training/train_ppo_v2.py
============================
PPO v2 Trainer — Episode-Aligned Training Pipeline.

1 epoch = 1 full episode (all N VNRs on 1 substrate).
GAE is computed over the full episode → last_value = 0 (done=True always).

Key design decisions:
  - train_env and eval_env are COMPLETELY SEPARATE instances.
  - eval_env runs on the held-out eval dataset (different seed).
  - evaluate() re-runs the SAME eval dataset n_episodes times to average
    out HPSO stochasticity → smooth, meaningful plots.

Usage:
    python -m src.training.train_ppo_v2 \\
        --train-dir dataset/rl_training/train \\
        --eval-dir  dataset/rl_training/eval  \\
        --num-epochs 500 \\
        --eval-every 10 \\
        --run-name ppo_v2_run1 \\
        --save-dir checkpoints
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from src.rl.environment_v2 import DataMode, VNEEnvironmentV2
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
from src.scheduler.model import VNRScheduler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfigV2:
    # --- Data paths (1 dataset per run) ---
    train_dir: str = ""
    eval_dir:  str = ""
    window_size: int = 50
    max_queue_delay: int = 100

    # --- PPO Hyper-params ---
    num_epochs: int   = 500
    batch_size: int   = 64
    n_ppo_epochs: int = 8
    lr: float         = 3e-4
    gamma: float      = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float   = 0.01
    vf_coef: float    = 0.5
    grad_clip: float  = 0.5

    # --- HPSO ---
    hpso_particles:  int = 20
    hpso_iterations: int = 10      # Reduced for training speed

    # --- Logging / Eval ---
    eval_every:    int = 10
    eval_episodes: int = 3
    log_dir:  str = "runs"
    save_dir: str = "checkpoints"
    save_every: int = 0
    run_name: str = "ppo_v2"
    device:   str = "auto"
    load_checkpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# GNN Actor-Critic (wraps VNRScheduler)
# ---------------------------------------------------------------------------

class GNNActorCriticV2(nn.Module):
    """
    Actor-Critic that reuses the existing VNRScheduler as the actor.

    Actor:  scores = scheduler(substrate, vnr_list) → Categorical distribution
    Critic: value  = linear(substrate_embedding)
    """

    def __init__(self, scheduler: VNRScheduler, substrate_emb_dim: int = 128):
        super().__init__()
        self.scheduler   = scheduler
        # Independent critic encoder to prevent Value gradients from destroying the Actor's GNN
        self.critic_encoder = copy.deepcopy(scheduler.substrate_encoder)
        self.value_head  = nn.Linear(substrate_emb_dim, 1)

    def forward(self, obs: dict):
        """
        Parameters
        ----------
        obs : dict with keys "substrate" (PyG Data) and "vnr_list" (list of PyG Data)

        Returns
        -------
        dist  : Categorical distribution over vnr_list
        value : scalar Tensor
        """
        sub_data = obs["substrate"]
        vnr_list = obs["vnr_list"]

        # Actor
        scores = self.scheduler(sub_data, vnr_list)   # [B]
        dist   = Categorical(logits=scores)

        # Critic — encode substrate using the independent critic encoder
        h_s_critic = self.critic_encoder(sub_data)           # [1, 128]
        value = self.value_head(h_s_critic).squeeze(-1)      # [1] or scalar
        return dist, value

    def get_action_and_value(self, obs: dict, action: Optional[torch.Tensor] = None):
        """
        Sample action and compute log_prob + entropy + value.
        If `action` is provided (replay buffer), compute log_prob for that action.
        """
        dist, value = self.forward(obs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# PPO Trainer v2
# ---------------------------------------------------------------------------

class PPOTrainerV2:

    def __init__(self, cfg: PPOConfigV2):
        self.cfg = cfg

        # Device
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        print(f"[PPOTrainerV2] Device: {self.device}")

        # HPSO params dict
        hpso_params = {
            "particles":  cfg.hpso_particles,
            "iterations": cfg.hpso_iterations,
        }

        # Environments — completely separate, never share state
        sub_path, vnr_path = self._sample_replica(cfg.train_dir)
        self.train_env = VNEEnvironmentV2(
            substrate_path=sub_path,
            vnr_path=vnr_path,
            window_size=cfg.window_size,
            max_queue_delay=cfg.max_queue_delay,
            hpso_params=hpso_params,
        )
        sub_path_eval, vnr_path_eval = self._sample_replica(cfg.eval_dir)
        self.eval_env = VNEEnvironmentV2(
            substrate_path=sub_path_eval,
            vnr_path=vnr_path_eval,
            window_size=cfg.window_size,
            max_queue_delay=cfg.max_queue_delay,
            hpso_params=hpso_params,
        )

        # Model
        scheduler = VNRScheduler(use_batch_context=True)
        self.ac   = GNNActorCriticV2(scheduler).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr, eps=1e-5)

        if cfg.load_checkpoint:
            self._load_checkpoint(cfg.load_checkpoint)
            print(f"[PPOTrainerV2] Loaded checkpoint: {cfg.load_checkpoint}")

        # TensorBoard
        if _TB_AVAILABLE:
            log_path = os.path.join(cfg.log_dir, cfg.run_name)
            self.writer = SummaryWriter(log_path)
            print(f"[PPOTrainerV2] TensorBoard → {log_path}")
        else:
            self.writer = None
            print("[PPOTrainerV2] TensorBoard not available; install tensorboard.")

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.best_eval_ar = -1.0
        self.global_step  = 0

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def _sample_replica(self, base_dir: str) -> Tuple[str, str]:
        """Returns (substrate_path, vnr_path) randomly sampled from available replicas."""
        import random
        if not os.path.exists(base_dir):
            return os.path.join(base_dir, "substrate.json"), os.path.join(base_dir, "vnr_stream.json")
        replicas = [d for d in os.listdir(base_dir) if d.startswith("replica_")]
        if not replicas:
            return os.path.join(base_dir, "substrate.json"), os.path.join(base_dir, "vnr_stream.json")
        rep = random.choice(replicas)
        return os.path.join(base_dir, rep, "substrate.json"), os.path.join(base_dir, rep, "vnr_stream.json")

    def _collect_one_episode(self) -> List[dict]:
        """
        Collect ONE complete episode from train_env.
        """
        # Load a random replica before starting the episode
        sub_path, vnr_path = self._sample_replica(self.cfg.train_dir)
        self.train_env.load_dataset(sub_path, vnr_path)
        
        transitions = []
        obs, _      = self.train_env.reset()
        done        = False

        while not done:
            if not obs["vnr_list"]:
                # Should never happen — env handles this in _check_done_and_advance
                raise RuntimeError("[PPOTrainerV2] vnr_list is empty mid-episode — bug in env")

            sub_data  = obs["substrate"].to(self.device)
            vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]
            obs_dev   = {"substrate": sub_data, "vnr_list": vnr_datas}

            with torch.no_grad():
                action, log_prob, _, value = self.ac.get_action_and_value(obs_dev)

            next_obs, reward, done, _, info = self.train_env.step(action.item())

            transitions.append({
                "obs":      obs,
                "action":   action.cpu(),
                "log_prob": log_prob.cpu(),
                "value":    value.squeeze().cpu(),
                "reward":   float(reward),
                "done":     done,
            })

            obs = next_obs
            self.global_step += 1

        return transitions

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(self, transitions: List[dict]):
        """
        GAE over the FULL episode.

        Because the episode always ends with done=True, last_value = 0.
        No bootstrapping from next_obs needed.
        """
        T       = len(transitions)
        gamma   = self.cfg.gamma
        lam     = self.cfg.gae_lambda

        values  = [t["value"].item() for t in transitions]
        rewards = [t["reward"]       for t in transitions]
        dones   = [t["done"]         for t in transitions]

        advantages = [0.0] * T
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val        = values[t + 1] if t < T - 1 else 0.0
            next_nonterminal = 0.0 if dones[t] else 1.0

            delta    = rewards[t] + gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae

        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        values_t     = torch.tensor(values,     dtype=torch.float32)
        returns_t    = advantages_t + values_t
        return advantages_t, returns_t

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update(
        self,
        transitions: List[dict],
        advantages:  torch.Tensor,
        returns:     torch.Tensor,
    ) -> dict:
        """
        Mini-batch PPO update over the collected transitions.
        Returns a dict of loss scalars for logging.
        """
        T   = len(transitions)
        cfg = self.cfg

        # Normalise advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = torch.stack([t["log_prob"] for t in transitions])  # [T]
        old_actions   = torch.stack([t["action"]   for t in transitions])  # [T]
        old_values    = torch.stack([t["value"]    for t in transitions])  # [T]

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_kl          = 0.0
        total_clip_frac   = 0.0
        n_updates         = 0

        for _ in range(cfg.n_ppo_epochs):
            # Shuffle indices for mini-batches
            indices = torch.randperm(T)

            for start in range(0, T, cfg.batch_size):
                mb_idx = indices[start : start + cfg.batch_size]

                # Re-evaluate log_probs and values for mini-batch
                new_log_probs_list = []
                new_values_list    = []
                entropies_list     = []

                for i in mb_idx.tolist():
                    t          = transitions[i]
                    sub_data   = t["obs"]["substrate"].to(self.device)
                    vnr_datas  = [v.to(self.device) for v in t["obs"]["vnr_list"]]
                    obs_dev    = {"substrate": sub_data, "vnr_list": vnr_datas}
                    act        = old_actions[i].to(self.device)

                    _, lp, ent, val = self.ac.get_action_and_value(obs_dev, action=act)
                    new_log_probs_list.append(lp)
                    new_values_list.append(val.squeeze())
                    entropies_list.append(ent)

                new_lp  = torch.stack(new_log_probs_list)
                new_val = torch.stack(new_values_list)
                ent_t   = torch.stack(entropies_list)

                mb_adv     = adv[mb_idx].to(self.device)
                mb_ret     = returns[mb_idx].to(self.device)
                mb_old_lp  = old_log_probs[mb_idx].to(self.device)
                mb_old_val = old_values[mb_idx].to(self.device)

                # Policy (clipped surrogate)
                ratio       = (new_lp - mb_old_lp).exp()
                pg1         = -mb_adv * ratio
                pg2         = -mb_adv * ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                policy_loss = torch.max(pg1, pg2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (new_val - mb_ret) ** 2
                v_clipped = mb_old_val + torch.clamp(
                    new_val - mb_old_val,
                    -cfg.clip_range,
                    cfg.clip_range,
                )
                v_loss_clipped = (v_clipped - mb_ret) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()

                # Entropy bonus
                entropy_loss = -ent_t.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.grad_clip)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    kl        = (mb_old_lp - new_lp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += (-entropy_loss).item()
                total_kl          += kl
                total_clip_frac   += clip_frac
                n_updates         += 1

        n = max(n_updates, 1)
        return {
            "PolicyLoss":  total_policy_loss / n,
            "ValueLoss":   total_value_loss  / n,
            "Entropy":     total_entropy     / n,
            "ApproxKL":    total_kl          / n,
            "ClipFraction": total_clip_frac  / n,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: Optional[int] = None) -> dict:
        """
        Greedy rollout on self.eval_env.
        self.train_env is NOT touched during evaluation.

        Runs the SAME eval dataset n_episodes times to average out HPSO
        stochasticity, giving smooth, meaningful eval plots.
        """
        n_episodes = n_episodes or self.cfg.eval_episodes
        self.ac.eval()

        all_ars, all_rcs, all_rewards = [], [], []

        with torch.no_grad():
            for _ in range(n_episodes):
                # Evaluate on a random eval replica
                sub_path, vnr_path = self._sample_replica(self.cfg.eval_dir)
                self.eval_env.load_dataset(sub_path, vnr_path)
                
                obs, _ = self.eval_env.reset()
                ep_reward = 0.0
                done = False

                while not done:
                    if not obs["vnr_list"]:
                        break  # Safety guard (should not occur)

                    sub_data  = obs["substrate"].to(self.device)
                    vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]
                    obs_dev   = {"substrate": sub_data, "vnr_list": vnr_datas}

                    dist, _ = self.ac(obs_dev)
                    action  = dist.logits.argmax()        # Greedy
                    obs, reward, done, _, _ = self.eval_env.step(action.item())
                    ep_reward += reward

                summary = self.eval_env.episode_summary()
                all_ars.append(summary["acceptance_rate"])
                all_rcs.append(summary["revenue_cost_ratio"])
                all_rewards.append(ep_reward)

        self.ac.train()

        ar_tensor = torch.tensor(all_ars, dtype=torch.float32)
        return {
            "Eval/AcceptanceRate":   ar_tensor.mean().item(),
            "Eval/RevenueCostRatio": sum(all_rcs) / len(all_rcs),
            "Eval/EpisodeReward":    sum(all_rewards) / len(all_rewards),
            "Eval/AR_std":           ar_tensor.std().item() if len(all_ars) > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_train(self, epoch: int, ep_summary: dict, loss_dict: dict):
        if self.writer is None:
            return

        step = epoch

        # PPO losses
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"Train/{k}", v, step)

        # Episode metrics
        self.writer.add_scalar("Train/EpisodeReward",   ep_summary.get("total_reward",      0.0), step)
        self.writer.add_scalar("Train/AcceptanceRate",  ep_summary.get("acceptance_rate",    0.0), step)
        self.writer.add_scalar("Train/RevenueCostRatio",ep_summary.get("revenue_cost_ratio", 0.0), step)
        self.writer.add_scalar("Train/NFailed",         ep_summary.get("n_rejected",         0),   step)

        # Substrate utilisation at end of episode
        util = self.train_env.substrate_utilisation()
        self.writer.add_scalar("Substrate/CpuUtilization", util["cpu_util"], step)
        self.writer.add_scalar("Substrate/BwUtilization",  util["bw_util"],  step)
        self.writer.add_scalar("Substrate/ActiveEmbeddingsPeak",
                               len(self.train_env.active_embeddings), step)
        self.writer.add_scalar("Dataset/EpochWindow",
                               self.train_env.window_idx / max(self.train_env.total_windows, 1), step)

    def _log_eval(self, epoch: int, eval_metrics: dict):
        if self.writer is None:
            return
        for k, v in eval_metrics.items():
            self.writer.add_scalar(k, v, epoch)

    def _maybe_save_best(self, epoch: int, eval_metrics: dict):
        ar = eval_metrics.get("Eval/AcceptanceRate", 0.0)
        if ar > self.best_eval_ar:
            self.best_eval_ar = ar
            path = os.path.join(self.cfg.save_dir, f"{self.cfg.run_name}_best.pt")
            self._save_checkpoint(path, epoch=epoch, eval_ar=ar)
            print(f"  ✓ New best AR={ar:.4f} — saved → {path}")

    def _save_checkpoint(self, path: str, **meta):
        ckpt = {
            "ac_state_dict":   self.ac.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch":           meta.get("epoch", -1),
            "eval_ar":         meta.get("eval_ar", 0.0),
            "cfg":             self.cfg.__dict__,
        }
        torch.save(ckpt, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        for epoch in range(num_epochs):
            transitions = collect_one_episode()    ← all N VNRs
            advantages, returns = compute_gae(transitions)
            loss_dict = update(transitions, advantages, returns)
            log_train(epoch, ...)
            if epoch % eval_every == 0:
                eval_metrics = evaluate()
                log_eval(epoch, eval_metrics)
                maybe_save_best(epoch, eval_metrics)
        """
        print(f"\n{'='*60}")
        print(f"  PPO v2 Training: {self.cfg.run_name}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.cfg.num_epochs} | Eval every: {self.cfg.eval_every}")
        print(f"{'='*60}\n")

        t0 = time.time()

        for epoch in range(self.cfg.num_epochs):
            t_ep = time.time()

            # 1. Collect one full episode
            transitions = self._collect_one_episode()
            ep_summary  = self.train_env.episode_summary()

            # 2. GAE
            advantages, returns = self._compute_gae(transitions)

            # 3. PPO update
            loss_dict = self._update(transitions, advantages, returns)

            # 4. Log training metrics
            self._log_train(epoch, ep_summary, loss_dict)

            # 5. Console log
            ep_time = time.time() - t_ep
            ar  = ep_summary["acceptance_rate"]
            rew = ep_summary["total_reward"]
            print(
                f"Epoch {epoch+1:4d}/{self.cfg.num_epochs} | "
                f"AR={ar:.3f} | Rew={rew:+.3f} | "
                f"PL={loss_dict['PolicyLoss']:.4f} | "
                f"VL={loss_dict['ValueLoss']:.4f} | "
                f"Ent={loss_dict['Entropy']:.3f} | "
                f"{ep_time:.1f}s"
            )

            # 6. Periodic evaluation
            if (epoch + 1) % self.cfg.eval_every == 0:
                print(f"  → Evaluating on held-out dataset ({self.cfg.eval_episodes} runs)...")
                eval_metrics = self.evaluate()
                self._log_eval(epoch, eval_metrics)
                self._maybe_save_best(epoch, eval_metrics)
                print(
                    f"  Eval: AR={eval_metrics['Eval/AcceptanceRate']:.4f} "
                    f"± {eval_metrics['Eval/AR_std']:.4f} | "
                    f"R/C={eval_metrics['Eval/RevenueCostRatio']:.4f}"
                )

            # 7. Periodic checkpoint saving
            if self.cfg.save_every > 0 and (epoch + 1) % self.cfg.save_every == 0:
                periodic_path = os.path.join(self.cfg.save_dir, f"{self.cfg.run_name}_epoch{epoch+1}.pt")
                self._save_checkpoint(periodic_path, epoch=epoch)
                print(f"  [Save] Periodic checkpoint → {periodic_path}")

        # Final save
        final_path = os.path.join(self.cfg.save_dir, f"{self.cfg.run_name}_final.pt")
        self._save_checkpoint(final_path, epoch=self.cfg.num_epochs - 1)
        print(f"\n[Done] Total time: {(time.time()-t0)/60:.1f} min")
        print(f"[Done] Final checkpoint → {final_path}")
        print(f"[Done] Best eval AR: {self.best_eval_ar:.4f}")

        if self.writer:
            self.writer.close()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train PPO v2 for VNR Ordering")

    p.add_argument("--config", type=str, default="config.json", help="Path to config file")
    temp_args, _ = p.parse_known_args()
    
    import json
    config_data = {}
    if os.path.exists(temp_args.config):
        with open(temp_args.config, 'r') as f:
            config_data = json.load(f).get("training", {})

    # Data
    p.add_argument("--train-dir", default=config_data.get("train_dir", ""), help="Directory with substrate.json + vnr_stream.json for training")
    p.add_argument("--eval-dir",  default=config_data.get("eval_dir", ""), help="Directory with substrate.json + vnr_stream.json for evaluation")
    p.add_argument("--window-size",     type=int, default=config_data.get("window_size", 50))
    p.add_argument("--max-queue-delay", type=int, default=config_data.get("max_queue_delay", 100))

    # PPO
    p.add_argument("--num-epochs",    type=int,   default=config_data.get("num_epochs", 500))
    p.add_argument("--batch-size",    type=int,   default=config_data.get("batch_size", 64))
    p.add_argument("--n-ppo-epochs",  type=int,   default=config_data.get("n_ppo_epochs", 8))
    p.add_argument("--lr",            type=float, default=config_data.get("lr", 3e-4))
    p.add_argument("--gamma",         type=float, default=config_data.get("gamma", 0.99))
    p.add_argument("--gae-lambda",    type=float, default=config_data.get("gae_lambda", 0.95))
    p.add_argument("--clip-range",    type=float, default=config_data.get("clip_range", 0.2))
    p.add_argument("--ent-coef",      type=float, default=config_data.get("ent_coef", 0.01))
    p.add_argument("--vf-coef",       type=float, default=config_data.get("vf_coef", 0.5))
    p.add_argument("--grad-clip",     type=float, default=config_data.get("grad_clip", 0.5))

    # HPSO
    p.add_argument("--hpso-particles",  type=int, default=config_data.get("hpso_particles", 20))
    p.add_argument("--hpso-iterations", type=int, default=config_data.get("hpso_iterations", 10))

    # Logging
    p.add_argument("--eval-every",    type=int, default=config_data.get("eval_every", 10))
    p.add_argument("--eval-episodes", type=int, default=config_data.get("eval_episodes", 3))
    p.add_argument("--log-dir",  default=config_data.get("log_dir", "runs"))
    p.add_argument("--save-dir", default=config_data.get("save_dir", "checkpoints"))
    p.add_argument("--save-every", type=int, default=config_data.get("save_every", 0), help="Save a checkpoint every N epochs (0 to disable)")
    p.add_argument("--run-name", default=config_data.get("run_name", "ppo_v2"))
    p.add_argument("--device",   default=config_data.get("device", "auto"))
    p.add_argument("--load-checkpoint", default=config_data.get("load_checkpoint", None))

    return p.parse_args()


def main():
    args = _parse_args()

    cfg = PPOConfigV2(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        window_size=args.window_size,
        max_queue_delay=args.max_queue_delay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        n_ppo_epochs=args.n_ppo_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        grad_clip=args.grad_clip,
        hpso_particles=args.hpso_particles,
        hpso_iterations=args.hpso_iterations,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        save_every=args.save_every,
        run_name=args.run_name,
        device=args.device,
        load_checkpoint=args.load_checkpoint,
    )

    trainer = PPOTrainerV2(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
