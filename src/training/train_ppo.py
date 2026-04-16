"""
src/training/train_ppo.py
==========================
Phase 2 — Custom PPO training loop for VNRScheduler.

This implements a CleanRL-style PPO trainer without depending on
Stable-Baselines3, which cannot directly handle variable-action-space
environments. The actor-critic network is ``GNNActorCritic`` from
``src/scheduler/policy.py``.

Algorithm (Proximal Policy Optimisation with GAE)
-------------------------------------------------
For each PPO epoch:
  1. Collect N timesteps (rollout phase):
     - Run GNNActorCritic.forward(obs) → dist, value
     - Sample action, record log_prob, value, reward, done
  2. Compute advantages using Generalised Advantage Estimation (GAE)
  3. Mini-batch gradient update:
     - Re-evaluate log_prob and entropy under current policy
     - Surrogate clip loss (actor)
     - MSE value loss (critic)
     - Entropy bonus

Usage (plugin mode)
-------------------
>>> from src.training.train_ppo import PPOConfig, PPOTrainerScheduler
>>> cfg = PPOConfig(total_timesteps=500_000, reward_mode="revenue")
>>> trainer = PPOTrainerScheduler(cfg)
>>> trainer.train()

Or from command line:
  python -m src.training.train_ppo --total-steps 500000 --reward revenue
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.scheduler.model import VNRScheduler
from src.scheduler.policy import GNNActorCritic
from src.scheduler.environment import VNEOrderingEnv
from src.scheduler.rewards import RewardMode
from src.training.generate_data import make_env_fns


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All hyper-parameters for the PPO training run."""

    # Training budget
    total_timesteps: int   = 500_000
    n_steps:         int   = 512    # timesteps per rollout
    batch_size:      int   = 64     # mini-batch size for gradient update
    n_epochs:        int   = 10     # gradient epochs per rollout

    # PPO hyper-parameters
    lr:              float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_range:      float = 0.2
    ent_coef:        float = 0.01   # entropy bonus coefficient
    vf_coef:         float = 0.5    # value function loss coefficient
    grad_clip:       float = 0.5

    # Reward
    reward_mode:     str   = "revenue"  # simple | revenue | longterm

    # Network
    use_batch_context: bool = True   # Phase 2: enable BatchContextEncoder

    # Environment
    substrate_nodes:  int   = 50
    batch_size_env:   int   = 10
    # VNR size range: each VNR in a batch independently samples its node count
    # uniformly from [vnr_min_nodes, vnr_max_nodes], giving a varied dataset.
    vnr_min_nodes:    int   = 2
    vnr_max_nodes:    int   = 8
    fixed_substrate:  bool  = False
    hpso_particles:   int   = 20
    hpso_iterations:  int   = 30

    # Logging / checkpointing
    log_every:  int = 10_000      # log every N timesteps
    save_every: int = 50_000
    save_dir:   str = "checkpoints"
    run_name:   str = "ppo_phase2"
    device:     str = "auto"

    # Checkpoint to resume from (Optional)
    load_checkpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class _RolloutBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self):
        self.obs:       List[dict]             = []
        self.actions:   List[torch.Tensor]     = []
        self.log_probs: List[torch.Tensor]     = []
        self.values:    List[torch.Tensor]     = []
        self.rewards:   List[float]            = []
        self.dones:     List[bool]             = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainerScheduler:
    """
    Custom PPO trainer with GAE for the variable-action-space VNE environment.

    Parameters
    ----------
    cfg : PPOConfig
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # Networks
        if cfg.load_checkpoint and os.path.exists(cfg.load_checkpoint):
            print(f"[PPO] Loading checkpoint from: {cfg.load_checkpoint}")
            # VNRScheduler.load() is a classmethod that returns a new instance;
            # assign it explicitly (the old code silently discarded the return value).
            scheduler = VNRScheduler.load(cfg.load_checkpoint)
        else:
            scheduler = VNRScheduler(use_batch_context=cfg.use_batch_context)

        self.ac = GNNActorCritic(scheduler).to(self.device)

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

        # Environment
        substrate_fn, batch_fn = make_env_fns(
            substrate_nodes = cfg.substrate_nodes,
            batch_size      = cfg.batch_size_env,
            vnr_min_nodes   = cfg.vnr_min_nodes,
            vnr_max_nodes   = cfg.vnr_max_nodes,
            fixed_substrate = cfg.fixed_substrate,
        )
        self.env = VNEOrderingEnv(
            substrate_fn = substrate_fn,
            batch_fn     = batch_fn,
            hpso_params  = dict(
                particles   = cfg.hpso_particles,
                iterations  = cfg.hpso_iterations,
            ),
            reward_mode  = cfg.reward_mode,
        )

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.buffer  = _RolloutBuffer()
        self.history: List[dict] = []
        
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join("runs", cfg.run_name)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Current episode state
        self._obs, _ = self.env.reset()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, n_steps: int):
        """Collect n_steps of (obs, action, log_prob, value, reward, done)."""
        self.buffer.clear()
        self.ac.eval()

        for _ in range(n_steps):
            obs = self._obs

            if not obs["vnr_list"]:
                # Episode ended between steps; reset
                self._obs, _ = self.env.reset()
                obs          = self._obs

            sub_data  = obs["substrate"].to(self.device)
            vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]
            obs_dev   = {"substrate": sub_data, "vnr_list": vnr_datas}

            with torch.no_grad():
                action, log_prob, _, value = self.ac.get_action_and_value(obs_dev)

            next_obs, reward, done, _, info = self.env.step(action.item())

            self.buffer.obs.append(obs)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.values.append(value.squeeze())
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            
            if done:
                # Log metrics for completed episode
                ep_info = self.env.episode_summary()
                global_step = getattr(self, "global_step", 0) + len(self.buffer)
                self.writer.add_scalar("Metrics/AcceptanceRate", ep_info["acc_rate"], global_step)
                self.writer.add_scalar("Metrics/RevenueCostRatio", ep_info["rc_ratio"], global_step)
                self.writer.add_scalar("Metrics/SubstrateCpuUtil", ep_info.get("cpu_util", 0.0), global_step)
                self.writer.add_scalar("Metrics/SubstrateBwUtil", ep_info.get("bw_util", 0.0), global_step)

            if done:
                self._obs, _ = self.env.reset()
            else:
                self._obs = next_obs

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self, last_obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and value targets.

        Returns
        -------
        advantages : Tensor  [T]
        returns    : Tensor  [T]  (value targets)
        """
        cfg      = self.cfg
        T        = len(self.buffer.rewards)
        rewards  = self.buffer.rewards
        dones    = self.buffer.dones
        values   = [v.item() for v in self.buffer.values]

        # Bootstrap value for last state
        if last_obs["vnr_list"]:
            sub  = last_obs["substrate"].to(self.device)
            vnrs = [v.to(self.device) for v in last_obs["vnr_list"]]
            with torch.no_grad():
                last_val = self.ac.get_value({"substrate": sub, "vnr_list": vnrs}).item()
        else:
            last_val = 0.0

        advantages = [0.0] * T
        last_gae   = 0.0
        next_val   = last_val

        for t in reversed(range(T)):
            if dones[t]:
                next_val   = 0.0
                last_gae   = 0.0

            delta    = rewards[t] + cfg.gamma * next_val - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * last_gae
            advantages[t] = last_gae
            next_val      = values[t]

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns    = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update(self, advantages: torch.Tensor, returns: torch.Tensor):
        """Run n_epochs of mini-batch PPO gradient updates."""
        cfg = self.cfg
        T   = len(self.buffer.obs)

        old_log_probs = torch.stack(self.buffer.log_probs).detach().to(self.device)  # [T]

        # Normalise advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        self.ac.train()
        for _ in range(cfg.n_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, cfg.batch_size):
                mb_idx = indices[start : start + cfg.batch_size]

                # Re-evaluate log-probs and values on mini-batch
                mb_log_probs, mb_values, mb_entropy = [], [], []

                for i in mb_idx.tolist():
                    obs       = self.buffer.obs[i]
                    sub_data  = obs["substrate"].to(self.device)
                    vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]
                    obs_dev   = {"substrate": sub_data, "vnr_list": vnr_datas}

                    act = self.buffer.actions[i].to(self.device)
                    _, lp, ent, val = self.ac.get_action_and_value(obs_dev, action=act)
                    mb_log_probs.append(lp)
                    mb_values.append(val.squeeze())
                    mb_entropy.append(ent)

                mb_log_probs = torch.stack(mb_log_probs)
                mb_values    = torch.stack(mb_values)
                mb_entropy   = torch.stack(mb_entropy)

                mb_old_lp    = old_log_probs[mb_idx]
                mb_adv       = adv[mb_idx]
                mb_ret       = returns[mb_idx]

                ratio = (mb_log_probs - mb_old_lp).exp()

                # Surrogate clip loss
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss  = nn.functional.mse_loss(mb_values, mb_ret)

                # Entropy bonus
                entropy_loss = -mb_entropy.mean()

                loss = (
                    policy_loss
                    + cfg.vf_coef  * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.grad_clip)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += (-entropy_loss.item())

        n_updates = cfg.n_epochs * max(1, T // cfg.batch_size)
        return dict(
            policy_loss = total_policy_loss / n_updates,
            value_loss  = total_value_loss  / n_updates,
            entropy     = total_entropy     / n_updates,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        print(f"[PPO] Starting training on {self.device}")
        print(f"  total_steps={cfg.total_timesteps}, n_steps={cfg.n_steps}, "
              f"batch={cfg.batch_size}, epochs={cfg.n_epochs}")
        print(f"  reward={cfg.reward_mode}, substrate={cfg.substrate_nodes}, "
              f"vnr_batch={cfg.batch_size_env}")
        print()

        t0           = time.time()
        self.global_step  = 0
        update_count = 0

        while self.global_step < cfg.total_timesteps:
            self._collect_rollout(cfg.n_steps)
            self.global_step += len(self.buffer)

            advantages, returns = self._compute_gae(self._obs)
            update_stats        = self._update(advantages, returns)
            update_count       += 1

            entry = dict(
                step         = self.global_step,
                update       = update_count,
                **update_stats,
            )
            self.history.append(entry)
            
            self.writer.add_scalar("Train/PolicyLoss", update_stats["policy_loss"], self.global_step)
            self.writer.add_scalar("Train/ValueLoss", update_stats["value_loss"], self.global_step)
            self.writer.add_scalar("Train/Entropy", update_stats["entropy"], self.global_step)

            if self.global_step % cfg.log_every < cfg.n_steps:
                elapsed = time.time() - t0
                print(
                    f"Step {self.global_step:8d}/{cfg.total_timesteps} | "
                    f"π_loss={update_stats['policy_loss']:+.4f} | "
                    f"v_loss={update_stats['value_loss']:.4f} | "
                    f"entropy={update_stats['entropy']:.3f} | "
                    f"t={elapsed:.0f}s"
                )

            if self.global_step % cfg.save_every < cfg.n_steps:
                ckpt = os.path.join(
                    cfg.save_dir, f"{cfg.run_name}_step{self.global_step}.pt"
                )
                self.save(ckpt)

        final = os.path.join(cfg.save_dir, f"{cfg.run_name}_final.pt")
        self.save(final)
        self.writer.close()
        print(f"\n[PPO] Training complete. Checkpoint: {final}")

    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.cfg.save_dir, f"{self.cfg.run_name}_final.pt")
        self.ac.scheduler.save(path, extra_meta={"config": self.cfg.__dict__})

    def evaluate(self, n_episodes: int = 50, verbose: bool = False) -> dict:
        """Greedy evaluation (argmax action)."""
        self.ac.eval()
        ar_list, rc_list = [], []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done   = False
            while not done:
                if not obs["vnr_list"]:
                    break
                sub  = obs["substrate"].to(self.device)
                vnrs = [v.to(self.device) for v in obs["vnr_list"]]
                with torch.no_grad():
                    scores = self.ac.scheduler(sub, vnrs)
                    action = scores.argmax().item()
                obs, _, done, _, _ = self.env.step(action)
            info = self.env.episode_summary()
            ar_list.append(info["acc_rate"])
            rc_list.append(info["rc_ratio"])

        result = dict(
            mean_ar = sum(ar_list) / len(ar_list),
            mean_rc = sum(rc_list) / len(rc_list),
        )
        if verbose:
            print(f"[Eval] mean AR={result['mean_ar']:.2%}  R/C={result['mean_rc']:.3f}")
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2 PPO training for VNRScheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--total-steps",  type=int,   default=500_000)
    p.add_argument("--n-steps",      type=int,   default=512)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--n-epochs",     type=int,   default=10)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--gae-lambda",   type=float, default=0.95)
    p.add_argument("--clip",         type=float, default=0.2)
    p.add_argument("--ent-coef",     type=float, default=0.01)
    p.add_argument("--reward",       type=str,   default="revenue",
                   choices=["simple", "revenue", "longterm"])
    p.add_argument("--no-ctx",       action="store_true",
                   help="Disable BatchContextEncoder")
    p.add_argument("--sub-nodes",    type=int,   default=50)
    p.add_argument("--vnr-batch",    type=int,   default=10)
    # VNR size range flags (replace old single --vnr-nodes)
    p.add_argument("--vnr-min-nodes", type=int,  default=2,
                   help="Minimum virtual nodes per VNR (inclusive)")
    p.add_argument("--vnr-max-nodes", type=int,  default=8,
                   help="Maximum virtual nodes per VNR (inclusive)")
    p.add_argument("--hpso-iter",    type=int,   default=30)
    p.add_argument("--log-every",    type=int,   default=10_000)
    p.add_argument("--save-every",   type=int,   default=50_000)
    p.add_argument("--save-dir",     type=str,   default="checkpoints")
    p.add_argument("--run-name",     type=str,   default="ppo_phase2")
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--load-checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume training from")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg  = PPOConfig(
        total_timesteps   = args.total_steps,
        n_steps           = args.n_steps,
        batch_size        = args.batch_size,
        n_epochs          = args.n_epochs,
        lr                = args.lr,
        gamma             = args.gamma,
        gae_lambda        = args.gae_lambda,
        clip_range        = args.clip,
        ent_coef          = args.ent_coef,
        reward_mode       = args.reward,
        use_batch_context = not args.no_ctx,
        substrate_nodes   = args.sub_nodes,
        batch_size_env    = args.vnr_batch,
        vnr_min_nodes     = args.vnr_min_nodes,
        vnr_max_nodes     = args.vnr_max_nodes,
        hpso_iterations   = args.hpso_iter,
        log_every         = args.log_every,
        save_every        = args.save_every,
        save_dir          = args.save_dir,
        run_name          = args.run_name,
        device            = args.device,
        load_checkpoint   = args.load_checkpoint,
    )
    trainer = PPOTrainerScheduler(cfg)
    trainer.train()
    trainer.evaluate(n_episodes=50, verbose=True)
