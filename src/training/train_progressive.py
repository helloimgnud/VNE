"""
src/training/train_progressive.py
=================================
Phase B — Progressive Deployment Curriculum (Level-based PPO).

This trainer builds on `train_ppo.py`, substituting the static environment
with the ProgressiveDeploymentWrapper. This ensures the substrate naturally
depletes, and VNR batch difficulty scales over time using the curriculum.
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
from src.training.progressive_env import ProgressiveDeploymentWrapper, ProgressiveConfig


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProgressivePPOConfig:
    """All hyper-parameters for the PPO training run."""

    # Training budget
    total_timesteps: int   = 1_000_000
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
    reward_mode:     str   = "simple"  # simple -> revenue -> longterm automatically

    # Network
    use_batch_context: bool = True

    # Environment default base bounds
    substrate_nodes:  int   = 50
    batch_size_env:   int   = 10
    # VNR size range: each VNR in a batch independently samples its node count
    # uniformly from [vnr_min_nodes, vnr_max_nodes], giving a varied dataset.
    vnr_min_nodes:    int   = 2
    vnr_max_nodes:    int   = 8
    hpso_particles:   int   = 20
    hpso_iterations:  int   = 30

    # Logging / checkpointing
    log_every:  int = 100      # log every N timesteps
    save_every: int = 200
    save_dir:   str = "checkpoints"
    run_name:   str = "ppo_progressive"
    device:     str = "auto"
    resume_path: Optional[str] = None


class _RolloutBuffer:
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


class ProgressiveTrainer:
    def __init__(self, cfg: ProgressivePPOConfig):
        self.cfg = cfg

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # Networks
        scheduler = VNRScheduler(use_batch_context=cfg.use_batch_context)
        
        if cfg.resume_path and os.path.exists(cfg.resume_path):
            print(f"[PROGRESSIVE] Resuming from checkpoint: {cfg.resume_path}")
            # Use VNRScheduler.load logic to get state_dict
            ckpt = torch.load(cfg.resume_path, map_location="cpu")
            scheduler.load_state_dict(ckpt["state_dict"])
        
        self.ac   = GNNActorCritic(scheduler).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

        # Environment
        substrate_fn, batch_fn = make_env_fns(
            substrate_nodes = cfg.substrate_nodes,
            batch_size      = cfg.batch_size_env,
            vnr_min_nodes   = cfg.vnr_min_nodes,
            vnr_max_nodes   = cfg.vnr_max_nodes,
        )
        base_env = VNEOrderingEnv(
            substrate_fn = substrate_fn,
            batch_fn     = batch_fn,
            hpso_params  = dict(
                particles   = cfg.hpso_particles,
                iterations  = cfg.hpso_iterations,
            ),
            reward_mode  = cfg.reward_mode,
        )
        
        # WE WRAP WITH PROGRESSIVE CURRICULUM
        prog_cfg = ProgressiveConfig()
        self.env = ProgressiveDeploymentWrapper(base_env, prog_cfg)

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.buffer  = _RolloutBuffer()
        self.history: List[dict] = []
        
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join("runs", cfg.run_name)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Current episode state
        self._obs, _ = self.env.reset()

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
                ep_info = self.env.inner_env.episode_summary()
                global_step = getattr(self, "global_step", 0) + len(self.buffer)
                
                # Check curriculum triggers!
                events = self.env.maybe_promote(ep_info)
                
                self.writer.add_scalar("Metrics/AcceptanceRate", ep_info["acc_rate"], global_step)
                self.writer.add_scalar("Metrics/RevenueCostRatio", ep_info["rc_ratio"], global_step)
                self.writer.add_scalar("Metrics/SubstrateCpuUtil", ep_info.get("cpu_util", 0.0), global_step)
                self.writer.add_scalar("Curriculum/Level", self.env.level, global_step)
                self.writer.add_scalar("Curriculum/RewardMode", list(RewardMode).index(self.env.inner_env.reward_mode), global_step)
                self.writer.add_scalar("Curriculum/BatchSize", self.env.current_batch_size, global_step)
                self.writer.add_scalar("Curriculum/CommittedVNRs", len(self.env.committed_vnrs), global_step)
                
                if events["levelled_up"]:
                    print(f"\n[CURRICULUM] LEVEL UP! Now Level {self.env.level}")
                    print(f"             Batch Size: {self.env.current_batch_size}, Demand Scale: {self.env.demand_scale:.2f}")
                    print(f"             Reward Mode: {self.env.inner_env.reward_mode}\n")

            if done:
                self._obs, _ = self.env.reset()
            else:
                self._obs = next_obs

    def _compute_gae(self, last_obs) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg      = self.cfg
        T        = len(self.buffer.rewards)
        rewards  = self.buffer.rewards
        dones    = self.buffer.dones
        values   = [v.item() for v in self.buffer.values]

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

    def _update(self, advantages: torch.Tensor, returns: torch.Tensor):
        cfg = self.cfg
        T   = len(self.buffer.obs)

        old_log_probs = torch.stack(self.buffer.log_probs).detach().to(self.device)

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        self.ac.train()
        for _ in range(cfg.n_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, cfg.batch_size):
                mb_idx = indices[start : start + cfg.batch_size]

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

                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = nn.functional.mse_loss(mb_values, mb_ret)

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

    def train(self):
        cfg = self.cfg
        print(f"[PPO Progressive] Starting training on {self.device}")
        print(f"  total_steps={cfg.total_timesteps}, n_steps={cfg.n_steps}")
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

            self.writer.add_scalar("Train/PolicyLoss", update_stats["policy_loss"], self.global_step)
            self.writer.add_scalar("Train/ValueLoss", update_stats["value_loss"], self.global_step)
            self.writer.add_scalar("Train/Entropy", update_stats["entropy"], self.global_step)

            if self.global_step % cfg.log_every < cfg.n_steps:
                elapsed = time.time() - t0
                print(
                    f"Step {self.global_step:8d}/{cfg.total_timesteps} | "
                    f"Lvl: {self.env.level} | "
                    f"π_loss={update_stats['policy_loss']:+.4f} | "
                    f"v_loss={update_stats['value_loss']:.4f} | "
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
        print(f"\n[PPO Progressive] Training complete. Checkpoint: {final}")

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.cfg.save_dir, f"{self.cfg.run_name}_final.pt")
        self.ac.scheduler.save(path, extra_meta={"config": self.cfg.__dict__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # VNR size range
    parser.add_argument("--vnr-min-nodes", type=int, default=2,
                        help="Minimum virtual nodes per VNR (inclusive)")
    parser.add_argument("--vnr-max-nodes", type=int, default=8,
                        help="Maximum virtual nodes per VNR (inclusive)")
    args = parser.parse_args()

    cfg = ProgressivePPOConfig(
        total_timesteps = args.total_steps,
        device          = args.device,
        resume_path     = args.resume,
        vnr_min_nodes   = args.vnr_min_nodes,
        vnr_max_nodes   = args.vnr_max_nodes,
    )
    trainer = ProgressiveTrainer(cfg)
    trainer.train()
