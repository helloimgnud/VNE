"""
src/training/train_reinforce.py
================================
Phase 1 — REINFORCE training loop for VNRScheduler.

Algorithm
---------
  For each episode:
    1. Reset env → get obs (substrate_pyg, vnr_list_pyg)
    2. Loop until done:
       a. Forward pass: scores = model(substrate, vnr_list)
       b. Sample action from Categorical(logits=scores)
       c. Step env → reward
    3. Compute discounted returns
    4. Subtract baseline (mean return) → advantages
    5. Loss = -(log_probs · advantages).mean()
    6. Backward + gradient clip + step

This is the simplest possible training loop — no value function, no
importance sampling. Use it to validate that the GNN can learn at all.

Once AR improvement is confirmed, graduate to ``train_ppo.py``.

Usage (plugin mode)
-------------------
>>> from src.training.train_reinforce import ReinforceTrainer, ReinforceConfig
>>> cfg = ReinforceConfig(num_episodes=2000, reward_mode="simple")
>>> trainer = ReinforceTrainer(cfg)
>>> trainer.train()
>>> trainer.save("checkpoints/reinforce_phase1.pt")

Or from the command line:
  python -m src.training.train_reinforce --episodes 2000 --reward simple
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

from src.scheduler.model import VNRScheduler
from src.scheduler.environment import VNEOrderingEnv
from src.scheduler.rewards import RewardMode
from src.training.generate_data import make_env_fns


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReinforceConfig:
    """All hyper-parameters for the REINFORCE training run."""

    # Training
    num_episodes:   int   = 2000
    lr:             float = 3e-4
    gamma:          float = 0.99
    grad_clip:      float = 0.5
    reward_mode:    str   = "simple"     # simple | revenue | longterm

    # Network
    use_batch_context: bool = False      # Phase 1: no context encoder

    # Environment (substrate / VNR generation)
    substrate_nodes:  int   = 50
    batch_size:       int   = 10
    vnr_nodes:        int   = 4
    fixed_substrate:  bool  = False

    # HPSO (passed to env)
    hpso_particles:   int   = 20
    hpso_iterations:  int   = 30

    # Logging / checkpointing
    log_every:    int   = 100
    save_every:   int   = 500
    save_dir:     str   = "checkpoints"
    run_name:     str   = "reinforce_phase1"
    device:       str   = "auto"


# ---------------------------------------------------------------------------
# Helper: compute discounted returns
# ---------------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted future returns for a list of rewards.

    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
    """
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class ReinforceTrainer:
    """
    REINFORCE trainer with baseline (mean-return subtraction).

    Parameters
    ----------
    cfg : ReinforceConfig
    """

    def __init__(self, cfg: ReinforceConfig):
        self.cfg = cfg

        # Device
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # Model
        self.model = VNRScheduler(
            use_batch_context=cfg.use_batch_context,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # Environment factories
        substrate_fn, batch_fn = make_env_fns(
            substrate_nodes = cfg.substrate_nodes,
            batch_size      = cfg.batch_size,
            vnr_nodes       = cfg.vnr_nodes,
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

        # Logging
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.history: List[dict] = []

    # ------------------------------------------------------------------

    def _run_episode(self):
        """
        Run one episode and collect (log_probs, rewards).

        Returns
        -------
        log_probs : list of Tensor (each scalar)
        rewards   : list of float
        info      : dict  (episode summary from env)
        """
        obs, _ = self.env.reset()
        log_probs: List[torch.Tensor] = []
        rewards:   List[float]        = []

        done = False
        while not done:
            if not obs["vnr_list"]:
                break

            # Move PyG data to device
            sub_data  = obs["substrate"].to(self.device)
            vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]

            # Forward pass
            scores = self.model(sub_data, vnr_datas)     # [B]
            dist   = Categorical(logits=scores)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))

            obs, reward, done, _, info = self.env.step(action.item())
            rewards.append(reward)

        episode_info = self.env.episode_summary()
        return log_probs, rewards, episode_info

    # ------------------------------------------------------------------

    def train(self):
        """Run the full REINFORCE training loop."""
        cfg = self.cfg
        print(f"[REINFORCE] Starting training on {self.device}")
        print(f"  episodes={cfg.num_episodes}, lr={cfg.lr}, reward={cfg.reward_mode}")
        print(f"  substrate={cfg.substrate_nodes} nodes, batch={cfg.batch_size} VNRs")
        print()

        t0 = time.time()

        for ep in range(1, cfg.num_episodes + 1):
            self.model.train()
            log_probs, rewards, ep_info = self._run_episode()

            if not log_probs:
                continue  # empty episode (no VNRs)

            # --- Compute advantages ---
            returns   = compute_returns(rewards, cfg.gamma).to(self.device)
            baseline  = returns.mean()
            advantages = returns - baseline
            # Normalise for stable gradients
            if advantages.std() > 1e-6:
                advantages = advantages / (advantages.std() + 1e-8)

            # --- Policy gradient loss ---
            loss = -(torch.stack(log_probs) * advantages).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            # --- Logging ---
            log_entry = dict(
                episode    = ep,
                loss       = loss.item(),
                total_rew  = sum(rewards),
                acc_rate   = ep_info["acc_rate"],
                rc_ratio   = ep_info["rc_ratio"],
            )
            self.history.append(log_entry)

            if ep % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"Ep {ep:5d}/{cfg.num_episodes} | "
                    f"loss={loss.item():+.4f} | "
                    f"reward={sum(rewards):.3f} | "
                    f"AR={ep_info['acc_rate']:.2%} | "
                    f"R/C={ep_info['rc_ratio']:.3f} | "
                    f"t={elapsed:.0f}s"
                )

            if ep % cfg.save_every == 0:
                ckpt_path = os.path.join(
                    cfg.save_dir, f"{cfg.run_name}_ep{ep}.pt"
                )
                self.save(ckpt_path)

        # Final save
        final_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_final.pt")
        self.save(final_path)
        print(f"\n[REINFORCE] Training complete. Checkpoint: {final_path}")

    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.cfg.save_dir, f"{self.cfg.run_name}_final.pt"
            )
        self.model.save(path, extra_meta={"config": self.cfg.__dict__})

    # ------------------------------------------------------------------

    def evaluate(
        self,
        n_episodes: int = 50,
        verbose:    bool = False,
    ) -> dict:
        """
        Run n_episodes with the current model (no gradient) and report
        mean AR, R/C, and total reward.
        """
        self.model.eval()
        ar_list, rc_list, rew_list = [], [], []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done   = False
            ep_rew = 0.0

            while not done:
                if not obs["vnr_list"]:
                    break
                sub_data  = obs["substrate"].to(self.device)
                vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]

                with torch.no_grad():
                    scores = self.model(sub_data, vnr_datas)
                    action = scores.argmax().item()

                obs, reward, done, _, _ = self.env.step(action)
                ep_rew += reward

            info = self.env.episode_summary()
            ar_list.append(info["acc_rate"])
            rc_list.append(info["rc_ratio"])
            rew_list.append(ep_rew)

        result = dict(
            mean_ar       = sum(ar_list) / len(ar_list),
            mean_rc       = sum(rc_list) / len(rc_list),
            mean_reward   = sum(rew_list) / len(rew_list),
        )
        if verbose:
            print(f"[Eval] mean AR={result['mean_ar']:.2%}  "
                  f"R/C={result['mean_rc']:.3f}  "
                  f"reward={result['mean_reward']:.3f}")
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1 REINFORCE training for VNRScheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",     type=int,   default=2000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--grad-clip",    type=float, default=0.5)
    p.add_argument("--reward",       type=str,   default="simple",
                   choices=["simple", "revenue", "longterm"])
    p.add_argument("--batch-size",   type=int,   default=10)
    p.add_argument("--sub-nodes",    type=int,   default=50)
    p.add_argument("--vnr-nodes",    type=int,   default=4)
    p.add_argument("--fixed-sub",    action="store_true",
                   help="Fix substrate across all episodes (easier curriculum)")
    p.add_argument("--use-ctx",      action="store_true",
                   help="Enable BatchContextEncoder (Phase 2+ feature)")
    p.add_argument("--hpso-iter",    type=int,   default=30)
    p.add_argument("--log-every",    type=int,   default=100)
    p.add_argument("--save-every",   type=int,   default=500)
    p.add_argument("--save-dir",     type=str,   default="checkpoints")
    p.add_argument("--run-name",     type=str,   default="reinforce_phase1")
    p.add_argument("--device",       type=str,   default="auto")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg  = ReinforceConfig(
        num_episodes      = args.episodes,
        lr                = args.lr,
        gamma             = args.gamma,
        grad_clip         = args.grad_clip,
        reward_mode       = args.reward,
        use_batch_context = args.use_ctx,
        batch_size        = args.batch_size,
        substrate_nodes   = args.sub_nodes,
        vnr_nodes         = args.vnr_nodes,
        fixed_substrate   = args.fixed_sub,
        hpso_iterations   = args.hpso_iter,
        log_every         = args.log_every,
        save_every        = args.save_every,
        save_dir          = args.save_dir,
        run_name          = args.run_name,
        device            = args.device,
    )
    trainer = ReinforceTrainer(cfg)
    trainer.train()
    trainer.evaluate(n_episodes=50, verbose=True)
