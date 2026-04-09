"""
src/scripts/train_rl_scheduler.py
==================================
Training script for the RL-based VNR Scheduler.

Trains a VNRSchedulerAgent (Graph Pointer Network + PPO) to learn an
optimal VNR processing order for the HPSO batch solver.

Usage
-----
# Quick test with synthetic data:
    python -m src.scripts.train_rl_scheduler

# Full training run:
    python -m src.scripts.train_rl_scheduler             \\
        --episodes 500                                   \\
        --window_size 10                                 \\
        --K_max 20                                       \\
        --substrate_nodes 30                             \\
        --save_path result/rl_scheduler.pt               \\
        --seed 42

# Training with generated dataset (recommended for real experiments):
    python -m src.scripts.train_rl_scheduler             \\
        --data_path dataset/train_windows.pkl            \\
        --episodes 1000                                  \\
        --save_path result/rl_scheduler_trained.pt

# Resume training:
    python -m src.scripts.train_rl_scheduler             \\
        --resume result/rl_scheduler.pt                  \\
        --episodes 200                                   \\
        --save_path result/rl_scheduler_finetuned.pt

Architecture overview (see idea.md / src/rl/networks.py):
  - Substrate Encoder : 3-layer GAT → mean-pool → h_p  ∈ R^128
  - VNR Encoder       : 3-layer GAT → max-pool  → h_vi ∈ R^64  (shared weights)
  - Context MLP       : [h_p || mean(h_vi)] → initial GRU state ∈ R^256
  - Pointer Decoder   : GRU + attention → logits over pending VNR queue
  - Critic Head       : [h_p || mean(h_vi)] → scalar V(s)
  - Training          : PPO (clip_eps=0.2, GAE λ=0.95, entropy coef=0.01)
"""

import argparse
import copy
import os
import pickle
import random
import time

import numpy as np
import torch

from src.rl import VNRSchedulerAgent, PPOTrainer, DEFAULT_CFG
from src.rl.utils import build_vnr_dgl, build_substrate_dgl
from src.generators.substrate_generator import generate_substrate
from src.generators.vnr_generator import generate_single_vnr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL-based VNR Scheduler (Graph Pointer Network + PPO)"
    )

    # Data
    p.add_argument('--data_path',      type=str,   default=None,
                   help='Path to pre-generated dataset pickle '
                        '(substrate, vnr_pool).  If not provided, synthetic '
                        'data is generated on the fly.')
    p.add_argument('--substrate_nodes',type=int,   default=30,
                   help='Synthetic substrate size (used when --data_path not set).')
    p.add_argument('--vnr_pool_size',  type=int,   default=200,
                   help='Synthetic VNR pool size.')
    p.add_argument('--vnr_min_nodes',  type=int,   default=3)
    p.add_argument('--vnr_max_nodes',  type=int,   default=8)

    # Training
    p.add_argument('--episodes',   type=int,   default=300,
                   help='Number of training episodes.')
    p.add_argument('--window_size',type=int,   default=10,
                   help='Number of VNRs per time window.')
    p.add_argument('--K_max',      type=int,   default=20,
                   help='Maximum VNRs per window (padding target).')
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--gamma',      type=float, default=0.98)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--clip_eps',   type=float, default=0.2)
    p.add_argument('--entropy_coef',type=float,default=0.01)
    p.add_argument('--value_coef', type=float, default=0.5)
    p.add_argument('--k_epochs',   type=int,   default=4)

    # Eval
    p.add_argument('--eval_every', type=int,   default=50,
                   help='Run evaluation every N episodes.')
    p.add_argument('--eval_episodes',type=int, default=5,
                   help='Number of episodes for each evaluation run.')

    # Checkpointing
    p.add_argument('--save_path',  type=str,   default='result/rl_scheduler.pt',
                   help='Where to save the trained model.')
    p.add_argument('--resume',     type=str,   default=None,
                   help='Path to checkpoint to resume from.')
    p.add_argument('--save_every', type=int,   default=100,
                   help='Save checkpoint every N episodes.')

    # Reproducibility
    p.add_argument('--seed',       type=int,   default=42)

    # HPSO parameters
    p.add_argument('--hpso_particles',  type=int,   default=20)
    p.add_argument('--hpso_iterations', type=int,   default=30)

    # Network
    p.add_argument('--substrate_hidden',type=int,   default=128)
    p.add_argument('--vnr_hidden',      type=int,   default=64)
    p.add_argument('--gru_hidden',      type=int,   default=256)
    p.add_argument('--critic_hidden',   type=int,   default=128)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str):
    """
    Load pre-generated dataset.
    Expected pickle format: (substrate_nx, [vnr_nx, ...])
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, (list, tuple)) and len(data) == 2:
        substrate, vnr_pool = data
        print(f"[Data] Loaded {len(vnr_pool)} VNRs from {path}")
        return substrate, list(vnr_pool)
    raise ValueError(
        f"Unexpected pickle format at {path}. "
        "Expected (substrate_nx, [vnr_nx, ...])."
    )


def make_synthetic_data(args, rng: random.Random):
    """Generate synthetic substrate and VNR pool for quick testing."""
    print("[Data] Generating synthetic substrate and VNR pool ...")
    substrate = generate_substrate(
        num_domains=3,
        num_nodes_total=args.substrate_nodes,
        p_intra=0.5,
        p_inter=0.1,
        cpu_range=(50, 150),
        bw_range=(100, 300),
        seed=args.seed,
    )
    # Ensure residual resources match capacity (fresh substrate)
    for n in substrate.nodes():
        nd = substrate.nodes[n]
        nd['cpu_res'] = nd.get('cpu', 100)
        nd.setdefault('mem', 50)
        nd['mem_res'] = nd['mem']

    for u, v in substrate.edges():
        ed = substrate[u][v]
        ed['bw_res'] = ed.get('bw', 100)

    vnr_pool = []
    for i in range(args.vnr_pool_size):
        num_nodes = rng.randint(args.vnr_min_nodes, args.vnr_max_nodes)
        vnr = generate_single_vnr(
            num_nodes=num_nodes,
            edge_prob=0.5,
            cpu_range=(5, 20),
            bw_range=(5, 20),
        )
        # Add mem and vnf_type attributes
        for n in vnr.nodes():
            vnr.nodes[n].setdefault('mem',      rng.uniform(2, 10))
            vnr.nodes[n].setdefault('vnf_type', rng.randint(0, 4))
        vnr_pool.append(vnr)

    print(f"[Data] Substrate: {len(substrate.nodes())} nodes, "
          f"{len(substrate.edges())} edges")
    print(f"[Data] VNR pool : {len(vnr_pool)} VNRs")
    return substrate, vnr_pool


def get_window(vnr_pool: list, window_size: int, rng: random.Random) -> list:
    """Sample a random window of VNRs from the pool."""
    return rng.sample(vnr_pool, min(window_size, len(vnr_pool)))


def make_hpso_fn(args):
    """Build the HPSO embed callable from CLI arguments."""
    try:
        from src.algorithms.fast_hpso import hpso_embed as _hpso

        def hpso_fn(sub, vnr):
            return _hpso(
                substrate_graph=sub,
                vnr_graph=vnr,
                particles=args.hpso_particles,
                iterations=args.hpso_iterations,
            )
        print("[HPSO] Using fast_hpso.hpso_embed")
        return hpso_fn

    except (ImportError, Exception) as e:
        print(f"[HPSO] fast_hpso not available ({e}). "
              f"Using dummy accept-all embed for development.")

        def dummy_fn(sub, vnr):
            mapping = {n: n % len(sub.nodes()) for n in vnr.nodes()}
            return (mapping, {})

        return dummy_fn


# ---------------------------------------------------------------------------
# Evaluation run
# ---------------------------------------------------------------------------

def run_eval(
    trainer: PPOTrainer,
    substrate,
    vnr_pool: list,
    args,
    rng: random.Random,
    hpso_fn,
    n_eval: int = 5,
) -> dict:
    """Run deterministic evaluation episodes; return averaged metrics."""
    trainer.agent.eval()
    metrics_list = []

    for _ in range(n_eval):
        window   = get_window(vnr_pool, args.window_size, rng)
        sub_copy = copy.deepcopy(substrate)

        _, _, m = trainer.collect_and_update(
            vnr_list      = window,
            substrate     = sub_copy,
            hpso_embed_fn = hpso_fn,
        )
        metrics_list.append(m)

    avg = {
        k: float(np.mean([m[k] for m in metrics_list]))
        for k in metrics_list[0]
    }
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    rng  = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # --- Config ---
    cfg = {
        **DEFAULT_CFG,
        'K_max':          args.K_max,
        'lr':             args.lr,
        'gamma':          args.gamma,
        'gae_lambda':     args.gae_lambda,
        'clip_eps':       args.clip_eps,
        'entropy_coef':   args.entropy_coef,
        'value_coef':     args.value_coef,
        'k_epochs':       args.k_epochs,
        'substrate_hidden': args.substrate_hidden,
        'vnr_hidden':     args.vnr_hidden,
        'gru_hidden':     args.gru_hidden,
        'critic_hidden':  args.critic_hidden,
    }

    # --- Data ---
    if args.data_path:
        substrate, vnr_pool = load_dataset(args.data_path)
    else:
        print("[INFO] No --data_path given. Using synthetic data.")
        substrate, vnr_pool = make_synthetic_data(args, rng)

    # --- HPSO function ---
    hpso_fn = make_hpso_fn(args)

    # --- Agent & Trainer ---
    agent   = VNRSchedulerAgent(cfg)
    trainer = PPOTrainer(agent, cfg)

    history: list = []

    # Resume from checkpoint if requested
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        history = trainer.load(args.resume)

    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"[INFO] Agent parameters: {total_params:,}")
    print(f"[INFO] Device: {next(agent.parameters()).device}")
    print(f"[INFO] Starting training for {args.episodes} episodes "
          f"(window={args.window_size}, K_max={args.K_max})")
    print("-" * 70)

    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        window   = get_window(vnr_pool, args.window_size, rng)
        sub_copy = copy.deepcopy(substrate)

        accepted, rejected, metrics = trainer.collect_and_update(
            vnr_list      = window,
            substrate     = sub_copy,
            hpso_embed_fn = hpso_fn,
        )
        history.append(metrics)

        # --- Periodic logging ---
        if ep % 10 == 0:
            recent = history[-10:]
            avg_acc    = np.mean([h['acc_rate']    for h in recent])
            avg_rc     = np.mean([h['rc_ratio']    for h in recent])
            avg_loss   = np.mean([h['total_loss']  for h in recent])
            avg_rew    = np.mean([h['reward_mean'] for h in recent])
            avg_rstd   = np.mean([h['reward_std']  for h in recent])
            elapsed    = time.time() - t0
            print(f"Ep {ep:4d}/{args.episodes} | "
                  f"Acc={avg_acc:.3f} | RC={avg_rc:.3f} | "
                  f"R={avg_rew:.3f}±{avg_rstd:.3f} | "
                  f"Loss={avg_loss:.4f} | {elapsed:.0f}s")

        # --- Periodic evaluation ---
        if ep % args.eval_every == 0:
            eval_m = run_eval(
                trainer, substrate, vnr_pool, args, rng, hpso_fn,
                n_eval=args.eval_episodes,
            )
            print(f"  [EVAL] ep={ep} | "
                  f"AccRate={eval_m['acc_rate']:.3f} | "
                  f"RC={eval_m['rc_ratio']:.3f}")
            trainer.agent.train()

        # --- Periodic checkpoint ---
        if ep % args.save_every == 0:
            ckpt_path = args.save_path.replace('.pt', f'_ep{ep}.pt')
            trainer.save(ckpt_path, history)

    # --- Final save ---
    trainer.save(args.save_path, history)
    elapsed = time.time() - t0
    print("-" * 70)
    print(f"[INFO] Training complete in {elapsed:.0f}s")
    print(f"[INFO] Final model saved to {args.save_path}")

    # Print final stats
    last_n = min(50, len(history))
    if last_n > 0:
        print(f"[INFO] Last {last_n} episodes avg: "
              f"AccRate={np.mean([h['acc_rate']  for h in history[-last_n:]]):.3f} | "
              f"RC={np.mean([h['rc_ratio']  for h in history[-last_n:]]):.3f}")


if __name__ == '__main__':
    main()
