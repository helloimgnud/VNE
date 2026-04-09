"""
src/scripts/train_rl_scheduler.py
==================================
Training script for the RL-based VNR Scheduler.

Trains a VNRSchedulerAgent (Graph Pointer Network + PPO) to learn an
optimal VNR processing order for the HPSO batch solver.

Two training modes
------------------
1. Standard mode (default)
   Each episode uses a fresh deepcopy of the substrate. Useful for debugging
   and quick sanity checks, but provides no learning signal when the substrate
   is easy (AccRate = 1.0 always).

2. PSD mode (--psd flag)
   Progressive Substrate Depletion curriculum.
   When the agent masters the current load level (AccRate >= ar_thresh AND
   RC >= rc_thresh for `patience` consecutive episodes), the BEST-RC episode's
   embeddings are permanently committed to the real substrate.
   The substrate fills monotonically → difficulty increases → real RL signal.

Usage examples
--------------
# Quick test (standard mode):
    python -m src.scripts.train_rl_scheduler --episodes 50

# Full standard training:
    python -m src.scripts.train_rl_scheduler \\
        --episodes 500 --window_size 10 --K_max 20 \\
        --substrate_nodes 30 --save_path result/rl_scheduler.pt

# PSD curriculum (recommended):
    python -m src.scripts.train_rl_scheduler \\
        --psd --patience 10 --ar_thresh 0.95 --rc_thresh 0.90 \\
        --substrate_nodes 30 --episodes 2000 \\
        --save_path result/rl_psd.pt

# Load pretrained and resume PSD:
    python -m src.scripts.train_rl_scheduler \\
        --psd --resume result/rl_psd_ep500.pt \\
        --episodes 500 --save_path result/rl_psd_finetuned.pt

# Train on real dataset:
    python -m src.scripts.train_rl_scheduler \\
        --data_path dataset/train_windows.pkl \\
        --psd --episodes 1000 --save_path result/rl_real.pt
"""

import argparse
import copy
import os
import pickle
import random
import time

import numpy as np
import torch

from src.rl import VNRSchedulerAgent, PPOTrainer, CurriculumManager, DEFAULT_CFG
from src.rl.utils import build_vnr_dgl, build_substrate_dgl
from src.generators.substrate_generator import generate_substrate
from src.generators.vnr_generator import generate_single_vnr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL-based VNR Scheduler (Graph Pointer Network + PPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    g_data = p.add_argument_group("Data")
    g_data.add_argument(
        '--data_path', type=str, default=None,
        help='Path to pre-generated dataset pickle (substrate_nx, [vnr_nx, ...]).'
             ' If omitted, synthetic data is generated on the fly.')
    g_data.add_argument('--substrate_nodes', type=int, default=30)
    g_data.add_argument('--vnr_pool_size',   type=int, default=200)
    g_data.add_argument('--vnr_min_nodes',   type=int, default=3)
    g_data.add_argument('--vnr_max_nodes',   type=int, default=8)

    # ── Training ─────────────────────────────────────────────────────────────
    g_train = p.add_argument_group("Training")
    g_train.add_argument('--episodes',    type=int,   default=300)
    g_train.add_argument('--window_size', type=int,   default=10,
                         help='VNRs per time window (= batch size for one episode).')
    g_train.add_argument('--K_max',       type=int,   default=20,
                         help='Max queue length (window_size <= K_max).')
    g_train.add_argument('--lr',          type=float, default=3e-4)
    g_train.add_argument('--gamma',       type=float, default=0.98)
    g_train.add_argument('--gae_lambda',  type=float, default=0.95)
    g_train.add_argument('--clip_eps',    type=float, default=0.2)
    g_train.add_argument('--entropy_coef',type=float, default=0.01)
    g_train.add_argument('--value_coef',  type=float, default=0.5)
    g_train.add_argument('--k_epochs',    type=int,   default=4)

    # ── PSD curriculum ────────────────────────────────────────────────────────
    g_psd = p.add_argument_group("PSD Curriculum (--psd to enable)")
    g_psd.add_argument('--psd',       action='store_true', default=False,
                       help='Enable Progressive Substrate Depletion curriculum.')
    g_psd.add_argument('--patience',  type=int,   default=10,
                       help='Consecutive mastered episodes before committing.')
    g_psd.add_argument('--ar_thresh', type=float, default=0.95,
                       help='AccRate threshold for "mastered".')
    g_psd.add_argument('--rc_thresh', type=float, default=0.90,
                       help='RC ratio threshold for "mastered".')
    g_psd.add_argument('--max_load',  type=float, default=0.85,
                       help='Stop PSD when substrate CPU is this fraction full.')

    # ── Eval ─────────────────────────────────────────────────────────────────
    g_eval = p.add_argument_group("Evaluation")
    g_eval.add_argument('--eval_every',   type=int, default=50)
    g_eval.add_argument('--eval_episodes',type=int, default=5)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    g_ckpt = p.add_argument_group("Checkpointing")
    g_ckpt.add_argument('--save_path',  type=str, default='result/rl_scheduler.pt')
    g_ckpt.add_argument('--resume',     type=str, default=None,
                        help='Path to checkpoint .pt file to resume from.')
    g_ckpt.add_argument('--save_every', type=int, default=100)

    # ── Network architecture ──────────────────────────────────────────────────
    g_net = p.add_argument_group("Network")
    g_net.add_argument('--substrate_hidden', type=int, default=128)
    g_net.add_argument('--vnr_hidden',       type=int, default=64)
    g_net.add_argument('--gru_hidden',       type=int, default=256)
    g_net.add_argument('--critic_hidden',    type=int, default=128)

    # ── HPSO ─────────────────────────────────────────────────────────────────
    g_hpso = p.add_argument_group("HPSO")
    g_hpso.add_argument('--hpso_particles',  type=int, default=20)
    g_hpso.add_argument('--hpso_iterations', type=int, default=30)

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--verbose', action='store_true', default=False)

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
    for n in substrate.nodes():
        nd = substrate.nodes[n]
        nd['cpu_res'] = nd.get('cpu', 100)
        nd.setdefault('mem', 50)
        nd['mem_res'] = nd['mem']
    for u, v in substrate.edges():
        ed = substrate[u][v]
        ed['bw_res'] = ed.get('bw', 100)

    vnr_pool = []
    for _ in range(args.vnr_pool_size):
        num_nodes = rng.randint(args.vnr_min_nodes, args.vnr_max_nodes)
        vnr = generate_single_vnr(
            num_nodes=num_nodes,
            edge_prob=0.5,
            cpu_range=(5, 20),
            bw_range=(5, 20),
        )
        for n in vnr.nodes():
            vnr.nodes[n].setdefault('mem',      rng.uniform(2, 10))
            vnr.nodes[n].setdefault('vnf_type', rng.randint(0, 4))
        vnr_pool.append(vnr)

    print(f"[Data] Substrate: {len(substrate.nodes())} nodes, "
          f"{len(substrate.edges())} edges")
    print(f"[Data] VNR pool : {len(vnr_pool)} VNRs")
    return substrate, vnr_pool


def get_window(vnr_pool: list, window_size: int, rng: random.Random) -> list:
    return rng.sample(vnr_pool, min(window_size, len(vnr_pool)))


def make_hpso_fn(args):
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
        print(f"[HPSO] fast_hpso unavailable ({e}). Using dummy embed.")

        def dummy_fn(sub, vnr):
            mapping = {n: n % len(sub.nodes()) for n in vnr.nodes()}
            return (mapping, {})

        return dummy_fn


def run_eval(trainer, substrate, vnr_pool, args, rng, hpso_fn):
    """Deterministic evaluation over n_eval fresh episodes."""
    trainer.agent.eval()
    results = []
    for _ in range(args.eval_episodes):
        window   = get_window(vnr_pool, args.window_size, rng)
        sub_copy = copy.deepcopy(substrate)
        _, _, m  = trainer.collect_and_update(window, sub_copy, hpso_fn)
        results.append(m)
    trainer.agent.train()
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def _print_header(args):
    mode = "PSD Curriculum" if args.psd else "Standard"
    print("=" * 70)
    print(f"  RL VNR Scheduler Training  [{mode}]")
    print("=" * 70)
    if args.psd:
        print(f"  patience={args.patience}  ar_thresh={args.ar_thresh}"
              f"  rc_thresh={args.rc_thresh}  max_load={args.max_load:.0%}")
    print(f"  episodes={args.episodes}  window={args.window_size}  "
          f"K_max={args.K_max}  lr={args.lr}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    rng  = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = {
        **DEFAULT_CFG,
        'K_max':            args.K_max,
        'lr':               args.lr,
        'gamma':            args.gamma,
        'gae_lambda':       args.gae_lambda,
        'clip_eps':         args.clip_eps,
        'entropy_coef':     args.entropy_coef,
        'value_coef':       args.value_coef,
        'k_epochs':         args.k_epochs,
        'substrate_hidden': args.substrate_hidden,
        'vnr_hidden':       args.vnr_hidden,
        'gru_hidden':       args.gru_hidden,
        'critic_hidden':    args.critic_hidden,
    }

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.data_path:
        substrate, vnr_pool = load_dataset(args.data_path)
    else:
        print("[INFO] No --data_path given. Using synthetic data.")
        substrate, vnr_pool = make_synthetic_data(args, rng)

    # ── HPSO fn ───────────────────────────────────────────────────────────────
    hpso_fn = make_hpso_fn(args)

    # ── Agent + Trainer ───────────────────────────────────────────────────────
    agent   = VNRSchedulerAgent(cfg)
    trainer = PPOTrainer(agent, cfg)
    history: list = []

    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        history = trainer.load(args.resume)

    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"[INFO] Agent parameters: {total_params:,}")
    print(f"[INFO] Device: {next(agent.parameters()).device}")
    _print_header(args)

    # ── PSD setup ─────────────────────────────────────────────────────────────
    if args.psd:
        real_substrate = copy.deepcopy(substrate)
        curriculum = CurriculumManager(
            real_substrate,
            patience  = args.patience,
            ar_thresh = args.ar_thresh,
            rc_thresh = args.rc_thresh,
            max_load  = args.max_load,
        )
        print(f"[PSD] Curriculum active. Initial substrate load: "
              f"{curriculum.substrate_load():.1%}")
    else:
        real_substrate = substrate
        curriculum     = None

    # ── Training loop ─────────────────────────────────────────────────────────
    t0            = time.time()
    commit_count  = 0

    for ep in range(1, args.episodes + 1):

        # PSD saturation check
        if curriculum and curriculum.is_saturated():
            print(f"\n[PSD] Substrate saturated "
                  f"({curriculum.substrate_load():.1%} >= {args.max_load:.0%}). "
                  f"Stopping at episode {ep}.")
            break

        # Sample a window from the VNR pool
        window = get_window(vnr_pool, args.window_size, rng)

        # Collect + PPO update
        # Trainer always deepcopies substrate internally for HPSO execution;
        # real_substrate is only read here, never mutated by trainer.
        accepted, rejected, metrics = trainer.collect_and_update(
            vnr_list      = window,
            substrate     = real_substrate,
            hpso_embed_fn = hpso_fn,
        )
        history.append(metrics)

        # ── Curriculum step ───────────────────────────────────────────────────
        if curriculum:
            should_commit = curriculum.step(metrics, accepted)
            if should_commit:
                commit_count += 1
                info = curriculum.commit()
                elapsed = time.time() - t0
                print(
                    f"\n[PSD] Commit #{commit_count} at ep={ep} | "
                    f"Best ep={info['episode']} "
                    f"(RC={info['rc_ratio']:.3f}, {info['n_vnrs']} VNRs) | "
                    f"Load: {info['load_before']:.1%} → {info['load_after']:.1%} | "
                    f"{elapsed:.0f}s"
                )

        # ── Periodic logging ──────────────────────────────────────────────────
        if ep % 10 == 0:
            recent   = history[-10:]
            avg_acc  = np.mean([h['acc_rate']    for h in recent])
            avg_rc   = np.mean([h['rc_ratio']    for h in recent])
            avg_loss = np.mean([h['total_loss']  for h in recent])
            avg_rew  = np.mean([h['reward_mean'] for h in recent])
            avg_rstd = np.mean([h['reward_std']  for h in recent])
            elapsed  = time.time() - t0

            load_str = ""
            if curriculum:
                load_str = f" | Load={curriculum.substrate_load():.1%}"

            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"Acc={avg_acc:.3f} | RC={avg_rc:.3f} | "
                f"R={avg_rew:.3f}±{avg_rstd:.3f} | "
                f"Loss={avg_loss:.4f}{load_str} | {elapsed:.0f}s"
            )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if ep % args.eval_every == 0:
            eval_sub = copy.deepcopy(real_substrate)
            m = run_eval(trainer, eval_sub, vnr_pool, args, rng, hpso_fn)
            load_str = f" | Load={curriculum.substrate_load():.1%}" if curriculum else ""
            print(
                f"  [EVAL] ep={ep} | "
                f"Acc={m['acc_rate']:.3f} | RC={m['rc_ratio']:.3f} | "
                f"R={m['reward_mean']:.3f}±{m['reward_std']:.3f}{load_str}"
            )

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if ep % args.save_every == 0:
            ckpt = args.save_path.replace('.pt', f'_ep{ep}.pt')
            trainer.save(ckpt, history)

    # ── Final save ────────────────────────────────────────────────────────────
    trainer.save(args.save_path, history)
    elapsed = time.time() - t0
    print("-" * 70)
    print(f"[INFO] Training complete in {elapsed:.0f}s")
    print(f"[INFO] Model saved → {args.save_path}")

    if curriculum:
        s = curriculum.summary()
        print(f"[PSD] Commits: {s['total_commits']} | "
              f"VNRs committed: {s['total_committed']} | "
              f"Final load: {s['final_load']:.1%}")

    last_n = min(50, len(history))
    if last_n > 0:
        print(f"[INFO] Last {last_n} eps avg: "
              f"Acc={np.mean([h['acc_rate'] for h in history[-last_n:]]):.3f} | "
              f"RC={np.mean([h['rc_ratio']  for h in history[-last_n:]]):.3f}")


if __name__ == '__main__':
    main()
