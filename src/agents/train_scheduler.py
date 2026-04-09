"""
train_scheduler.py
==================
Training script for VNRSchedulerAgent.

Usage:
    python train_scheduler.py --episodes 500 --K_max 20 --save_path model.pt
"""

import argparse
import copy
import os
import pickle
import time
import random
import torch
import numpy as np

from src.agents.vnr_scheduler import (
    VNRSchedulerAgent, PPOTrainer,
    build_vnr_dgl, build_substrate_dgl,
    hpso_batch_rl,
    DEFAULT_CFG,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes',   type=int,   default=500)
    p.add_argument('--K_max',      type=int,   default=20,    help='Max VNRs per window')
    p.add_argument('--window_size',type=int,   default=10,    help='Actual VNRs per window')
    p.add_argument('--save_path',  type=str,   default='result/scheduler_model.pt')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--gamma',      type=float, default=0.98)
    p.add_argument('--clip_eps',   type=float, default=0.2)
    p.add_argument('--k_epochs',   type=int,   default=4)
    p.add_argument('--eval_every', type=int,   default=50)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(path: str):
    """
    Load pre-generated VNR list and substrate graph.
    Expected pickle format: (substrate_nx, [vnr_nx, ...])
    Adapt to your actual dataset format.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data  # (substrate, vnr_list)


def get_window(vnr_pool, window_size, rng=None):
    """Sample a random window of VNRs from the pool."""
    if rng is None:
        rng = random.Random()
    return rng.sample(vnr_pool, min(window_size, len(vnr_pool)))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # --- Config ---
    cfg = {**DEFAULT_CFG,
           'K_max':      args.K_max,
           'lr':         args.lr,
           'gamma':      args.gamma,
           'clip_eps':   args.clip_eps,
           'k_epochs':   args.k_epochs,
           }

    # --- Agent & Trainer ---
    agent  = VNRSchedulerAgent(cfg)
    trainer = PPOTrainer(agent, cfg)

    # --- Load data ---
    # TODO: replace with your actual data loading
    # substrate, vnr_pool = load_dataset('data/geant_vnrs.pkl')
    # For demonstration we create dummy graphs:
    print("[INFO] Using synthetic dummy data. Replace with real dataset.")
    import networkx as nx
    substrate = nx.path_graph(23, create_using=nx.DiGraph())
    for n in substrate.nodes():
        substrate.nodes[n].update({'cpu': 50, 'mem': 50, 'cpu_res': 50, 'mem_res': 50})
    for u, v in substrate.edges():
        substrate[u][v]['bw'] = 100
        substrate[u][v]['bw_res'] = 100
    vnr_pool = []
    for _ in range(200):
        vnr = nx.path_graph(random.randint(3, 8), create_using=nx.DiGraph())
        for n in vnr.nodes():
            vnr.nodes[n].update({
                'cpu': random.uniform(5, 20),
                'mem': random.uniform(5, 20),
                'vnf_type': random.randint(0, 4),
            })
        for u, v in vnr.edges():
            vnr[u][v]['bw'] = random.uniform(10, 20)
        vnr_pool.append(vnr)

    # --- HPSO embed function (single VNR) ---
    try:
        from src.algorithms.fast_hpso import hpso_embed as _hpso
        def hpso_fn(sub, vnr):
            return _hpso(substrate_graph=sub, vnr_graph=vnr,
                         particles=20, iterations=30)
    except ImportError:
        print("[WARN] hpso_embed not found. Using dummy accept-all embed.")
        def hpso_fn(sub, vnr):
            # Dummy: always accept with trivial mapping
            mapping = {n: n % len(sub.nodes()) for n in vnr.nodes()}
            return (mapping, {})

    # --- Training loop ---
    rng = random.Random(args.seed)
    history = []

    print(f"[INFO] Starting training for {args.episodes} episodes")
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        # Sample window
        window = get_window(vnr_pool, args.window_size, rng)
        sub_copy = copy.deepcopy(substrate)

        accepted, rejected, metrics = trainer.collect_and_update(
            vnr_list=window,
            substrate=sub_copy,
            hpso_embed_fn=hpso_fn,
            build_vnr_dgl_fn=build_vnr_dgl,
            build_sub_dgl_fn=build_substrate_dgl,
        )
        history.append(metrics)

        if ep % 10 == 0:
            avg_acc = np.mean([h['acc_rate']  for h in history[-10:]])
            avg_rc  = np.mean([h['rc_ratio']  for h in history[-10:]])
            avg_loss= np.mean([h['total_loss']for h in history[-10:]])
            elapsed = time.time() - t0
            print(f"Ep {ep:4d}/{args.episodes} | "
                  f"AccRate={avg_acc:.3f} | RC={avg_rc:.3f} | "
                  f"Loss={avg_loss:.4f} | {elapsed:.0f}s")

        if ep % args.eval_every == 0:
            # Quick deterministic eval
            eval_metrics = []
            for _ in range(5):
                window = get_window(vnr_pool, args.window_size, rng)
                sub_copy = copy.deepcopy(substrate)
                acc, rej, m = trainer.collect_and_update(
                    window, sub_copy, hpso_fn,
                    build_vnr_dgl, build_substrate_dgl
                )
                eval_metrics.append(m)
            print(f"  [EVAL] AccRate={np.mean([m['acc_rate'] for m in eval_metrics]):.3f} "
                  f"RC={np.mean([m['rc_ratio'] for m in eval_metrics]):.3f}")

    # --- Save model ---
    torch.save({
        'model_state': agent.state_dict(),
        'cfg': cfg,
        'history': history,
    }, args.save_path)
    print(f"[INFO] Model saved to {args.save_path}")


if __name__ == '__main__':
    main()
