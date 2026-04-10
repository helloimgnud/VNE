"""
src/training/evaluate.py
=========================
Inference and evaluation utilities for the trained VNRScheduler.

Functions
---------
evaluate_scheduler(scheduler, substrate_fn, batch_fn, n_episodes)
    → Run n episodes with the trained scheduler and report AR / R/C vs
      baseline (revenue-sort).

run_inference(scheduler, substrate, vnr_list)
    → Return the GNN-ordered list of VNR indices for a single batch.

EvaluationReport.compare_baseline(...)
    → Run both sorting strategies and produce a side-by-side comparison dict.

Usage (CLI)
-----------
  python -m src.training.evaluate \
      --checkpoint checkpoints/reinforce_phase1_final.pt \
      --episodes 100 --sub-nodes 50 --vnr-batch 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.scheduler.model import VNRScheduler
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
from src.evaluation.eval import revenue_of_vnr, cost_of_vnr
from src.algorithms.fast_hpso import hpso_embed
from src.utils.graph_utils import copy_substrate
from src.training.generate_data import make_env_fns


# ---------------------------------------------------------------------------
# Single-batch inference
# ---------------------------------------------------------------------------

def run_inference(
    scheduler,
    substrate,
    vnr_list: list,
) -> List[int]:
    """
    Compute a GNN-based processing order for one batch of VNRs.

    Parameters
    ----------
    scheduler : VNRScheduler
    substrate : networkx substrate graph (READ-ONLY; not modified)
    vnr_list  : list of VNR networkx graphs

    Returns
    -------
    list[int] : indices into vnr_list, highest-priority VNR first
    """
    sub_data  = substrate_to_pyg(substrate)
    vnr_datas = [vnr_to_pyg(v) for v in vnr_list]
    scores    = scheduler.predict(sub_data, vnr_datas)   # [B], no grad
    order     = scores.argsort(descending=True).tolist()
    return order


# ---------------------------------------------------------------------------
# Batch embedding wrappers for evaluation
# ---------------------------------------------------------------------------

def _embed_with_order(substrate, vnr_list, order, hpso_params) -> Tuple[list, list]:
    """Embed VNRs in the given order using HPSO."""
    accepted, rejected = [], []
    substrate = copy_substrate(substrate)   # work on a copy

    for idx in order:
        vnr    = vnr_list[idx]
        result = hpso_embed(substrate_graph=substrate, vnr_graph=vnr, **hpso_params)
        if result is not None:
            mapping, link_paths = result
            accepted.append((vnr, mapping, link_paths))
        else:
            rejected.append(vnr)

    return accepted, rejected


def _revenue_sort(vnr_list: list) -> List[int]:
    return sorted(
        range(len(vnr_list)),
        key=lambda i: revenue_of_vnr(vnr_list[i]),
        reverse=True,
    )


def _episode_metrics(accepted, rejected) -> dict:
    n_total = len(accepted) + len(rejected)
    ar      = len(accepted) / (n_total + 1e-9)
    rev     = sum(revenue_of_vnr(v) for v, _, _ in accepted)
    cost    = sum(cost_of_vnr(v)    for v, _, _ in accepted)
    rc      = rev / (cost + 1e-6)
    return dict(acc_rate=ar, rc_ratio=rc, n_accepted=len(accepted), n_total=n_total)


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Results of a comparative evaluation run."""
    n_episodes: int
    # Scheduler (GNN ordering)
    scheduler_ar:  float
    scheduler_rc:  float
    # Baseline (revenue-sort ordering)
    baseline_ar:   float
    baseline_rc:   float

    @property
    def delta_ar(self) -> float:
        return self.scheduler_ar - self.baseline_ar

    @property
    def delta_rc(self) -> float:
        return self.scheduler_rc - self.baseline_rc

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report ({self.n_episodes} episodes) ===",
            f"  Acceptance Rate : scheduler={self.scheduler_ar:.2%}  "
            f"baseline={self.baseline_ar:.2%}  Δ={self.delta_ar:+.2%}",
            f"  Revenue/Cost    : scheduler={self.scheduler_rc:.3f}  "
            f"baseline={self.baseline_rc:.3f}  Δ={self.delta_rc:+.3f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dict(
            n_episodes   = self.n_episodes,
            scheduler_ar = self.scheduler_ar,
            scheduler_rc = self.scheduler_rc,
            baseline_ar  = self.baseline_ar,
            baseline_rc  = self.baseline_rc,
            delta_ar     = self.delta_ar,
            delta_rc     = self.delta_rc,
        )


def create_depleted_substrate(substrate_fn, batch_fn, target_cpu_util: float = 0.5, hpso_params: dict = None) -> nx.Graph:
    """
    Artificially deplete a substrate by embedding random VNRs until a target
    CPU utilisation is reached. Returns the depleted substrate.
    """
    _hpso = hpso_params or dict(particles=20, iterations=30)
    substrate = substrate_fn()
    from src.utils.graph_utils import substrate_utilisation
    
    while True:
        util = substrate_utilisation(substrate)
        if util['cpu_util'] >= target_cpu_util:
            break
            
        vnr_list = batch_fn()
        for vnr in vnr_list:
            result = hpso_embed(substrate_graph=substrate, vnr_graph=vnr, **_hpso)
            if result is not None:
                mapping, link_paths = result
                # Commit (hpso_embed already mutates the substrate)
                pass
            
            util = substrate_utilisation(substrate)
            if util['cpu_util'] >= target_cpu_util:
                break
    
    return substrate


def evaluate_scheduler(
    scheduler,
    substrate_fn: Callable,
    batch_fn:     Callable,
    n_episodes:   int  = 100,
    hpso_params:  Optional[dict] = None,
    verbose:      bool = True,
    depletion_target: float = 0.0,
) -> EvaluationReport:
    """
    Run n_episodes of both strategies (GNN-ordered and revenue-sort) and
    compare Acceptance Rate and R/C ratio.

    Parameters
    ----------
    scheduler        : VNRScheduler (trained)
    substrate_fn     : callable() → fresh substrate graph per episode
    batch_fn         : callable() → fresh VNR list per episode
    n_episodes       : number of evaluation episodes
    hpso_params      : override HPSO hyper-parameters
    verbose          : print per-episode progress
    depletion_target : float [0,1], if >0, artificially pre-fill substrate before evaluating

    Returns
    -------
    EvaluationReport
    """
    _hpso = hpso_params or dict(particles=20, iterations=30)

    sched_ars, sched_rcs = [], []
    base_ars,  base_rcs  = [], []

    scheduler.eval()

    for ep in range(n_episodes):
        if depletion_target > 0:
            substrate = create_depleted_substrate(substrate_fn, batch_fn, depletion_target, _hpso)
        else:
            substrate = substrate_fn()
            
        vnr_list  = batch_fn()

        # --- Scheduler ordering ---
        sched_order  = run_inference(scheduler, substrate, vnr_list)
        s_acc, s_rej = _embed_with_order(substrate, vnr_list, sched_order,  _hpso)
        s_metrics    = _episode_metrics(s_acc, s_rej)

        # --- Baseline ordering ---
        base_order   = _revenue_sort(vnr_list)
        b_acc, b_rej = _embed_with_order(substrate, vnr_list, base_order, _hpso)
        b_metrics    = _episode_metrics(b_acc, b_rej)

        sched_ars.append(s_metrics["acc_rate"])
        sched_rcs.append(s_metrics["rc_ratio"])
        base_ars.append(b_metrics["acc_rate"])
        base_rcs.append(b_metrics["rc_ratio"])

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            print(
                f"  Ep {ep+1:4d}: "
                f"sched AR={s_metrics['acc_rate']:.2%} RC={s_metrics['rc_ratio']:.3f} | "
                f"base  AR={b_metrics['acc_rate']:.2%} RC={b_metrics['rc_ratio']:.3f}"
            )

    report = EvaluationReport(
        n_episodes   = n_episodes,
        scheduler_ar = sum(sched_ars) / n_episodes,
        scheduler_rc = sum(sched_rcs) / n_episodes,
        baseline_ar  = sum(base_ars)  / n_episodes,
        baseline_rc  = sum(base_rcs)  / n_episodes,
    )
    if verbose:
        print()
        print(report.summary())

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained VNRScheduler vs revenue-sort baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to .pt checkpoint (saved by VNRScheduler.save)")
    p.add_argument("--episodes",    type=int, default=100)
    p.add_argument("--sub-nodes",   type=int, default=50)
    p.add_argument("--vnr-batch",   type=int, default=10)
    p.add_argument("--vnr-nodes",   type=int, default=4)
    p.add_argument("--hpso-iter",   type=int, default=30)
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--no-ctx",      action="store_true")
    p.add_argument("--depletion",   type=float, default=0.0, help="Pre-fill substrate CPU to this %%")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    scheduler = VNRScheduler.load(
        args.checkpoint,
        device=device,
        use_batch_context=not args.no_ctx,
    )

    substrate_fn, batch_fn = make_env_fns(
        substrate_nodes = args.sub_nodes,
        batch_size      = args.vnr_batch,
        vnr_nodes       = args.vnr_nodes,
    )

    print(f"[Evaluate] checkpoint  : {args.checkpoint}")
    print(f"[Evaluate] episodes    : {args.episodes}")
    print(f"[Evaluate] substrate   : {args.sub_nodes} nodes")
    print(f"[Evaluate] VNR batch   : {args.vnr_batch}")
    if args.depletion > 0:
        print(f"[Evaluate] Depletion : Pre-filled to {args.depletion:.0%}")
    print()

    report = evaluate_scheduler(
        scheduler,
        substrate_fn,
        batch_fn,
        n_episodes  = args.episodes,
        depletion_target = args.depletion,
        hpso_params = dict(particles=20, iterations=args.hpso_iter),
    )
    print(report.summary())
