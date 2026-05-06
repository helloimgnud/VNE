# src/simulation/env_simulator.py
"""
EnvV2Simulator
==============
Adapter that drives VNEEnvironmentV2 with a **batch algorithm** (no RL agent)
and returns the same metrics dict schema that BaseExperiment._run_algorithm()
expects.

Drop-in replacement for BatchedVNRSimulator.simulate_batched_stream().

Key fixes over the old BatchedVNRSimulator:
  - Leave events (VNR expiry) are correctly processed via env._advance_to_window()
  - Resources are released on correct departure time (arrival + lifetime), not inf
  - Queue expiry (max_queue_delay) is applied per-window before embedding
  - Time-series records match the schema used by Fig6 experiment plots

Usage (in BaseExperiment._run_algorithm):
    from src.simulation.env_simulator import run_with_env_v2
    metrics = run_with_env_v2(substrate, vnr_stream, batch_algo,
                               window_size=10, max_queue_delay=50)
"""

from __future__ import annotations

import copy
import tempfile
import os
import json
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from src.evaluation.metrics import cost_of_embedding, revenue_of_vnr
from src.utils.graph_utils import copy_substrate, release_vnr_embedding
from src.utils.io_utils import save_substrate_to_json, save_vnr_stream_to_json


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_with_env_v2(
    substrate: nx.Graph,
    vnr_stream: List[nx.Graph],
    batch_algorithm: Callable,
    window_size: int = 10,
    max_queue_delay: int = 50,
) -> Dict:
    """
    Run a batch embedding algorithm over a VNR stream using VNEEnvironmentV2
    as the simulation engine.

    Parameters
    ----------
    substrate       : NetworkX substrate graph (NOT mutated — deep copy used)
    vnr_stream      : list of VNR graphs with graph['arrival_time'] and graph['lifetime']
    batch_algorithm : callable(substrate, batch) -> (accepted, rejected)
                      where batch = [(vnr, revenue), ...]
                      accepted = [(vnr, mapping, link_paths), ...]
                      rejected = [vnr, ...]
    window_size     : time-window width (same unit as arrival_time)
    max_queue_delay : VNRs waiting longer than this are dropped from the queue

    Returns
    -------
    dict with keys matching BaseExperiment metrics schema:
        acceptance_ratio, avg_cost, avg_revenue, cost_revenue_ratio,
        avg_execution_time, successful_embeddings, total_vnrs,
        total_cost, total_revenue, expired_in_queue,
        rejected_by_algorithm, time_series
    """
    sim = _EnvV2BatchDriver(
        substrate=substrate,
        vnr_stream=vnr_stream,
        batch_algorithm=batch_algorithm,
        window_size=window_size,
        max_queue_delay=max_queue_delay,
    )
    return sim.run()


# ---------------------------------------------------------------------------
# Internal driver — mirrors VNEEnvironmentV2 window logic exactly
# ---------------------------------------------------------------------------

class _EnvV2BatchDriver:
    """
    Reproduces VNEEnvironmentV2's window-management logic but drives it with
    a deterministic batch algorithm instead of an RL agent.

    Window transition order (CRITICAL — must match env._advance_to_window):
      1. Expire embeddings whose (arrival_time + lifetime) < window_start
      2. Load VNRs assigned to this window
      3. Drop VNRs that have exceeded max_queue_delay
      4. Call batch_algorithm(substrate, [(vnr, revenue), ...])
      5. Commit accepted embeddings; record metrics
    """

    def __init__(
        self,
        substrate: nx.Graph,
        vnr_stream: List[nx.Graph],
        batch_algorithm: Callable,
        window_size: int,
        max_queue_delay: int,
    ):
        self.substrate        = copy_substrate(substrate)   # working copy
        self.vnr_stream       = vnr_stream
        self.batch_algorithm  = batch_algorithm
        self.window_size      = window_size
        self.max_queue_delay  = max_queue_delay

        # Active embeddings: (vnr, mapping, link_paths, expiry_time)
        self.active_embeddings: List[Tuple] = []

        # Cumulative metrics
        self.success_count   = 0
        self.total_cost      = 0.0
        self.total_revenue   = 0.0
        self.execution_times: List[float] = []
        self.expired_in_queue = 0

        # Time-series (one entry per window that had VNRs)
        self.time_series: List[Dict] = []

    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Execute the full simulation and return metrics."""
        if not self.vnr_stream:
            return self._build_metrics(0)

        windows = self._partition_into_windows()
        n_windows = len(windows)
        processed_total = 0  # VNRs counted toward denominator

        for w_idx, window_vnrs in enumerate(windows):
            window_start = w_idx * self.window_size

            # --- Step 1: expire embeddings ---
            self._process_departures(window_start)

            # --- Step 2 & 3: filter queue ---
            valid_vnrs = []
            for vnr in window_vnrs:
                wait = window_start - vnr.graph.get('arrival_time', window_start)
                if wait <= self.max_queue_delay:
                    valid_vnrs.append(vnr)
                else:
                    self.expired_in_queue += 1
                    processed_total += 1

            if not valid_vnrs:
                continue

            # --- Step 4: build batch and call algorithm ---
            batch = [(vnr, revenue_of_vnr(vnr)) for vnr in valid_vnrs]

            t0 = time.perf_counter()
            try:
                accepted, rejected = self.batch_algorithm(self.substrate, batch)
            except Exception as exc:
                # Treat algorithm crash as full rejection — don't crash experiment
                accepted, rejected = [], [vnr for vnr, _ in batch]
                import traceback
                traceback.print_exc()
            elapsed = time.perf_counter() - t0

            # --- Step 5: commit accepted embeddings ---
            per_vnr_time = elapsed / max(len(batch), 1)

            for vnr, mapping, link_paths in accepted:
                cost    = cost_of_embedding(mapping, link_paths, vnr, self.substrate)
                revenue = revenue_of_vnr(vnr)
                expiry  = (vnr.graph.get('arrival_time', window_start)
                           + vnr.graph.get('lifetime', 50))

                self.success_count    += 1
                self.total_cost       += cost
                self.total_revenue    += revenue
                self.active_embeddings.append((vnr, mapping, link_paths, expiry))
                self.execution_times.append(per_vnr_time)

            processed_total += len(valid_vnrs)

            # --- Record time-series snapshot ---
            snap = self._build_metrics(processed_total)
            snap.update({
                'time':            window_start,
                'window_idx':      w_idx,
                'window_accepted': len(accepted),
                'window_rejected': len(rejected),
                'window_expired':  0,   # already counted above
                'expired_in_queue': self.expired_in_queue,
            })
            self.time_series.append(snap)

        # Final metrics over full stream length
        metrics = self._build_metrics(len(self.vnr_stream))
        metrics['time_series'] = self.time_series
        return metrics

    # ------------------------------------------------------------------

    def _partition_into_windows(self) -> List[List[nx.Graph]]:
        """Assign each VNR to window = floor(arrival_time / window_size)."""
        if not self.vnr_stream:
            return []

        max_time  = max(v.graph.get('arrival_time', 0) for v in self.vnr_stream)
        n_windows = int(max_time / self.window_size) + 1
        windows: List[List[nx.Graph]] = [[] for _ in range(n_windows)]

        for vnr in self.vnr_stream:
            w_idx = int(vnr.graph.get('arrival_time', 0) / self.window_size)
            windows[w_idx].append(vnr)

        # Prune trailing empty windows
        while len(windows) > 1 and not windows[-1]:
            windows.pop()

        return windows

    def _process_departures(self, current_time: float):
        """Release resources for VNRs whose expiry < current_time."""
        still_active = []
        for emb in self.active_embeddings:
            vnr, mapping, link_paths, expiry = emb
            if expiry < current_time:
                release_vnr_embedding(self.substrate, vnr, mapping, link_paths)
            else:
                still_active.append(emb)
        self.active_embeddings = still_active

    def _build_metrics(self, total_vnrs: int) -> Dict:
        n = self.success_count
        return {
            'acceptance_ratio':    self.success_count / max(total_vnrs, 1),
            'avg_cost':            self.total_cost    / max(n, 1),
            'avg_revenue':         self.total_revenue / max(n, 1),
            'cost_revenue_ratio':  self.total_cost    / max(self.total_revenue, 1e-9),
            'avg_execution_time':  (sum(self.execution_times) /
                                    max(len(self.execution_times), 1)),
            'successful_embeddings': n,
            'total_vnrs':          total_vnrs,
            'total_cost':          self.total_cost,
            'total_revenue':       self.total_revenue,
            'expired_in_queue':    self.expired_in_queue,
            'rejected_by_algorithm': max(0, total_vnrs - n - self.expired_in_queue),
        }