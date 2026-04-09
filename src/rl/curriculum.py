"""
src/rl/curriculum.py
====================
Progressive Substrate Depletion (PSD) Curriculum Manager.

Controls the training curriculum by:
  1. Tracking a rolling buffer of `patience` episodes.
  2. When ALL episodes in the buffer satisfy AccRate >= ar_thresh AND
     RC >= rc_thresh, selecting the BEST-RC episode's embeddings and
     committing them permanently to the real substrate.
  3. No VNR lifetimes: resources are deducted once and never freed.
  4. Training stops when substrate_load() >= max_load.

Usage
-----
    curriculum = CurriculumManager(real_substrate, patience=10)

    for ep in range(max_episodes):
        if curriculum.is_saturated():
            break
        accepted, _, metrics = trainer.collect_and_update(
            window, real_substrate, hpso_fn
        )
        if curriculum.step(metrics, accepted):
            info = curriculum.commit()
            print(f"Committed ep={info['episode']}, RC={info['rc_ratio']:.3f}, "
                  f"load {info['load_before']:.1%} -> {info['load_after']:.1%}")
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Dict, List, Optional, Tuple


class CurriculumManager:
    """
    Progressive Substrate Depletion curriculum manager.

    Parameters
    ----------
    real_substrate : NetworkX graph
        The live physical substrate. **Modified in-place** on each commit.
        The caller must NOT modify it between episodes (use deepcopies for HPSO).
    patience : int
        Number of consecutive episodes that must all satisfy the mastery
        thresholds before a commit is triggered. Default 10.
    ar_thresh : float
        Minimum AccRate (acceptance rate) for an episode to count as
        "mastered". Default 0.95.
    rc_thresh : float
        Minimum RC ratio for an episode to count as "mastered". Default 0.90.
    max_load : float
        Fraction of total CPU capacity at which training is considered
        saturated and should stop. Default 0.85.
    """

    def __init__(
        self,
        real_substrate,
        patience:  int   = 10,
        ar_thresh: float = 0.95,
        rc_thresh: float = 0.90,
        max_load:  float = 0.85,
    ):
        self.real_substrate = real_substrate
        self.patience   = patience
        self.ar_thresh  = ar_thresh
        self.rc_thresh  = rc_thresh
        self.max_load   = max_load

        # Rolling buffer: each entry is a dict with keys:
        #   ep, acc_rate, rc_ratio, accepted
        self._buffer: deque = deque(maxlen=patience)

        # Baseline total CPU (computed once at construction)
        self._total_cpu: float = self._compute_total_cpu()

        # Tracking
        self._committed_count: int   = 0
        self._episode_counter: int   = 0
        self._commit_log: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metrics: dict, accepted: list) -> bool:
        """
        Register one completed episode. Returns True if a commit should fire.

        Parameters
        ----------
        metrics  : dict returned by PPOTrainer.collect_and_update()
                   Must contain 'acc_rate' and 'rc_ratio'.
        accepted : list of (vnr, mapping, link_paths) from the same episode.

        Returns
        -------
        bool — True means: call commit() now.
        """
        self._episode_counter += 1

        entry = {
            'ep':       self._episode_counter,
            'acc_rate': float(metrics.get('acc_rate', 0.0)),
            'rc_ratio': float(metrics.get('rc_ratio', 0.0)),
            'accepted': accepted,   # (vnr, mapping, link_paths) list
        }
        self._buffer.append(entry)

        return self._should_commit()

    def commit(self) -> dict:
        """
        Commit the best-RC episode's embeddings to the real substrate.

        Selects the entry in the current patience window with the highest
        rc_ratio, then permanently deducts its resource usage from
        real_substrate. Clears the buffer afterward so the patience counter
        resets.

        Returns
        -------
        dict with keys: episode, rc_ratio, n_vnrs, load_before, load_after,
                        n_nodes_affected, n_links_affected
        """
        best = self._best_in_window()
        load_before = self.substrate_load()

        n_nodes, n_links = self._apply_embeddings(best['accepted'])

        self._committed_count += len(best['accepted'])
        self._buffer.clear()

        load_after = self.substrate_load()

        info = {
            'episode':          best['ep'],
            'rc_ratio':         best['rc_ratio'],
            'acc_rate':         best['acc_rate'],
            'n_vnrs':           len(best['accepted']),
            'load_before':      load_before,
            'load_after':       load_after,
            'n_nodes_affected': n_nodes,
            'n_links_affected': n_links,
        }
        self._commit_log.append(info)
        return info

    def substrate_load(self) -> float:
        """
        Fraction of total CPU capacity consumed: 1 - (sum cpu_res / total_cpu).
        Returns 0.0 if total_cpu == 0.
        """
        if self._total_cpu <= 0:
            return 0.0
        residual = sum(
            float(self.real_substrate.nodes[n].get('cpu_res',
                  self.real_substrate.nodes[n].get('cpu', 0)))
            for n in self.real_substrate.nodes()
        )
        return max(0.0, 1.0 - residual / self._total_cpu)

    def is_saturated(self) -> bool:
        """Returns True when substrate_load() >= max_load."""
        return self.substrate_load() >= self.max_load

    def best_in_window(self) -> Optional[dict]:
        """
        Returns the buffer entry with the highest rc_ratio.
        Returns None if the buffer is empty.
        """
        return self._best_in_window()

    def summary(self) -> dict:
        """Return full training summary for logging / analysis."""
        return {
            'total_episodes':   self._episode_counter,
            'total_commits':    len(self._commit_log),
            'total_committed':  self._committed_count,
            'final_load':       self.substrate_load(),
            'commit_log':       self._commit_log,
        }

    def reset(self, new_substrate=None) -> None:
        """
        Optionally replace the substrate (e.g. after a simulation reset).
        Clears the buffer and resets the total-CPU baseline.
        """
        if new_substrate is not None:
            self.real_substrate = new_substrate
        self._buffer.clear()
        self._total_cpu = self._compute_total_cpu()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_commit(self) -> bool:
        """True when buffer is full AND every entry passes both thresholds."""
        if len(self._buffer) < self.patience:
            return False
        return all(
            e['acc_rate'] >= self.ar_thresh and e['rc_ratio'] >= self.rc_thresh
            for e in self._buffer
        )

    def _best_in_window(self) -> Optional[dict]:
        if not self._buffer:
            return None
        return max(self._buffer, key=lambda e: e['rc_ratio'])

    def _compute_total_cpu(self) -> float:
        """Sum of initial cpu (or cpu_res) across all substrate nodes."""
        total = 0.0
        for n in self.real_substrate.nodes():
            nd = self.real_substrate.nodes[n]
            total += float(nd.get('cpu', nd.get('cpu_res', 0.0)))
        return total

    def _apply_embeddings(
        self,
        accepted: list,
    ) -> Tuple[int, int]:
        """
        Permanently deduct resources from real_substrate for each embedding.

        Returns
        -------
        (n_nodes_affected, n_links_affected)
        """
        nodes_affected: set = set()
        links_affected: set = set()

        for item in accepted:
            # accepted entries are (vnr, mapping, link_paths)
            if len(item) != 3:
                continue
            vnr, mapping, link_paths = item

            # ── Node resources ───────────────────────────────────────
            for vnode, pnode in mapping.items():
                cpu_req = float(vnr.nodes[vnode].get('cpu', 0.0))
                if self.real_substrate.has_node(pnode):
                    nd = self.real_substrate.nodes[pnode]
                    nd['cpu_res'] = max(
                        0.0,
                        float(nd.get('cpu_res', nd.get('cpu', 0.0))) - cpu_req
                    )
                    nodes_affected.add(pnode)

            # ── Link resources ────────────────────────────────────────
            for edge in vnr.edges():
                u, v   = edge
                bw_req = float(vnr.edges[u, v].get('bw', 0.0))

                # link_paths may store (u,v) or (v,u) as key
                path = (link_paths.get((u, v))
                        or link_paths.get((v, u))
                        or [])

                for p, q in zip(path[:-1], path[1:]):
                    for a, b in [(p, q), (q, p)]:
                        if self.real_substrate.has_edge(a, b):
                            ed = self.real_substrate[a][b]
                            ed['bw_res'] = max(
                                0.0,
                                float(ed.get('bw_res', ed.get('bw', 0.0))) - bw_req
                            )
                            links_affected.add((a, b))

        return len(nodes_affected), len(links_affected)
