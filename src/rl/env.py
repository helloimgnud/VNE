"""
src/rl/env.py
=============
MDP environment for one VNR scheduling time window.

Wraps the interaction between the RL agent and the HPSO solver.
Each episode = one time window with K VNRs; the agent selects the processing
order and HPSO provides the embedding outcome (reward signal).

MDP formulation (idea.md §3):
  State  s_t = (G_p_current, active_VNRs, mask_t)
  Action a_t = index of next VNR to embed
  Reward r_t = R/C ratio if HPSO succeeds, else -penalty
  Terminal    = all K VNRs processed

State encoding is handled externally by the agent (DGL graph + GATEncoder);
this class is responsible for the *transition* logic only.
"""

import copy
import torch
from typing import Callable, List, Optional, Tuple

from src.rl.utils import DEFAULT_CFG


class SchedulerEnv:
    """
    Thin environment wrapper for one time-window scheduling episode.

    Parameters
    ----------
    vnr_list    : list of networkx VNR graphs for this window
    substrate   : mutable networkx substrate graph
                  (This env takes ownership of a deep copy; the original is
                   NOT modified during training.)
    hpso_embed_fn : callable(substrate, vnr) → (mapping, link_paths) | None
    penalty     : penalty reward for failed HPSO embedding
    """

    def __init__(
        self,
        vnr_list:      list,
        substrate,
        hpso_embed_fn: Callable,
        penalty:       float = 0.1,
    ):
        self.vnr_list  = vnr_list
        # Work on an internal copy to avoid mutating the caller's substrate
        self.substrate = copy.deepcopy(substrate)
        self.hpso_fn   = hpso_embed_fn
        self.penalty   = penalty

        self.K          = len(vnr_list)
        self.processed  = [False] * self.K
        self.step_count = 0

        # Cached results for post-episode analysis
        self.results:    List[Optional[Tuple]] = [None] * self.K
        self.rewards:    List[float]           = []

    # ------------------------------------------------------------------

    @property
    def active_mask(self) -> List[bool]:
        """Boolean list: True = VNR still in queue."""
        return [not p for p in self.processed]

    @property
    def done(self) -> bool:
        return all(self.processed)

    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[float, bool]:
        """
        Process the VNR at position `action`.

        Parameters
        ----------
        action : int  — index into vnr_list (must be unprocessed)

        Returns
        -------
        reward : float
        done   : bool
        """
        if self.processed[action]:
            raise ValueError(f"VNR {action} already processed.")

        vnr    = self.vnr_list[action]
        result = self.hpso_fn(self.substrate, vnr)

        self.results[action]  = result
        self.processed[action] = True
        self.step_count       += 1

        reward = self._compute_reward(result, vnr)
        self.rewards.append(reward)

        return reward, self.done

    # ------------------------------------------------------------------

    def _compute_reward(self, result, vnr) -> float:
        """
        Reward = R/C ratio if embedding succeeds, else -penalty.
        (idea.md §3.3)
        """
        if result is None:
            return -self.penalty

        try:
            from src.evaluation.eval import revenue_of_vnr
            mapping, link_paths = result
            rev = revenue_of_vnr(vnr)

            # Simple cost proxy: sum of CPU demands (full cost requires substrate)
            cost = sum(vnr.nodes[n].get('cpu', 1.0) for n in vnr.nodes()) + \
                   sum(vnr.edges[e].get('bw', 1.0)  for e in vnr.edges())
            if cost <= 0:
                return 0.0
            return rev / cost
        except Exception:
            return 0.0

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return episode summary statistics."""
        accepted = [(i, r) for i, r in enumerate(self.results)
                    if r is not None and self.processed[i]]
        rejected_count = sum(1 for r in self.results
                             if r is None and self.processed[self.results.index(r)]
                             ) if len(self.results) > 0 else 0

        # Simpler: count non-None results
        n_acc = sum(1 for r in self.results if r is not None)
        n_rej = self.K - n_acc

        return dict(
            n_total     = self.K,
            n_accepted  = n_acc,
            n_rejected  = n_rej,
            acc_rate    = n_acc / self.K if self.K > 0 else 0.0,
            total_reward= sum(self.rewards),
            avg_reward  = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0,
        )
