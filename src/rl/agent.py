"""
src/rl/agent.py
===============
High-level VNRSchedulerAgent — wraps VNRSchedulerNetwork and exposes
the rollout / evaluate_actions / forward_rl_order interface.

Separating the "agent logic" (rollout, ordering) from the "network definition"
(networks.py) keeps both files focused and easier to test independently.
"""

import copy
import torch
import torch.nn.functional as F
import dgl
from typing import Callable, Dict, List, Optional, Tuple

from src.rl.networks import VNRSchedulerNetwork
from src.rl.utils import DEFAULT_CFG, build_vnr_dgl, build_substrate_dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VNRSchedulerAgent(VNRSchedulerNetwork):
    """
    Actor-critic agent for VNR scheduling.

    Inherits all network components from VNRSchedulerNetwork and adds:
      - rollout()           : collect one episode trajectory
      - evaluate_actions()  : re-evaluate stored actions with current policy
      - forward_rl_order()  : deterministic ordering for inference (hpso_batch)
      - state_dict / load   : model serialisation helpers

    Parameters
    ----------
    cfg : dict
        Overrides for DEFAULT_CFG.  Only the keys you want to change need
        to be specified; missing keys use DEFAULT_CFG defaults.
    """

    def __init__(self, cfg: Optional[dict] = None):
        merged = {**DEFAULT_CFG, **(cfg or {})}
        super().__init__(merged)

    # ------------------------------------------------------------------
    # Episode rollout  (used during training)
    # ------------------------------------------------------------------

    def rollout(
        self,
        g_sub:      dgl.DGLGraph,
        vnr_graphs: List[Optional[dgl.DGLGraph]],
        K_real:     int,
        deterministic: bool = False,
    ) -> Dict:
        """
        Run one full episode (scheduling of K_real VNRs) and collect trajectory.

        Parameters
        ----------
        g_sub      : substrate DGLGraph **at start of window** (not mutated here)
        vnr_graphs : K_max-length list of VNR DGLGraphs; None slots are padding
        K_real     : actual number of VNRs (≤ K_max)
        deterministic : greedy if True, stochastic otherwise

        Returns
        -------
        dict with keys:
            'actions'   : List[int]          — chosen slot indices, length K_real
            'log_probs' : List[Tensor(1,)]   — log π(a_t | s_t)
            'values'    : List[Tensor(1,)]   — V(s_t) from critic
            'masks'     : List[Tensor bool]  — K_max-length mask before each step
        """
        g_sub_dev = g_sub.to(device)

        # Encode substrate (fixed; substrate state is snapshotted at window start)
        h_p = self.encode_substrate(g_sub_dev)

        # Encode all VNR slots
        keys, h_stack = self.encode_vnr_queue(vnr_graphs)

        # Initialise mask: True for real VNRs
        mask = torch.zeros(self.K_max, dtype=torch.bool, device=device)
        mask[:K_real] = True

        # Initialise GRU state from context
        h_queue_mean = h_stack[:K_real].mean(dim=0, keepdim=True)
        h_gru   = self.context_mlp(h_p, h_queue_mean)  # (1, gru_hidden)
        last_key = self.start_token                       # (1, gru_hidden)

        trajectory = dict(actions=[], log_probs=[], values=[], masks=[])

        for _ in range(K_real):
            action, lp, val, h_gru = self.step(
                h_p, keys, h_stack, mask, h_gru, last_key, deterministic
            )

            trajectory['actions'].append(action)
            trajectory['log_probs'].append(lp)
            trajectory['values'].append(val)
            trajectory['masks'].append(mask.clone())

            # Update state
            mask[action] = False
            last_key = keys[action:action + 1]   # (1, gru_hidden)

        return trajectory

    # ------------------------------------------------------------------
    # Re-evaluate stored actions for PPO update
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        g_sub:      dgl.DGLGraph,
        vnr_graphs: List[Optional[dgl.DGLGraph]],
        K_real:     int,
        actions:    List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-run episode with fixed `actions` under current policy.
        Needed for PPO's clipped surrogate objective (requires new log_probs).

        Returns
        -------
        log_probs : (K_real,)
        values    : (K_real,)
        entropies : (K_real,)
        """
        g_sub_dev = g_sub.to(device)
        h_p = self.encode_substrate(g_sub_dev)
        keys, h_stack = self.encode_vnr_queue(vnr_graphs)

        mask = torch.zeros(self.K_max, dtype=torch.bool, device=device)
        mask[:K_real] = True

        h_queue_mean = h_stack[:K_real].mean(dim=0, keepdim=True)
        h_gru   = self.context_mlp(h_p, h_queue_mean)
        last_key = self.start_token

        log_probs, values, entropies = [], [], []

        for t in range(K_real):
            logits, h_gru = self.pointer(keys, h_gru, last_key, mask)
            probs = F.softmax(logits, dim=0)
            dist  = torch.distributions.Categorical(probs)

            a = torch.tensor(actions[t], device=device)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

            active_h = h_stack[mask]
            h_queue_mean = active_h.mean(dim=0, keepdim=True) if active_h.numel() > 0 \
                           else torch.zeros(1, h_stack.shape[-1], device=device)
            values.append(self.critic(h_p, h_queue_mean))

            # Advance state
            mask[actions[t]] = False
            last_key = keys[actions[t]:actions[t] + 1]

        return (
            torch.stack(log_probs),   # (K_real,)
            torch.cat(values),         # (K_real,)
            torch.stack(entropies),    # (K_real,)
        )

    # ------------------------------------------------------------------
    # Inference-time ordering  (called by hpso_batch.py)
    # ------------------------------------------------------------------

    def forward_rl_order(
        self,
        substrate,
        vnr_list:       list,
        build_sub_fn:   Optional[Callable] = None,
        build_vnr_fn:   Optional[Callable] = None,
    ) -> List[int]:
        """
        Produce a deterministic processing order for `vnr_list` using the
        trained policy.  Called by hpso_batch_rl() at inference time.

        Parameters
        ----------
        substrate   : networkx substrate graph
        vnr_list    : list of networkx VNR graphs
        build_sub_fn: optional DGL builder; defaults to build_substrate_dgl
        build_vnr_fn: optional DGL builder; defaults to build_vnr_dgl

        Returns
        -------
        order : List[int]  — indices into vnr_list in chosen scheduling order
                             (length ≤ K_max; any remainder appended as-is)
        """
        _build_sub = build_sub_fn or build_substrate_dgl
        _build_vnr = build_vnr_fn or build_vnr_dgl

        K_real = min(len(vnr_list), self.K_max)

        g_sub = _build_sub(substrate).to(device)
        vnr_dgls = [_build_vnr(v) for v in vnr_list[:K_real]]
        padded   = vnr_dgls + [None] * (self.K_max - K_real)

        self.eval()
        with torch.no_grad():
            traj = self.rollout(g_sub, padded, K_real, deterministic=True)

        order     = traj['actions']    # list of K_real ints
        remaining = [i for i in range(len(vnr_list)) if i not in set(order)]
        return order + remaining
