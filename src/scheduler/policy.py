"""
src/scheduler/policy.py
=======================
SB3-compatible Actor-Critic policy wrapping VNRScheduler.

This module provides ``GNNActorCriticPolicy``, which plugs the GCN-based
scorer into Stable-Baselines3 PPO as a *custom policy*.

Architecture (see network_encoder_rl.md §7):
  - Actor: VNRScheduler → scores [B] → Categorical(logits=scores)
  - Critic: linear head applied to the substrate embedding h_s [128]

Note on variable action space
------------------------------
Each episode step has a different number of remaining VNRs, so the policy
receives the observation as a Python dict (not a fixed numpy array). The
policy overrides SB3's ``predict`` to work with this dict directly via the
custom REINFORCE loop. For proper SB3 integration (PPO), see the comments
in the class docstring.

This module is *optional*: in Phase 1 the REINFORCE training loop does not
use SB3 at all. Only Phase 2 (PPO via SB3) needs this file.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.distributions import Categorical


class GNNActorCritic(nn.Module):
    """
    Thin actor-critic wrapper around VNRScheduler.

    This is NOT a full SB3 BasePolicy subclass (that requires significant
    boilerplate to handle the variable action space). Instead, it is a
    standalone nn.Module used directly by the custom PPO training loop in
    ``src/training/train_ppo.py``.

    Parameters
    ----------
    scheduler : VNRScheduler
        The GCN scoring network (shared between actor and critic).
    substrate_emb_dim : int
        Dimension of the substrate embedding (used as critic input). Default 128.
    """

    def __init__(self, scheduler, substrate_emb_dim: int = 128):
        super().__init__()
        self.scheduler      = scheduler
        self.value_head     = nn.Linear(substrate_emb_dim, 1)
        self._sub_emb_dim   = substrate_emb_dim

    def forward(self, obs: dict) -> Tuple[Categorical, torch.Tensor]:
        """
        Parameters
        ----------
        obs : dict
            "substrate" → PyG Data (single graph)
            "vnr_list"  → list of PyG Data  (remaining VNRs)

        Returns
        -------
        dist  : Categorical over remaining VNRs
        value : Tensor  shape [1]  (critic estimate)
        """
        substrate_data = obs["substrate"]
        vnr_data_list  = obs["vnr_list"]

        # Actor: score each remaining VNR
        scores = self.scheduler(substrate_data, vnr_data_list)   # [B]
        dist   = Categorical(logits=scores)

        # Critic: value of the current substrate state
        h_s   = self.scheduler.substrate_encoder(substrate_data) # [1, 128]
        value = self.value_head(h_s).squeeze(-1)                 # [1]

        return dist, value

    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: dict,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action (or evaluate a given action) and compute log-prob,
        entropy, and value.

        Signature follows CleanRL-style PPO helpers.

        Returns
        -------
        action, log_prob, entropy, value
        """
        dist, value = self.forward(obs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: dict) -> torch.Tensor:
        """Critic value only."""
        substrate_data = obs["substrate"]
        h_s   = self.scheduler.substrate_encoder(substrate_data)
        return self.value_head(h_s).squeeze(-1)
