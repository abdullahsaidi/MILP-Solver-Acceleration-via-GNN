"""
BACKDOOR/GAT.py
===============
BACKDOOR task model: SharedEncoder + per-variable scoring head.

CHANGE LOG (vs. submitted version)
------------------------------------
1. BACKDOOR_GATPolicy_multitask removed — was defined but never
   instantiated or used anywhere in the codebase (reviewer comment).
   The multitask trainer (MULTITASK/train_multitask.py) uses
   MultiTaskModel which manages its own backdoor head directly.

2. GATPolicy_multitask alias removed (pointed to the now-deleted class).

Documented deviations from the paper
--------------------------------------
* The scoring head is a 2-layer MLP (EMB → EMB → 1) rather than the
  single linear layer implied by the paper's notation, following the
  original repo implementation.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from shared.shared_encoder import SharedEncoder, EMB


class BACKDOOR_GATPolicy(nn.Module):
    """SharedEncoder + per-variable backdoor scoring head."""

    def __init__(self):
        super().__init__()
        self.encoder      = SharedEncoder()
        self.scoring_head = nn.Sequential(
            nn.Linear(EMB, EMB), nn.ReLU(), nn.Linear(EMB, 1, bias=False),
        )
        for p in self.parameters():
            if p.dim() == 2: init.orthogonal_(p)
            else:            init.normal_(p)

    def freeze_normalization(self) -> bool:
        return self.encoder.freeze_normalization()

    def forward(self, con_feats, edge_idx, edge_feats, var_feats):
        h_var, _ = self.encoder(var_feats, con_feats, edge_idx, edge_feats)
        return self.scoring_head(h_var).squeeze(-1)


# Single alias — multitask class removed (was never instantiated)
GATPolicy = BACKDOOR_GATPolicy
