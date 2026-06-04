"""
CONFIG/GAT.py
=============
CONFIG task model: SharedEncoder + graph-level pooling + 33-dim output head.

CHANGE LOG (vs. submitted version)
------------------------------------
1. HEADS restored to 8 via shared_encoder.py (was 4 in submitted code).
   This file imports SharedEncoder which now uses HEADS=8.
   No change needed here directly.

2. The 33-dim output is correct — see CONFIG/MIPDataset.py header for
   full explanation of the one-hot encoding that maps 17 SCIP parameters
   to 33 dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_scatter import scatter_mean

from shared.shared_encoder import SharedEncoder, EMB
from CONFIG.MIPDataset import CONFIG_OUT_DIM, PARAM_SEQUENCE, CATEGORICAL_PARAMS

# Positions in the 33-dim output that form categorical groups -> apply softmax
_SOFTMAX_GROUPS = [
    [1, 2, 3],
    [8, 9, 10],
    [14, 15, 16, 17, 18, 19, 20],
    [22, 23, 24, 25, 26, 27, 28],
]


class CONFIG_GATPolicy(nn.Module):
    """SharedEncoder + graph-level pooling + 33-dim Config head.

    forward() takes batch tensors and returns (B, 33) parameter vectors.
    """

    def __init__(self):
        super().__init__()
        self.encoder       = SharedEncoder()
        self.output_module = nn.Sequential(
            nn.Linear(EMB * 2, EMB), nn.ELU(),
            nn.Linear(EMB, CONFIG_OUT_DIM, bias=False),
        )
        for t in self.parameters():
            if t.dim() == 2: init.orthogonal_(t)
            else:            init.normal_(t)

    def freeze_normalization(self) -> bool:
        return self.encoder.freeze_normalization()

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, variable_features_batch, constraint_features_batch):
        h_var, h_con = self.encoder(variable_features, constraint_features,
                                     edge_indices, edge_features)
        v_pool = scatter_mean(h_var, variable_features_batch,   dim=0)
        c_pool = scatter_mean(h_con, constraint_features_batch, dim=0)
        raw    = self.output_module(torch.cat([v_pool, c_pool], dim=-1))
        out    = raw.clone()
        for group in _SOFTMAX_GROUPS:
            out[..., group] = torch.softmax(raw[..., group], dim=-1)
        return out

    def get_backbone_params(self):
        return list(self.encoder.parameters())

    def get_head_params(self):
        return list(self.output_module.parameters())


GATPolicy = CONFIG_GATPolicy
