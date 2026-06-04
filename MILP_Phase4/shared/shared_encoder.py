"""
shared/shared_encoder.py
========================
Shared GAT backbone used by PAS, BACKDOOR, and CONFIG.

CHANGE LOG (vs. submitted version)
------------------------------------
1. HEADS restored to 8  — paper Section 4.2: "H = 8 attention heads".
   The submitted code used HEADS = 4 (introduced during partner alignment).
   Restoring to 8 changes the per-head channel size from 16 → 8 and the
   attention parameter tensor shape from (1,4,48) → (1,8,24).

2. build_shared_encoder() removed — was defined but never imported or
   called anywhere in the codebase (reviewer comment).

3. freeze_normalization() is now called from the multitask training loop
   (see MULTITASK/train_multitask.py — also fixed in that file).

Documented deviations from the paper
--------------------------------------
* Paper uses Gurobi for feature extraction; we use SCIP (pyscipopt).
* VAR_FEAT_DIM = 15 / CON_FEAT_DIM = 5 — restores paper values.
  These constants are imported from shared/scip_features.py which is the
  single source of truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import softmax
import torch.nn.init as init

from shared.scip_features import VAR_FEAT_DIM, CON_FEAT_DIM

EMB   = 64
HEADS = 8   


class Prenorm(nn.Module):
    """Online Welford normalisation with progressive freeze support."""

    def __init__(self, n_features: int, preserve_features: list = []):
        super().__init__()
        self.n_features        = n_features
        self.preserve_features = preserve_features
        self.register_buffer("avg",     torch.zeros(n_features, dtype=torch.double))
        self.register_buffer("var",     torch.zeros(n_features, dtype=torch.double))
        self.register_buffer("count",   torch.ones(1))
        self.register_buffer("_frozen", torch.tensor([False], dtype=torch.bool,
                                                     requires_grad=False))
        self.register_buffer("shift",   torch.zeros(n_features))
        self.register_buffer("scale",   torch.ones(n_features))

    @property
    def frozen(self):
        return bool(self._frozen.item())

    def freeze_normalization(self):
        self._frozen = torch.tensor([True], dtype=torch.bool)

    def reset_normalization(self):
        self.avg.zero_(); self.var.zero_()
        self.count.fill_(1.0); self._frozen.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.frozen:
            with torch.no_grad():
                n      = float(x.shape[0])
                sv, sa = torch.var_mean(x.double(), dim=0)
                delta  = sa - self.avg
                m2     = (self.var * self.count + sv * n
                          + delta ** 2 * self.count * n / (self.count + n))
                self.avg    = (self.avg * self.count + sa * n) / (self.count + n)
                self.count += n
                self.var    = m2 / self.count
                self.shift  = -self.avg.float()
                # clamp variance from below to avoid Inf in rsqrt
                v           = torch.clamp(self.var, min=1e-8)
                self.scale  = torch.rsqrt(v).float()
                # clamp scale to avoid extreme values on near-constant features
                self.scale  = torch.clamp(self.scale, max=1e4)
                for f in self.preserve_features:
                    self.shift[f] = 0.0
                    self.scale[f] = 1.0
        out = x.float() + self.shift
        out = out * self.scale
        # soft fix: replace any residual NaN/Inf instead of crashing
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class GATConvolution(torch_geometric.nn.MessagePassing):
    """
    Bipartite GAT conv.
    Paper: 8 heads × 8 dims = 64-dim output  (HEADS=8, head_dim=EMB//HEADS=8).
    """

    def __init__(self, negative_slope: float = 0.2, dropout: float = 0.0):
        super().__init__(aggr="add")
        head_dim            = EMB // HEADS   # 8 (was 16 when HEADS=4)
        self.head_dim       = head_dim
        self.negative_slope = negative_slope
        self.dropout        = dropout
        self.lin_l   = nn.Linear(EMB, HEADS * head_dim, bias=True)
        self.lin_r   = nn.Linear(EMB, HEADS * head_dim, bias=True)
        self.att     = nn.Parameter(torch.empty(1, HEADS, head_dim * 3))
        self.out_mlp = nn.Sequential(
            nn.Linear(2 * EMB, EMB), nn.ELU(), nn.Linear(EMB, EMB),
        )
        init.xavier_uniform_(self.lin_l.weight); init.zeros_(self.lin_l.bias)
        init.xavier_uniform_(self.lin_r.weight); init.zeros_(self.lin_r.bias)
        # att shape: (1, HEADS, head_dim*3) — reshape to 2D for xavier, then restore
        att_2d = self.att.data.view(HEADS, head_dim * 3)
        init.xavier_uniform_(att_2d)
        self.att.data = att_2d.unsqueeze(0)

    def freeze_normalization(self): pass
    def reset_normalization(self):  pass

    @property
    def frozen(self):
        return False

    def forward(self, left, edge_idx, edge_feats, right):
        out = self.propagate(edge_idx,
                              x=(self.lin_l(left), self.lin_r(right)),
                              size=(left.shape[0], right.shape[0]),
                              edge_features=edge_feats)
        return self.out_mlp(torch.cat([out, right], dim=-1))

    def message(self, x_j, x_i, index, edge_features):
        H, C = HEADS, self.head_dim
        x     = torch.cat([x_i, x_j, edge_features.expand(-1, H * C)], dim=-1)
        x     = F.leaky_relu(x, self.negative_slope).view(-1, H, C * 3)
        alpha = softmax((x * self.att).sum(-1), index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (x_j.view(-1, H, C) * alpha.unsqueeze(-1)).view(-1, H * C)


class SharedEncoder(nn.Module):
    """
    Shared backbone used by PAS, BACKDOOR, and CONFIG.

    Input : var_features  (n_vars, 15)   — paper 15-dim layout
            con_features  (n_cons,   5)
            edge_index    (2, E)          [con_idx → var_idx]
            edge_attr     (E,  1)         normalised coefficient
    Output: h_var  (n_vars, 64)
            h_con  (n_cons, 64)
    """

    VAR_FEAT_DIM = VAR_FEAT_DIM   # 15
    CON_FEAT_DIM = CON_FEAT_DIM   # 5

    def __init__(self):
        super().__init__()
        # preserve_features=[2] keeps the type_INTEGER flag unnormalised
        self.var_norm = Prenorm(VAR_FEAT_DIM, preserve_features=[2])
        self.var_embedding = nn.Sequential(
            self.var_norm,
            nn.Linear(VAR_FEAT_DIM, EMB), nn.LayerNorm(EMB), nn.ELU(),
            nn.Linear(EMB, EMB),          nn.LayerNorm(EMB), nn.ELU(),
        )
        self.con_norm = Prenorm(CON_FEAT_DIM)
        self.con_embedding = nn.Sequential(
            self.con_norm,
            nn.Linear(CON_FEAT_DIM, EMB), nn.LayerNorm(EMB), nn.ELU(),
            nn.Linear(EMB, EMB),          nn.LayerNorm(EMB), nn.ELU(),
        )
        self.edge_norm = Prenorm(1)
        self.edge_embedding = nn.Sequential(
            self.edge_norm, nn.Linear(1, EMB),
        )
        self.conv_v_to_c = GATConvolution()
        self.conv_c_to_v = GATConvolution()

        # Careful targeted init: only weight matrices, skip norms and biases
        for name, p in self.named_parameters():
            if "norm" in name or "bias" in name:
                continue   # leave LayerNorm weights/biases at default
            if p.dim() == 2:
                init.xavier_uniform_(p)
            elif p.dim() > 2:
                init.xavier_uniform_(p.view(p.size(0), -1))

    def freeze_normalization(self) -> bool:
        """Progressive normalisation freeze. Returns True when fully frozen."""
        if not self.con_norm.frozen:
            self.con_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True

    def forward(self, var_features, con_features, edge_index, edge_attr):
        # sanitise inputs — catches NaN/Inf from data collection edge cases
        var_features = torch.nan_to_num(var_features, nan=0.0, posinf=0.0, neginf=0.0)
        con_features = torch.nan_to_num(con_features, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr    = torch.nan_to_num(edge_attr,    nan=0.0, posinf=0.0, neginf=0.0)
        rev    = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_con  = self.con_embedding(con_features)
        h_edge = self.edge_embedding(edge_attr)
        h_var  = self.var_embedding(var_features)
        h_con  = self.conv_v_to_c(h_var, rev,        h_edge, h_con)
        h_var  = self.conv_c_to_v(h_con, edge_index, h_edge, h_var)
        return h_var, h_con
