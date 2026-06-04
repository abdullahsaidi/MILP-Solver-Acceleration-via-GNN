import torch
import torch.nn as nn
from shared.shared_encoder import SharedEncoder, EMB


class PaSHead(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EMB, EMB // 2), nn.LayerNorm(EMB // 2), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(EMB // 2, 1),
        )

    def forward(self, h_var):
        return self.mlp(h_var).squeeze(-1)


class PaSModel(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.encoder = SharedEncoder()
        self.head    = PaSHead(dropout=dropout)

    def forward(self, var_features, con_features, edge_index, edge_attr):
        h_var, _ = self.encoder(var_features, con_features, edge_index, edge_attr)
        return self.head(h_var)

    def predict_probs(self, var_features, con_features, edge_index, edge_attr):
        return torch.sigmoid(self.forward(var_features, con_features, edge_index, edge_attr))

    def freeze_normalization(self):
        return self.encoder.freeze_normalization()


def build_PAS_model(dropout: float = 0.1) -> PaSModel:
    return PaSModel(dropout=dropout)
