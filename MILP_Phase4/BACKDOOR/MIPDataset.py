import os
import numpy as np
import torch
import torch_geometric as tg
import pandas as pd
from ast import literal_eval

from shared.scip_features import (
    VAR_FEAT_DIM, CON_FEAT_DIM,
    VAR_FEATURE_NAMES, CON_FEATURE_NAMES,
    extract_bipartite_graph,
)

variable_feature_names   = VAR_FEATURE_NAMES
constraint_feature_names = CON_FEATURE_NAMES
num_var_features         = VAR_FEAT_DIM
num_con_features         = CON_FEAT_DIM


class BipartiteNodeData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_cons_to_vars":
            return torch.tensor([[self.x_cons.size(0)], [self.x_vars.size(0)]])
        if key == "edge_index_var_to_cons":
            return torch.tensor([[self.x_vars.size(0)], [self.x_cons.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


def compute_mip_representation(mip_path: str) -> BipartiteNodeData:
    vf, cf, ei, ef, _, obj_sense = extract_bipartite_graph(str(mip_path))
    if vf is None:
        raise ValueError(f"Feature extraction failed for {mip_path}")
    ei_t = torch.tensor(ei, dtype=torch.long)
    return BipartiteNodeData(
        x_vars                  = torch.tensor(vf, dtype=torch.float32),
        x_var_names             = VAR_FEATURE_NAMES,
        x_cons                  = torch.tensor(cf, dtype=torch.float32),
        x_con_names             = CON_FEATURE_NAMES,
        edge_index_cons_to_vars = ei_t,
        edge_index_var_to_cons  = torch.stack([ei_t[1], ei_t[0]], dim=0),
        edge_attr               = torch.tensor(ef, dtype=torch.float32),
        obj_sense               = obj_sense,
        discrete_var_mask       = torch.ones(len(vf), dtype=torch.bool),
    )


def _load_backdoor_labels(graph, csv_path, n_samples=5):
    df     = pd.read_csv(csv_path, index_col=0)
    n_vars = graph.x_vars.shape[0]
    graph.pos_sample = []
    graph.neg_sample = []
    for rank in range(min(n_samples, len(df))):
        pos_vec = np.zeros(n_vars)
        for v in literal_eval(df.iloc[rank]["backdoor_list"]):
            pos_vec[int(v)] = 1
        graph.pos_sample.append(pos_vec)
        neg_vec = np.zeros(n_vars)
        for v in literal_eval(df.iloc[-rank - 1]["backdoor_list"]):
            neg_vec[int(v)] = 1
        graph.neg_sample.append(neg_vec)
    return graph


class BACKDOOR_GraphDataset(tg.data.InMemoryDataset):
    def __init__(self, sample_pairs):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_pairs = sample_pairs
        self.data = [self._load(i) for i in range(len(sample_pairs))]

    def len(self):    return len(self.data)
    def get(self, i): return self.data[i]

    def _load(self, idx):
        ins_path, res_path = self.sample_pairs[idx]
        print(f"  [BACKDOOR] loading {os.path.basename(ins_path)}")
        return _load_backdoor_labels(compute_mip_representation(ins_path), res_path)


class BACKDOOR_GraphDataset_filter(tg.data.InMemoryDataset):
    """Filtered variant: only keep pos samples with runtime <= 1s and neg >= 1s."""
    def __init__(self, sample_pairs):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_pairs = sample_pairs
        self.data = [self._load(i) for i in range(len(sample_pairs))]

    def len(self):    return len(self.data)
    def get(self, i): return self.data[i]

    def _load(self, idx):
        ins_path, res_path = self.sample_pairs[idx]
        df    = pd.read_csv(res_path, index_col=0)
        graph = compute_mip_representation(ins_path)
        n_vars = graph.x_vars.shape[0]
        graph.pos_sample = []
        graph.neg_sample = []
        for rank in range(min(5, len(df))):
            pos_vec = np.zeros(n_vars)
            for v in literal_eval(df.iloc[rank]["backdoor_list"]):
                pos_vec[int(v)] = 1
            if df.iloc[rank]["run_time"] <= 1:
                graph.pos_sample.append(pos_vec)
            neg_vec = np.zeros(n_vars)
            for v in literal_eval(df.iloc[-rank - 1]["backdoor_list"]):
                neg_vec[int(v)] = 1
            if df.iloc[-rank - 1]["run_time"] >= 1:
                graph.neg_sample.append(neg_vec)
        return graph
