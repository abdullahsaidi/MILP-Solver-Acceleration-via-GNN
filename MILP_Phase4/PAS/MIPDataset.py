import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch_geometric.data import Data

from shared.scip_features import (
    VAR_FEAT_DIM, CON_FEAT_DIM,
    VAR_FEATURE_NAMES, CON_FEATURE_NAMES,
    extract_bipartite_graph,
)


class PAS_Dataset(Dataset):
    TASK_ID = {"CA": 0, "MIS": 1, "MVC": 2}

    def __init__(self, problem: str, split: str, dataset_dir: str = "dataset"):
        self.problem = problem
        self.task_id = self.TASK_ID[problem]
        data_dir     = os.path.join(dataset_dir, problem, split)
        self.files   = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {data_dir}. Run PAS/collect_data.py first.")
        print(f"  [PAS/{problem}/{split}] {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz = np.load(self.files[idx])
        return Data(
            x            = torch.tensor(npz["var_features"],  dtype=torch.float32),
            con_features = torch.tensor(npz["con_features"],  dtype=torch.float32),
            edge_index   = torch.tensor(npz["edge_indices"],  dtype=torch.long),
            edge_attr    = torch.tensor(npz["edge_features"], dtype=torch.float32),
            y            = torch.tensor(npz["labels"],        dtype=torch.float32),
            obj_sense    = torch.tensor([int(npz["obj_sense"][0])], dtype=torch.long),
            task_id      = torch.tensor([self.task_id], dtype=torch.long),
        )


class PAS_MultiTaskDataset(Dataset):
    def __init__(self, problems, split, dataset_dir="dataset"):
        self.datasets = [PAS_Dataset(p, split, dataset_dir) for p in problems]
        self.concat   = ConcatDataset(self.datasets)
        print(f"[PAS/MultiTask/{split}] {len(self.concat)} total samples from {problems}")

    def __len__(self):
        return len(self.concat)

    def __getitem__(self, idx):
        return self.concat[idx]


def PAS_collate(data_list):
    var_offset = con_offset = 0
    out_vf, out_cf, out_ei, out_ea, out_lbl, out_obj, out_tid = [], [], [], [], [], [], []
    var_batch, con_batch = [], []

    for b_idx, d in enumerate(data_list):
        nv = d.x.size(0)
        nc = d.con_features.size(0)
        out_vf.append(d.x)
        out_cf.append(d.con_features)
        ei = d.edge_index.clone()
        ei[0] += con_offset
        ei[1] += var_offset
        out_ei.append(ei)
        out_ea.append(d.edge_attr)
        out_lbl.append(d.y)
        out_obj.append(d.obj_sense)
        out_tid.append(d.task_id)
        var_batch.extend([b_idx] * nv)
        con_batch.extend([b_idx] * nc)
        var_offset += nv
        con_offset += nc

    return {
        "var_features": torch.cat(out_vf,  dim=0),
        "con_features": torch.cat(out_cf,  dim=0),
        "edge_index":   torch.cat(out_ei,  dim=1),
        "edge_attr":    torch.cat(out_ea,  dim=0),
        "labels":       torch.cat(out_lbl, dim=0),
        "obj_sense":    torch.cat(out_obj, dim=0),
        "task_id":      torch.cat(out_tid, dim=0),
        "var_batch":    torch.tensor(var_batch, dtype=torch.long),
        "con_batch":    torch.tensor(con_batch, dtype=torch.long),
        "batch_size":   len(data_list),
    }


def build_PAS_loader(dataset, batch_size=16, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=PAS_collate)
