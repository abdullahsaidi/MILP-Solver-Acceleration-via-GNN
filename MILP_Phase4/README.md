# MILP Multitask — Refactored Codebase

**CPAIOR 2025: Multi-task Representation Learning for Mixed Integer Linear Programming**

> All Gurobi / Ecole dependencies removed. Everything runs on **SCIP via pySCIPOpt**.

---

## 1. Project Overview

This project trains a shared Graph Attention Network (GAT) encoder on three complementary MILP tasks simultaneously. The shared encoder learns richer MIP representations than any single-task model, and each task benefits from the others.

| Module | Task | What it learns |
|--------|------|----------------|
| **PAS** | Predict-and-Search | Marginal probabilities p(x_i = 1) per variable. High-confidence variables are fixed to reduce the search space before calling SCIP. |
| **BACKDOOR** | Backdoor detection | Ranks variables by likelihood of being in a backdoor set — a small subset that, once fixed, makes the rest fast to solve. |
| **CONFIG** | Solver configuration | Predicts the best SCIP parameter configuration (33-dim vector) for an instance, replacing expensive per-instance SMAC runs. |

All three tasks share a single **SharedEncoder** backbone. Task-specific heads attach to that backbone.

---

## 2. Architecture

```
Input: Bipartite Graph (MILP Instance)
  var_features : (n_vars, 19)
  con_features : (n_cons,  5)
  edge_index   : (2, n_edges)    [con_idx -> var_idx]
  edge_attr    : (n_edges, 1)    normalised coefficients
         |
         v
 +------------------------------------+
 |         SharedEncoder             |
 |  Prenorm -> Linear -> ELU x 2    |
 |  GATConvolution (vars -> cons)    |
 |  GATConvolution (cons -> vars)    |
 |  Output: h_var (n_vars, 64)       |
 |          h_con (n_cons, 64)       |
 +------+----------+-----------------+
        |          |          |
        v          v          v
 +----------+ +----------+ +------------------+
 | PaS Head | | Backdoor | |  Config Head     |
 | MLP -> 1 | | MLP -> 1 | | pool+MLP -> 33  |
 | per-var  | | per-var  | | per-graph        |
 +----------+ +----------+ +------------------+
```

**Feature dimensions (do not change):**
- Variable features: **19-dim**
- Constraint features: **5-dim**
- Edge features: **1-dim** (normalised coefficient)
- Hidden embedding: **64-dim** (EMB)
- Attention heads: **4** (HEADS)

`GATConvolution` uses additive attention with separate source/target/edge projections. `Prenorm` performs online Welford mean/variance normalisation that is progressively frozen during training.

---

## 3. Environment Setup

### 3.1 Install SCIP

SCIP must be installed before pySCIPOpt. Easiest route via conda:

```bash
conda install -c conda-forge scip pyscipopt
```

Or manually:
```bash
# 1. Download from https://www.scipopt.org/index.php#download
# 2. Build with cmake, then:
export SCIPOPTDIR=/path/to/scip
pip install pyscipopt
```

### 3.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

Key packages: `pyscipopt`, `torch`, `torch-geometric`, `torch-scatter`, `pytorch-metric-learning`.

### 3.3 Verify SCIP works

```python
import pyscipopt
m = pyscipopt.Model()
m.hideOutput(True)
m.readProblem("instances/MIS/train/MIS_train_0000.lp")
m.optimize()
print("SCIP OK, status:", m.getStatus())
```

---

## 4. Data Generation Pipeline

All data generation uses SCIP exclusively. No Gurobi, no Ecole.

### Step 1 — Generate MILP instances

```bash
python generate_instances.py --base_dir instances --problems CA MIS MVC
```

Outputs `.lp` files for CA (combinatorial auction), MIS (maximum independent set), and MVC (minimum vertex cover):

```
instances/
  CA/train/  CA_train_0000.lp ... CA_train_0199.lp   (200 instances)
  CA/test/   CA_test_0000.lp  ... CA_test_0099.lp    (100 instances)
  MIS/train/ ...
  MIS/test/  ...
  MVC/train/ ...
  MVC/test/  ...
```

Estimated time: **< 1 minute**

---

### Step 2 — Collect PaS training data

SCIP solves each instance and records energy-weighted marginal probabilities as labels.

```bash
# All problems, all splits, parallelised
python -m PAS.collect_data --problem all --split all --n_jobs 4

# Or one at a time
python -m PAS.collect_data --problem CA  --split train
python -m PAS.collect_data --problem MIS --split train --n_jobs 4
python -m PAS.collect_data --problem MVC --split train --n_jobs 4
```

Outputs one `.npz` per instance containing `var_features`, `con_features`, `edge_indices`, `edge_features`, `labels`, `obj_sense`:

```
dataset/
  CA/train/  CA_train_0000.npz ...
  CA/test/   ...
  MIS/...    MVC/...
```

Estimated time: **2–4 hours** for all 900 instances at `--n_jobs 4`.

---

### Step 3 — Collect Backdoor training data

The Backdoor task requires a CSV file per instance recording which variable subsets were tested as backdoors and how long SCIP took when those variables were fixed. These are produced by running a separate randomised backdoor search (see the paper supplementary for the search procedure).

**What the search does:** for each instance, it randomly samples small variable subsets, fixes them to their optimal values, and times how long SCIP takes to solve the rest. Fast subsets are positive examples; slow ones are negative.

**Expected directory layout:**
```
data/backdoor_results/
  CA_train_0000.lp          <- filename matches the instance name
  CA_train_0001.lp
  ...
```

**Expected CSV format** (each file named identically to its instance):
```
,backdoor_list,run_time
0,"[3, 17, 42]",0.23
1,"[8, 11, 55]",0.41
2,"[1, 99, 12]",3.87
...
```

Each row is one tested backdoor. The training code reads the top 5 rows as positive examples (fastest) and the bottom 5 as negative examples (slowest). The CSV must have at least 10 rows.

The script `backdoor_search.py` is included in the project root. Run it for a single instance:

```bash
python backdoor_search.py --instance instances/CA/train/CA_train_0000.lp --output data/backdoor_results --n_trials 50 --backdoor_size 8
```

Or process an entire directory in parallel:

```bash
python backdoor_search.py --instance_dir instances/CA/train --output data/backdoor_results --n_trials 50 --backdoor_size 8 --n_jobs 4
```

Repeat for MIS and MVC:

```bash
python backdoor_search.py --instance_dir instances/MIS/train --output data/backdoor_results --n_trials 50 --backdoor_size 8 --n_jobs 4
python backdoor_search.py --instance_dir instances/MVC/train --output data/backdoor_results --n_trials 50 --backdoor_size 8 --n_jobs 4
```

Each CSV is saved to `--output` with the same filename as its instance (e.g. `CA_train_0000.lp`). You need at least 10 feasible trials per instance for training. If you see the warning `only N feasible trials`, increase `--n_trials`.

Once you have result CSVs, train the Backdoor model (see §5, Step 3).

---

### Step 4 — Collect Config training data

The Config task requires a CSV per instance recording different SCIP parameter configurations and their solve cost. These are produced by running SMAC (Sequential Model-based Algorithm Configuration) on each instance.

**What SMAC does:** it explores the SCIP parameter space, running SCIP with different configurations on the instance and recording solve time. After many trials it produces a ranked list of configurations.

**Expected directory layout:**
```
datasets/
  CA/
    CA_train_0000.lp          <- filename matches the instance (no extension needed)
    CA_train_0001.lp
    ...
```

**Expected CSV format** (named `<instance_filename>.csv`):
```
,cost,params
0,0.45,"{'branching/scorefunc': 'q', 'branching/lpgainnormalize': 's', 'lp/pricing': 'l', ...}"
1,0.61,"{'branching/scorefunc': 's', 'branching/lpgainnormalize': 'd', 'lp/pricing': 'a', ...}"
2,1.20,"{'branching/scorefunc': 'p', ...}"
...
```

`cost` is solve time in seconds (lower = better). The CSV is sorted ascending by cost so row 0 is the best config. The training code takes the top 5 as positive examples and bottom 5 as negative examples.

The script `run_smac.py` is included in the project root. It runs a random search over the SCIP parameter space — no SMAC library needed.

Single instance:
```bash
python run_smac.py --instance instances/CA/train/CA_train_0000.lp --output datasets/CA --n_trials 50 --time_limit 300
```

Whole directory in parallel:
```bash
python run_smac.py --instance_dir instances/CA/train --output datasets/CA --n_trials 50 --time_limit 300 --n_jobs 4
```

Repeat for MIS and MVC:
```bash
python run_smac.py --instance_dir instances/MIS/train --output datasets/MIS --n_trials 50 --time_limit 300 --n_jobs 4
python run_smac.py --instance_dir instances/MVC/train --output datasets/MVC --n_trials 50 --time_limit 300 --n_jobs 4
```

Each CSV is named `<instance_name>.csv` (without the `.lp`) and saved in `--output`. You need at least 10 successful trials per instance. If you see the warning `only N successful trials`, increase `--n_trials` or `--time_limit`.

The 17 SCIP parameters tuned are defined in `run_smac.py` under `PARAM_SPACE` and must match `PARAM_SEQUENCE` in `CONFIG/MIPDataset.py`.

Once you have result CSVs, train the Config model (see §5, Step 3).

---

## 5. Training Pipeline

### Step 1 — Generate instances
See §4 Step 1.

### Step 2 — Train PaS (initialises the shared encoder)

```bash
# Multi-task (recommended — encoder sees all three problem types)
python -m PAS.train --problems CA MIS MVC --task_name pas_multitask --dataset_dir dataset --epochs 100 --batch_size 16 --lr 1e-3

# Single-problem example
python -m PAS.train --problems MIS --task_name pas_mis --dataset_dir dataset
```

Checkpoint saved to `pretrain/pas_multitask/model_best.pt`.

Estimated time: **~45 min CPU** | **~8 min GPU** (100 epochs, 600 train samples)

### Step 3 — Single-task fine-tuning (PAS / BACKDOOR / CONFIG)

**PAS** — already done in Step 2.

**BACKDOOR** (requires data from §4 Step 3):
```bash
python -m BACKDOOR.train --taskName      BACKDOOR_IP --instance_dir  instances --result_dir    data/backdoor_results --pretrainModel pretrain/pas_multitask/model_best.pt --freeze 1
```

**CONFIG** (requires data from §4 Step 4):
```bash
python -m CONFIG.train --taskName        config_CA --instance_dir    instances/CA/train --result_dir      datasets/CA --pretrain_path   pretrain/pas_multitask/model_best.pt --freeze_backbone
```

Estimated time per task: **~30–120 min CPU** depending on dataset size.

### Step 4 — Joint multi-task training

```bash
python -m MULTITASK.train_multitask --task_name        multitask_v1 --pas_dataset_dir  dataset --pas_problems     CA MIS MVC --bd_instance_dir  instances --bd_result_dir    data/backdoor_results --cfg_instance_dir instances/CA/train --cfg_result_dir   datasets/CA --pretrain_model   pretrain/pas_multitask/model_best.pt --epochs 100 --w_pas 0.5 --w_backdoor 0.25 --w_config 0.25
```

If Backdoor or Config data are not ready yet, just omit those flags — the trainer will use only the available tasks:

```bash
python -m MULTITASK.train_multitask --pas_dataset_dir dataset --task_name multitask_pas_only
```

Estimated time: **~1–3 hours GPU** | **~6–12 hours CPU**

---

## 6. File Structure

```
milp_multitask/
├── generate_instances.py         # CA, MIS, MVC instance generator (.lp files)
├── requirements.txt
├── README.md
│
├── shared/                       # Single source of truth imported by all tasks
│   ├── __init__.py
│   ├── scip_features.py          # 19-dim var + 5-dim con extraction via SCIP
│   └── shared_encoder.py         # SharedEncoder backbone (Prenorm + GATConvolution)
│
├── PAS/
│   ├── collect_data.py           # SCIP solve + Boltzmann-weighted marginal labels
│   ├── MIPDataset.py             # Loads .npz files into PyG Data objects
│   ├── model.py                  # PaSModel = SharedEncoder + PaSHead
│   ├── train.py                  # BCE training, single-task or multi-problem
│   └── evaluate.py               # Trust-region search eval via SCIP
│
├── BACKDOOR/
│   ├── MIPDataset.py             # Loads (instance, backdoor CSV) pairs
│   ├── GAT.py                    # BACKDOOR_GATPolicy = SharedEncoder + scoring head
│   ├── train.py                  # InfoNCE contrastive training
│   └── evaluate.py               # SCIP branching priority injection + timing
│
├── CONFIG/
│   ├── MIPDataset.py             # Loads (instance, SMAC CSV) pairs, 33-dim param encoding
│   ├── GAT.py                    # CONFIG_GATPolicy = SharedEncoder + pooling + 33-dim head
│   ├── train.py                  # InfoNCE training on SCIP parameter configurations
│   └── evaluate.py               # Decodes 33-dim output -> SCIP params, solves, records primal log
│
└── MULTITASK/
    └── train_multitask.py        # MultiTaskModel + joint training loop (PaS + Backdoor + Config)
```

---

## 7. Expected Outputs per Stage

| Stage | Output |
|-------|--------|
| `generate_instances.py` | `instances/{CA,MIS,MVC}/{train,test}/*.lp` |
| `PAS/collect_data.py` | `dataset/{CA,MIS,MVC}/{train,test}/*.npz` |
| `PAS/train.py` | `pretrain/{task_name}/model_best.pt`, `model_final.pt`, `training_log.csv` |
| `BACKDOOR/train.py` | `pretrain/{taskName}/model_best.pth`, `model_last.pth` |
| `CONFIG/train.py` | `pretrain/{taskName}/model_best.pth`, `model_last.pth` |
| `MULTITASK/train_multitask.py` | `pretrain/{task_name}/model_best.pt`, `training_log.csv` |
| `PAS/evaluate.py` | `logs/{task_name}/PAS_results_{prob}_{split}.csv` |
| `BACKDOOR/evaluate.py` | `results/{expname}/{instance}.csv` |
| `CONFIG/evaluate.py` | `results/{expname}/scip_default/*.pkl`, `results/{expname}/{model}/*.pkl` |

---

## 8. Estimated Training Times

| Stage | CPU (8-core) | GPU (A100) |
|-------|-------------|-----------|
| Instance generation | < 1 min | — |
| PaS data collection (4 jobs) | 2–4 hours | — |
| PaS training, 100 epochs | ~45 min | ~8 min |
| Backdoor search (50 trials/instance) | ~4–8 hours | — |
| BACKDOOR model training, 1000 epochs | ~2 hours | ~20 min |
| SMAC config search (50 trials/instance) | ~6–12 hours | — |
| CONFIG model training, 1000 epochs | ~3 hours | ~25 min |
| Multi-task training, 100 epochs | ~3 hours | ~30 min |

---

## 9. Troubleshooting

### SCIP not found / pyscipopt import error
```
ImportError: cannot import name 'Model' from 'pyscipopt'
```
SCIP must be installed before pyscipopt. Set `SCIPOPTDIR` and reinstall:
```bash
export SCIPOPTDIR=/path/to/scip
pip install pyscipopt --force-reinstall
```

### `model.getValsLinear(c)` returns empty dict
Some SCIP versions don't expose constraint coefficients after presolving. The code already sets `presolving/maxrounds = 0`. If the problem persists, check that SCIP >= 7.0 is installed.

### NaN losses during training
Usually means a batch had no positive or negative samples. Check that:
- Backdoor CSVs have at least 10 rows per instance
- Config CSVs have at least 10 rows per instance
- The `backdoor_list` column contains valid Python list strings

### `torch_scatter` not installing
Match your torch and CUDA version exactly:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### Checkpoint format mismatch (`.pt` vs `.pth`)
Both formats are handled. When loading manually:
```python
ckpt = torch.load("model_best.pt", map_location="cpu")
state_dict = ckpt.get("model_state", ckpt)
model.load_state_dict(state_dict, strict=False)
```

### SCIP `limits/solutions` parameter error on older pySCIPOpt
The collect_data script wraps all `setParam` calls in try/except, so this is handled silently.

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `shared/scip_features.py` as single source | Eliminates the original bug where BACKDOOR used Gurobi (15/4 dims) and CONFIG used SCIP (19/5 dims). Now all three tasks pull from one place. |
| `shared/shared_encoder.py` | All three GAT files import and wrap `SharedEncoder` instead of each duplicating the backbone. |
| 19-dim var / 5-dim con features | Frozen. Defined once, imported everywhere. Do not change without updating all three task heads. |
| SCIP for all solving | No Gurobi license required anywhere in the pipeline. |
| Progressive `Prenorm` freeze | Kept exactly as in the original — changing this affects training stability. |
