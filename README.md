# MILP Solver Acceleration with a Multi-Task Graph Attention Network

Reproduction and extension of **"Multi-task Representation Learning for Mixed Integer Linear Programming" (CPAIOR 2025)**, built as a university team project at Princess Sumaya University for Technology.

Mixed-Integer Linear Programs (MILPs) power scheduling, routing and allocation problems, but solving them can be slow. This project trains **one shared Graph Attention Network (GAT)** on three learning tasks that each help the open-source **SCIP** solver work faster:

| Task | What the model learns | How it helps the solver |
|---|---|---|
| **PaS** (Predict-and-Search) | Probability that each binary variable equals 1 | Confident variables are fixed before solving, shrinking the search space |
| **Backdoor** | Which small set of variables to branch on first | Variables in the predicted backdoor get higher branching priority |
| **Config** | Best SCIP parameter configuration (33-dim) for an instance | Replaces expensive per-instance parameter search |

## How it works

1. **Instances → graphs.** Each MILP is converted into a bipartite variable–constraint graph (19 variable features, 5 constraint features, normalized coefficients on the edges), extracted directly from SCIP.
2. **Shared encoder.** A GAT encoder with online feature normalization produces 64-dim embeddings for every variable and constraint.
3. **Task heads.** Three lightweight heads sit on top of the shared encoder: per-variable PaS probabilities, per-variable backdoor scores, and a pooled per-instance configuration vector.
4. **Joint training.** The three losses (BCE for PaS, contrastive InfoNCE for Backdoor and Config) are combined, so the encoder learns from all tasks at once.

Experiments use **900 synthetic instances** of three problem families: Combinatorial Auction (CA), Maximum Independent Set (MIS) and Minimum Vertex Cover (MVC). The original paper used Gurobi; this implementation runs entirely on **SCIP via PySCIPOpt**.

## Our extensions

**1. Adaptive confidence-thresholded fixing (PaS).** The original method always fixes a fixed percentage of variables, however unsure the model is. On MVC this pinned many variables to wrong values and the search failed completely. We added confidence thresholds: a variable is fixed only if the model is confident enough, and if nothing qualifies the solver falls back to plain SCIP.

**2. Learnable multi-task loss weights.** Instead of hand-tuned fixed weights for the three task losses, each task gets a learnable uncertainty (log-variance) parameter, following Kendall, Gal & Cipolla (CVPR 2018). The model learns how much to trust each task during training.

## Results: PaS on test instances (100 per problem)

Metric: **primal integral**, the area under the primal-gap curve over the time limit. Lower is better.

| Problem | SCIP baseline | PaS (paper method) | PaS + adaptive thresholds |
|---|---|---|---|
| CA | 0.04 | 0.04 | **0.03** |
| MIS | 0.50 | 0.08 | **0.07** |
| MVC | 0.08 | 1000 (failed, gap = 100%) | **0.08** |

The adaptive thresholds removed the MVC failure, where the model is not confident and safely falls back to SCIP, while keeping or slightly improving the gains on CA and MIS.

## Repository layout

```
MILP_Phase4/
├── PAS/              # PaS data collection, model, training, evaluation (incl. adaptive thresholds)
├── BACKDOOR/         # Backdoor dataset, GAT policy, contrastive training, evaluation
├── CONFIG/           # Config dataset, GAT policy, training, evaluation
├── MULTITASK/        # Joint multi-task model and training loop (incl. learnable loss weights)
├── backdoor_search.py
├── dataset/          # Pre-processed PaS graphs (.npz)
├── data/             # Backdoor search results
└── README.md         # Full setup, data-generation and training guide
```

See **[MILP_Phase4/README.md](MILP_Phase4/README.md)** for environment setup, the data-generation pipeline and all training commands.

## Tech stack

Python · PyTorch · PyTorch Geometric · Graph Attention Networks · SCIP / PySCIPOpt · Multi-task learning · Contrastive learning (InfoNCE)

## My contributions

Team project. I implemented the **PaS** task (data collection, model, training and evaluation), integrated the PaS, Backdoor and Config heads with the shared encoder into a single multi-task model, took part in data generation and ran the experiments. The two extensions were designed and tested together as a team.
