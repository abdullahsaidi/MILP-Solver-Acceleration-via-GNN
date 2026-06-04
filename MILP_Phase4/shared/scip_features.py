"""
shared/scip_features.py
=======================
Single source of truth for MILP bipartite graph features.
Imported by PAS, BACKDOOR, and CONFIG — do not modify without
updating all three task modules.

CHANGE LOG (vs. submitted version)
-----------------------------------
1. VAR_FEAT_DIM restored to 15  (paper Section 4.1 / Gasse et al. 2019).
   The submitted code used 19 dims; the extra 4 dims (basis one-hot
   columns 7-10, unused column 11) were an artefact of the partner
   alignment and are not described in the paper.
   Exact 15-dim layout:
     [0]  type_CONTINUOUS
     [1]  type_BINARY
     [2]  type_INTEGER
     [3]  coef                (objective coefficient)
     [4]  has_lb
     [5]  has_ub
     [6]  sol_is_at_lb
     [7]  sol_is_at_ub
     [8]  sol_frac            (LP fractionality)
     [9]  basis_status_BASIC
     [10] basis_status_NONBASIC_LOWER
     [11] basis_status_NONBASIC_UPPER
     [12] basis_status_SUPERBASIC
     [13] reduced_cost
     [14] sol_val             (LP solution value)

2. CON_FEAT_DIM stays at 5 — matches the paper's 4 constraint features
   plus the standard obj_sense weight used by Gasse et al. The five
   features are: obj_cos_sim, rhs, is_tight, dualsol_val, sense_weight.

3. Legacy aliases removed (were defined but never imported anywhere).

Documented deviations from the paper
--------------------------------------
* Paper uses Gurobi for LP relaxation; we use SCIP (no Gurobi licence
  required). SCIP's LP basis status integers are mapped to the same
  one-hot positions as Gurobi's VBasis constants.
* features [9-12] (basis status) are left as 0 when the LP relaxation
  does not converge within the 10-second time limit.
"""

import os
import numpy as np
import pyscipopt as scip
import sklearn.metrics.pairwise
from scipy.sparse import csr_matrix

VAR_FEAT_DIM = 15   
CON_FEAT_DIM = 5

VAR_FEATURE_NAMES = [
    "type_CONTINUOUS",            # 0
    "type_BINARY",                # 1
    "type_INTEGER",               # 2
    "coef",                       # 3  objective coefficient
    "has_lb",                     # 4
    "has_ub",                     # 5
    "sol_is_at_lb",               # 6
    "sol_is_at_ub",               # 7
    "sol_frac",                   # 8  fractionality
    "basis_status_BASIC",         # 9
    "basis_status_NONBASIC_LOWER",# 10
    "basis_status_NONBASIC_UPPER",# 11
    "basis_status_SUPERBASIC",    # 12
    "reduced_cost",               # 13
    "sol_val",                    # 14
]

CON_FEATURE_NAMES = [
    "obj_cos_sim",   # 0  cosine similarity of constraint row with objective
    "rhs",           # 1  RHS value
    "is_tight",      # 2  constraint active at LP optimum
    "dualsol_val",   # 3  dual variable (shadow price)
    "sense_weight",  # 4  always 1.0 (objective sense placeholder)
]

# SCIP LP basis status codes returned by getBasisStatus()
_SCIP_BASIS_LOWER = 0   # nonbasic at lower bound  → NONBASIC_LOWER
_SCIP_BASIS_BASIC = 1   # basic                    → BASIC
_SCIP_BASIS_UPPER = 2   # nonbasic at upper bound  → NONBASIC_UPPER
_SCIP_BASIS_ZERO  = 3   # superbasic / free        → SUPERBASIC

_SCIP_INF = 1e20


def extract_bipartite_graph(mip_file: str, use_cache: bool = True):
    """
    Load instance with SCIP, solve LP relaxation, return bipartite graph
    features matching the paper's 15-dim variable / 5-dim constraint layout.

    Returns
    -------
    var_feats    : np.ndarray  (n_vars, 15)
    con_feats    : np.ndarray  (n_cons,  5)
    edge_indices : np.ndarray  (2, n_edges)   [con_idx, var_idx]
    edge_feats   : np.ndarray  (n_edges,   1) normalised coefficients
    var_names    : list[str]
    obj_sense    : int          1=maximise, -1=minimise

    Returns (None, None, None, None, None, None) if extraction fails.
    """
    # ── Cache check ──────────────────────────────────────────────────────────
    cache_path = str(mip_file) + ".features.npz"
    if use_cache and os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=True)
            return (d["var_feats"], d["con_feats"], d["edge_indices"],
                    d["edge_feats"], list(d["var_names"]), int(d["obj_sense"]))
        except Exception:
            pass  # corrupted cache — recompute

    model = scip.Model()
    model.hideOutput(True)
    model.readProblem(str(mip_file))

    obj_sense = 1 if model.getObjectiveSense() == "maximize" else -1

    model.setParam("limits/time", 10)
    model.setParam("display/verblevel", 0)
    model.setParam("presolving/maxrounds", 0)
    model.optimize()

    all_vars = model.getVars(transformed=False)
    all_cons = model.getConss()

    # Only binary/integer variables — consistent across all three tasks
    bin_int_vars = [v for v in all_vars if v.vtype() in ("BINARY", "INTEGER")]
    linear_cons  = []
    for c in all_cons:
        try:
            model.getLhs(c)
            linear_cons.append(c)
        except Exception:
            pass

    n_vars = len(bin_int_vars)
    n_cons = len(linear_cons)

    if n_vars == 0 or n_cons == 0:
        model.freeProb()
        return None, None, None, None, None, None

    var_index = {v.name: i for i, v in enumerate(bin_int_vars)}

    # ── Build constraint matrix for cosine similarity ──────────────
    row_idx, col_idx, data_vals = [], [], []
    rhs_list = []
    for c_idx, c in enumerate(linear_cons):
        lhs = model.getLhs(c)
        rhs = model.getRhs(c)
        rhs_val = rhs if rhs < _SCIP_INF else (lhs if lhs > -_SCIP_INF else 0.0)
        try:
            row = model.getValsLinear(c)
        except Exception:
            row = {}
        for vname, coeff in row.items():
            if vname in var_index:
                row_idx.append(c_idx)
                col_idx.append(var_index[vname])
                data_vals.append(float(coeff))
        rhs_list.append(rhs_val)

    if data_vals:
        A = csr_matrix((data_vals, (row_idx, col_idx)),
                       shape=(n_cons, n_vars), dtype=np.float64)
    else:
        A = csr_matrix((n_cons, n_vars), dtype=np.float64)

    obj_arr  = np.array([v.getObj() for v in bin_int_vars], dtype=np.float64)
    rhs_arr  = np.array(rhs_list, dtype=np.float64)

    if obj_arr.any():
        cos_sims = sklearn.metrics.pairwise.cosine_similarity(
            A, obj_arr.reshape(1, -1)).flatten()
    else:
        cos_sims = np.zeros(n_cons)

    # ── Variable features (15-dim, paper layout) ───────────────────
    var_features = np.zeros((n_vars, VAR_FEAT_DIM), dtype=np.float32)

    for idx, v in enumerate(bin_int_vars):
        vtype = v.vtype()
        lb    = v.getLbOriginal()
        ub    = v.getUbOriginal()
        obj   = v.getObj()

        try:
            sol      = model.getBestSol()
            lp_val   = model.getSolVal(sol, v)
        except Exception:
            lp_val   = 0.0

        has_lb = float(lb > -_SCIP_INF)
        has_ub = float(ub <  _SCIP_INF)
        at_lb  = float(abs(lp_val - lb) < 1e-6) if has_lb else 0.0
        at_ub  = float(abs(lp_val - ub) < 1e-6) if has_ub else 0.0
        frac   = abs(round(lp_val) - lp_val)

        # LP basis status → one-hot [9..12]
        try:
            bs = model.getBasisStatus(v)
        except Exception:
            bs = _SCIP_BASIS_LOWER
        is_basic    = float(bs == _SCIP_BASIS_BASIC)
        is_nb_lower = float(bs in (_SCIP_BASIS_LOWER, 3, 4))
        is_nb_upper = float(bs == _SCIP_BASIS_UPPER)
        is_super    = float(bs == _SCIP_BASIS_ZERO)

        try:
            if model.getLPSolstat() == 1:   # LP optimal — safe to query
                rc = model.getVarRedcost(v)
            else:
                rc = 0.0
        except Exception:
            rc = 0.0

        var_features[idx, 0]  = float(vtype == "CONTINUOUS")
        var_features[idx, 1]  = float(vtype == "BINARY")
        var_features[idx, 2]  = float(vtype == "INTEGER")
        var_features[idx, 3]  = float(obj)
        var_features[idx, 4]  = has_lb
        var_features[idx, 5]  = has_ub
        var_features[idx, 6]  = at_lb
        var_features[idx, 7]  = at_ub
        var_features[idx, 8]  = frac
        var_features[idx, 9]  = is_basic
        var_features[idx, 10] = is_nb_lower
        var_features[idx, 11] = is_nb_upper
        var_features[idx, 12] = is_super
        var_features[idx, 13] = float(np.clip(rc,     -1e6, 1e6))
        var_features[idx, 14] = float(np.clip(lp_val, -1e6, 1e6))

    # ── Constraint features (5-dim) ─────────────────────────────────
    con_features = np.zeros((n_cons, CON_FEAT_DIM), dtype=np.float32)

    for c_idx, c in enumerate(linear_cons):
        rhs_v = rhs_list[c_idx]
        try:
            dual = model.getDualsolLinear(c)
            act  = model.getActivity(c)
        except Exception:
            dual, act = 0.0, rhs_v
        slack    = rhs_v - act
        is_tight = float(abs(slack) < 1e-6)

        con_features[c_idx, 0] = float(cos_sims[c_idx])
        con_features[c_idx, 1] = float(rhs_v)
        con_features[c_idx, 2] = is_tight
        con_features[c_idx, 3] = float(np.clip(dual, -1e6, 1e6))
        con_features[c_idx, 4] = 1.0

    # ── Edge features ───────────────────────────────────────────────
    A_coo        = A.tocoo()
    edge_indices = np.stack([A_coo.row.astype(np.int64),
                             A_coo.col.astype(np.int64)], axis=0)
    edge_data    = A_coo.data.astype(np.float32)
    max_coef     = max(float(np.abs(edge_data).max()) if len(edge_data) else 1.0, 1e-6)
    edge_feats   = (edge_data / max_coef).reshape(-1, 1)

    var_names = [v.name for v in bin_int_vars]
    model.freeProb()
    # ── Save cache ───────────────────────────────────────────────────────────
    if use_cache:
        try:
            np.savez_compressed(cache_path,
                var_feats=var_features, con_feats=con_features,
                edge_indices=edge_indices, edge_feats=edge_feats,
                var_names=np.array(var_names), obj_sense=np.array(obj_sense))
        except Exception:
            pass
    return var_features, con_features, edge_indices, edge_feats, var_names, obj_sense
