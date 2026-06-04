"""
backdoor_search.py
==================
Generates the training data CSV required by BACKDOOR/train.py.

For each instance it:
  1. Solves the LP relaxation with SCIP to get variable values
  2. Randomly samples subsets of binary/integer variables (backdoor candidates)
  3. Fixes each subset to its LP-rounded values and times SCIP on the reduced problem
  4. Saves all results sorted by run_time ascending

Output CSV columns:  backdoor_list, run_time
  - backdoor_list : Python list of variable indices (as a string)
  - run_time      : wall-clock seconds SCIP needed after fixing those variables

The training code uses the top 5 rows (fastest = positive examples)
and bottom 5 rows (slowest = negative examples), so you need at least
10 trials that all find a feasible solution.

Usage — single instance:
    python backdoor_search.py \
        --instance instances/CA/train/CA_train_0000.lp \
        --output   data/backdoor_results \
        --n_trials 50 \
        --backdoor_size 8

Usage — whole directory (parallel):
    python backdoor_search.py \
        --instance_dir instances/CA/train \
        --output       data/backdoor_results \
        --n_trials     50 \
        --backdoor_size 8 \
        --n_jobs        4
"""

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import pyscipopt as scip
from multiprocessing import Pool

SOLVE_TIME_LIMIT = 60   # seconds per reduced MILP solve


# ---------------------------------------------------------------------------

def _get_lp_solution(model):
    """Return {var_name: lp_value} for binary/integer variables."""
    lp_vals = {}
    vars_list = model.getVars(transformed=False)
    best_sol  = model.getBestSol()
    for v in vars_list:
        if v.vtype() in ("BINARY", "INTEGER"):
            try:
                lp_vals[v.name] = model.getSolVal(best_sol, v)
            except Exception:
                lp_vals[v.name] = 0.0
    return lp_vals


def _solve_with_fixed_vars(instance_path, var_names, fixed_indices, lp_vals, time_limit):
    """
    Fix the variables at fixed_indices to their rounded LP values,
    then solve with SCIP. Returns wall-clock seconds, or None if infeasible.
    """
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(instance_path)
    m.setParam("limits/time", time_limit)
    m.setParam("display/verblevel", 0)

    # build a name -> var mapping for the fresh model
    name_to_var = {v.name: v for v in m.getVars(transformed=False)
                   if v.vtype() in ("BINARY", "INTEGER")}

    for idx in fixed_indices:
        vname = var_names[idx]
        if vname not in name_to_var:
            continue
        v   = name_to_var[vname]
        val = float(round(lp_vals.get(vname, 0.0)))
        val = max(v.getLbOriginal(), min(v.getUbOriginal(), val))
        m.chgVarLb(v, val)
        m.chgVarUb(v, val)

    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0

    if m.getNSols() == 0:
        return None     # infeasible or timeout with no solution
    return elapsed


def search_one_instance(instance_path, output_dir, n_trials, backdoor_size, seed=0):
    """Run backdoor search on a single instance and save the result CSV."""
    out_name = os.path.basename(instance_path)   # keeps the .lp extension
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        print(f"[skip] {out_name} (already exists)")
        return

    print(f"[search] {out_name}  trials={n_trials}  size={backdoor_size}")

    # --- solve LP relaxation to get values to fix variables to ---
    lp_model = scip.Model()
    lp_model.hideOutput(True)
    lp_model.readProblem(instance_path)
    lp_model.setParam("presolving/maxrounds", 0)
    lp_model.setParam("limits/time", 60)
    lp_model.optimize()

    bin_int_vars = [v for v in lp_model.getVars(transformed=False)
                    if v.vtype() in ("BINARY", "INTEGER")]

    if len(bin_int_vars) < backdoor_size:
        print(f"  [skip] {out_name}: only {len(bin_int_vars)} integer vars, need {backdoor_size}")
        return

    var_names = [v.name for v in bin_int_vars]
    lp_vals   = _get_lp_solution(lp_model)
    lp_model.freeProb()

    # --- random backdoor search ---
    rng     = random.Random(seed)
    results = []
    n_vars  = len(var_names)

    for trial in range(n_trials):
        indices = sorted(rng.sample(range(n_vars), backdoor_size))
        elapsed = _solve_with_fixed_vars(instance_path, var_names, indices,
                                         lp_vals, SOLVE_TIME_LIMIT)
        if elapsed is None:
            continue    # skip infeasible / no-solution trials
        results.append({"backdoor_list": str(indices), "run_time": elapsed})

    if len(results) < 10:
        print(f"  [warn] {out_name}: only {len(results)} feasible trials "
              f"(need >= 10 for training). Try increasing --n_trials.")
        if not results:
            return

    df = pd.DataFrame(results).sort_values("run_time").reset_index(drop=True)
    df.to_csv(out_path)
    print(f"  saved {len(df)} rows -> {out_path}  "
          f"(best={df.iloc[0]['run_time']:.2f}s  worst={df.iloc[-1]['run_time']:.2f}s)")


def _worker(args):
    search_one_instance(*args)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    # single-instance mode
    p.add_argument("--instance",      default=None,
                   help="Path to a single .lp instance file")

    # batch mode
    p.add_argument("--instance_dir",  default=None,
                   help="Directory of .lp files to process in batch")

    p.add_argument("--output",        required=True,
                   help="Output directory for result CSVs")
    p.add_argument("--n_trials",      type=int, default=50,
                   help="Number of random backdoor subsets to try per instance")
    p.add_argument("--backdoor_size", type=int, default=8,
                   help="Number of variables to fix in each trial")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--n_jobs",        type=int, default=1,
                   help="Parallel workers (batch mode only)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.instance:
        # single file
        search_one_instance(args.instance, args.output,
                            args.n_trials, args.backdoor_size, args.seed)

    elif args.instance_dir:
        # batch over a directory
        files = sorted(f for f in os.listdir(args.instance_dir) if f.endswith(".lp"))
        if not files:
            print(f"No .lp files found in {args.instance_dir}")
            return

        print(f"Found {len(files)} instances in {args.instance_dir}")
        job_args = [(os.path.join(args.instance_dir, f), args.output,
                     args.n_trials, args.backdoor_size, args.seed + i)
                    for i, f in enumerate(files)]

        if args.n_jobs == 1:
            for a in job_args:
                _worker(a)
        else:
            with Pool(args.n_jobs) as pool:
                pool.map(_worker, job_args)

    else:
        p.error("Provide either --instance or --instance_dir")

    print("\nBackdoor search complete.")


if __name__ == "__main__":
    main()
