"""
run_smac.py
===========
Generates the training data CSV required by CONFIG/train.py.

For each instance it randomly samples SCIP parameter configurations,
solves the instance with each one, and saves the results ranked by
solve time. No SMAC library required — this is a simple random search
that produces the same CSV format the Config training code expects.

If you have SMAC installed and want proper Bayesian optimisation,
replace the _random_config() function with a SMAC run.

Output CSV columns:  cost, params
  - cost   : wall-clock solve time in seconds (lower = better)
  - params : dict of SCIP parameter name -> value (as a string)

Row 0 = best (fastest) config. The training code uses the top 5 rows
as positive examples and the bottom 5 as negative examples, so you
need at least 10 trials that complete within the time limit.

Usage - single instance:
    python run_smac.py --instance instances/CA/train/CA_train_0000.lp --output datasets/CA --n_trials 50 --time_limit 300

Usage - whole directory (parallel):
    python run_smac.py --instance_dir instances/CA/train --output datasets/CA --n_trials 50 --time_limit 300 --n_jobs 4
"""

import os
import time
import random
import argparse
import pandas as pd
import pyscipopt as scip
from multiprocessing import Pool

# The 17 SCIP parameters being tuned (must match CONFIG/MIPDataset.py PARAM_SEQUENCE)
PARAM_SPACE = {
    "branching/scorefunc":              ["s", "p", "q"],
    "branching/lpgainnormalize":        ["d", "l", "s"],
    "branching/clamp":                  (0.0, 0.5),       # continuous
    "branching/midpull":                (0.0, 1.0),
    "branching/midpullreldomtrig":      (0.0, 1.0),
    "branching/preferbinary":           [True, False],
    "branching/scorefac":               (0.0, 1.0),
    "cutselection/hybrid/minortho":     (0.0, 1.0),
    "cutselection/hybrid/minorthoroot": (0.0, 1.0),
    "lp/pricing":                       ["l", "a", "f", "p", "s", "q", "d"],
    "lp/colagelimit":                   (0, 2147483647),   # integer
    "lp/rowagelimit":                   (0, 2147483647),
    "nodeselection/childsel":           ["d", "u", "p", "i", "l", "r", "h"],
    "separating/cutagelimit":           (0, 2147483647),
    "separating/maxcutsgenfactor":      (0, 2147483647),
    "separating/maxcutsrootgenfactor":  (0, 2147483647),
    "separating/poolfreq":              (0, 65534),
}

INTEGER_PARAMS = {
    "lp/colagelimit", "lp/rowagelimit",
    "separating/cutagelimit", "separating/maxcutsgenfactor",
    "separating/maxcutsrootgenfactor", "separating/poolfreq",
}


def _random_config(rng):
    """Sample one random configuration from PARAM_SPACE."""
    cfg = {}
    for name, space in PARAM_SPACE.items():
        if isinstance(space, list):
            cfg[name] = rng.choice(space)
        elif isinstance(space, tuple):
            lo, hi = space
            if name in INTEGER_PARAMS:
                cfg[name] = rng.randint(lo, hi)
            else:
                cfg[name] = rng.uniform(lo, hi)
    return cfg


def _solve_with_config(instance_path, params, time_limit):
    """Solve instance with given SCIP params. Returns solve time or None on failure."""
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(instance_path)
    m.setParam("limits/time", time_limit)
    m.setParam("display/verblevel", 0)

    for k, v in params.items():
        try:
            m.setParam(k, v)
        except Exception:
            pass  # skip params not supported by this SCIP version

    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0

    if m.getNSols() == 0:
        return None
    return elapsed


def search_one_instance(instance_path, output_dir, n_trials, time_limit, seed=0):
    """Run config search on one instance and save result CSV."""
    basename = os.path.splitext(os.path.basename(instance_path))[0]
    out_path = os.path.join(output_dir, basename + ".csv")

    if os.path.exists(out_path):
        print(f"[skip] {basename} (already exists)")
        return

    print(f"[search] {basename}  trials={n_trials}  time_limit={time_limit}s")

    rng     = random.Random(seed)
    results = []

    for trial in range(n_trials):
        cfg     = _random_config(rng)
        elapsed = _solve_with_config(instance_path, cfg, time_limit)
        if elapsed is None:
            continue
        results.append({"cost": elapsed, "params": str(cfg)})

    if len(results) < 10:
        print(f"  [warn] {basename}: only {len(results)} successful trials "
              f"(need >= 10). Try increasing --n_trials or --time_limit.")
        if not results:
            return

    df = pd.DataFrame(results).sort_values("cost").reset_index(drop=True)
    df.to_csv(out_path)
    print(f"  saved {len(df)} rows -> {out_path}  "
          f"(best={df.iloc[0]['cost']:.2f}s  worst={df.iloc[-1]['cost']:.2f}s)")


def _worker(args):
    search_one_instance(*args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance",     default=None,
                   help="Path to a single .lp instance file")
    p.add_argument("--instance_dir", default=None,
                   help="Directory of .lp files to process in batch")
    p.add_argument("--output",       required=True,
                   help="Output directory for result CSVs")
    p.add_argument("--n_trials",     type=int, default=50,
                   help="Number of random configurations to try per instance")
    p.add_argument("--time_limit",   type=int, default=300,
                   help="SCIP time limit per solve in seconds")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--n_jobs",       type=int, default=1,
                   help="Parallel workers (batch mode only)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.instance:
        search_one_instance(args.instance, args.output,
                            args.n_trials, args.time_limit, args.seed)

    elif args.instance_dir:
        files = sorted(f for f in os.listdir(args.instance_dir) if f.endswith(".lp"))
        if not files:
            print(f"No .lp files found in {args.instance_dir}")
            return

        print(f"Found {len(files)} instances in {args.instance_dir}")
        job_args = [(os.path.join(args.instance_dir, f), args.output,
                     args.n_trials, args.time_limit, args.seed + i)
                    for i, f in enumerate(files)]

        if args.n_jobs == 1:
            for a in job_args:
                _worker(a)
        else:
            with Pool(args.n_jobs) as pool:
                pool.map(_worker, job_args)

    else:
        p.error("Provide either --instance or --instance_dir")

    print("\nConfig search complete.")


if __name__ == "__main__":
    main()
