import os, argparse
import numpy as np
import pyscipopt as scip
from multiprocessing import Pool

from shared.scip_features import extract_bipartite_graph

TIME_LIMIT    = 300
MAX_SOLUTIONS = 50
ENERGY_TEMP   = 1.0


class _SolutionCollector(scip.Eventhdlr):
    def __init__(self, var_names, max_solutions):
        self.var_names     = var_names
        self.max_solutions = max_solutions
        self.solutions     = []

    def eventinit(self):
        self.model.catchEvent(scip.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(scip.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        if len(self.solutions) >= self.max_solutions:
            return {}
        sol = self.model.getBestSol()
        obj = self.model.getSolObjVal(sol)
        vals = {}
        for vname in self.var_names:
            try:
                v = self.model.getVarByName(vname)
                vals[vname] = self.model.getSolVal(sol, v)
            except Exception:
                vals[vname] = 0.0
        self.solutions.append((obj, vals))
        return {}


def _compute_labels(solutions, var_names, obj_sense, temperature=ENERGY_TEMP):
    if not solutions:
        return np.zeros(len(var_names), dtype=np.float32)
    objs   = np.array([s[0] for s in solutions], dtype=np.float64)
    scaled = objs if obj_sense == 1 else -objs
    scaled -= scaled.max()
    weights = np.exp(scaled / temperature)
    weights /= weights.sum()
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    labels = np.zeros(len(var_names), dtype=np.float32)
    for w, (_, vals) in zip(weights, solutions):
        for vname, val in vals.items():
            if vname in var_to_idx:
                labels[var_to_idx[vname]] += w * float(val > 0.5)
    return labels


def _process_instance(args):
    inst_path, out_path, time_limit, max_solutions = args
    if os.path.exists(out_path):
        return f"SKIP {os.path.basename(inst_path)}"
    try:
        model = scip.Model()
        model.hideOutput(True)
        model.readProblem(inst_path)
        obj_sense = 1 if model.getObjectiveSense() == "maximize" else -1

        try: model.setParam("limits/time", time_limit)
        except Exception: pass
        try: model.setParam("limits/solutions", max_solutions)
        except Exception: pass
        try: model.setParam("presolving/maxrounds", 0)
        except Exception: pass
        try: model.setParam("randomization/randomseedshift", 42)
        except Exception: pass

        var_names = [v.name for v in model.getVars(transformed=False)
                     if v.vtype() in ("BINARY", "INTEGER")]
        if not var_names:
            return f"SKIP {os.path.basename(inst_path)} (no binary/int vars)"

        collector = _SolutionCollector(var_names, max_solutions)
        model.includeEventhdlr(collector, "SolCollector", "Collects solutions")
        model.optimize()

        if model.getNSols() < 1:
            return f"SKIP {os.path.basename(inst_path)} (no solutions)"

        vf, cf, ei, ef, vnames, _ = extract_bipartite_graph(inst_path)
        if vf is None:
            return f"SKIP {os.path.basename(inst_path)} (feature extraction failed)"

        labels = _compute_labels(collector.solutions, vnames, obj_sense)
        np.savez_compressed(out_path, var_features=vf, con_features=cf,
                            edge_indices=ei, edge_features=ef,
                            labels=labels, obj_sense=np.array([obj_sense]))
        return f"OK  {os.path.basename(inst_path)} | vars={len(vnames)} | sols={model.getNSols()}"
    except Exception as e:
        return f"ERR {os.path.basename(inst_path)}: {e}"


def collect_data(problem, split, instance_dir="instances", dataset_dir="dataset",
                 n_jobs=1, time_limit=TIME_LIMIT, max_solutions=MAX_SOLUTIONS):
    inst_dir = os.path.join(instance_dir, problem, split)
    out_dir  = os.path.join(dataset_dir,  problem, split)
    os.makedirs(out_dir, exist_ok=True)
    files    = sorted(f for f in os.listdir(inst_dir) if f.endswith(".lp"))
    if not files:
        print(f"No .lp files in {inst_dir}"); return

    print(f"\n[{problem}/{split}] {len(files)} instances | {n_jobs} worker(s)")
    args_list = [(os.path.join(inst_dir, f),
                  os.path.join(out_dir, f.replace(".lp", ".npz")),
                  time_limit, max_solutions) for f in files]

    results = [_process_instance(a) for a in args_list] if n_jobs == 1 \
              else Pool(n_jobs).map(_process_instance, args_list)

    ok   = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    err  = sum(1 for r in results if r.startswith("ERR"))
    for r in results: print(f"  {r}")
    print(f"\n[{problem}/{split}] Done: {ok} saved, {skip} skipped, {err} errors")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--problem",       default="all", choices=["CA","MIS","MVC","all"])
    p.add_argument("--split",         default="all", choices=["train","test","all"])
    p.add_argument("--instance_dir",  default="instances")
    p.add_argument("--dataset_dir",   default="dataset")
    p.add_argument("--n_jobs",        type=int, default=1)
    p.add_argument("--time_limit",    type=int, default=TIME_LIMIT)
    p.add_argument("--max_solutions", type=int, default=MAX_SOLUTIONS)
    args = p.parse_args()

    problems = ["CA","MIS","MVC"] if args.problem == "all" else [args.problem]
    splits   = ["train","test"]   if args.split   == "all" else [args.split]
    for prob in problems:
        for spl in splits:
            collect_data(prob, spl, instance_dir=args.instance_dir,
                         dataset_dir=args.dataset_dir, n_jobs=args.n_jobs,
                         time_limit=args.time_limit, max_solutions=args.max_solutions)
    print("\nPAS data collection complete.")
