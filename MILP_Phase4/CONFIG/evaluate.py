import os, pickle, time, argparse
import numpy as np
import torch
import pyscipopt
from pyscipopt import Model

from CONFIG.MIPDataset import compute_mip_representation, decode_params
from CONFIG.GAT import CONFIG_GATPolicy as GATPolicy

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

EVAL_TIME_LIMIT = 900


class PrimalLogger(pyscipopt.Eventhdlr):
    def eventinit(self):
        self._t0 = time.monotonic()
        self.primal_values = []
        self.timestamps    = []
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self): pass

    def eventexec(self, event):
        self.primal_values.append(self.model.getSolObjVal(self.model.getBestSol()))
        self.timestamps.append(time.monotonic() - self._t0)

    @property
    def log(self):
        return [self.primal_values, self.timestamps]


def _solve_with_scip(instance_path, params=None):
    model = Model(); model.hideOutput(); model.readProblem(str(instance_path))
    event = PrimalLogger(); model.includeEventhdlr(event, "", "")
    model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParam("limits/memory", 12 * 1024)
    model.setParam("limits/time",   EVAL_TIME_LIMIT)
    if params:
        for k, v in params.items():
            try: model.setParam(k, v)
            except Exception: pass
    model.optimize(); model.freeProb()
    return event


def evaluate_model(model_dir, expname, instance_path):
    out_dir  = os.path.join("results", expname, os.path.basename(model_dir.rstrip("/\\")))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(instance_path) + ".pkl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path}"); return

    print(f"[CONFIG model] {os.path.basename(instance_path)}")
    data = compute_mip_representation(instance_path)

    gat  = GATPolicy()
    ckpt = torch.load(os.path.join(model_dir, "model_best.pth"), map_location="cpu")
    gat.load_state_dict(ckpt.get("model_state", ckpt)); gat.eval()

    data.x_vars_batch = torch.zeros(data.x_vars.shape[0], dtype=torch.long)
    data.x_cons_batch = torch.zeros(data.x_cons.shape[0], dtype=torch.long)
    with torch.no_grad():
        raw = gat(data.x_cons, data.edge_index_cons_to_vars, data.edge_attr,
                  data.x_vars, data.x_vars_batch, data.x_cons_batch)

    event = _solve_with_scip(instance_path, decode_params(raw[0].numpy()))
    with open(out_path, "wb") as f: pickle.dump(event.log, f)
    print(f"  Saved -> {out_path}")


def evaluate_scip_default(expname, instance_path):
    out_dir  = os.path.join("results", expname, "scip_default")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(instance_path) + ".pkl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path}"); return
    print(f"[SCIP default] {os.path.basename(instance_path)}")
    event = _solve_with_scip(instance_path)
    with open(out_path, "wb") as f: pickle.dump(event.log, f)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instance",  required=True)
    p.add_argument("--expname",   required=True)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--scip_only", action="store_true")
    args = p.parse_args()
    if not args.scip_only and args.model_dir:
        evaluate_model(args.model_dir, args.expname, args.instance)
    evaluate_scip_default(args.expname, args.instance)
