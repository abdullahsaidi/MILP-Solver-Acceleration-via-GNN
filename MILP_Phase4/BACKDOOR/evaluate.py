import os, argparse, time
import torch
import pyscipopt as scip
import pandas as pd

from BACKDOOR.MIPDataset import compute_mip_representation
from BACKDOOR.GAT import BACKDOOR_GATPolicy as GATPolicy

TOP_K_VARS = 8


def _load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state", ckpt)
    net  = GATPolicy()
    # Handle multitask checkpoint: remap encoder + backdoor_head -> scoring_head
    if any(k.startswith("encoder.") and not k.startswith("encoder.var_norm") for k in sd):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                new_sd[k] = v
            elif k.startswith("backdoor_head."):
                new_k = k.replace("backdoor_head.", "scoring_head.")
                new_sd[new_k] = v
        net.load_state_dict(new_sd, strict=False)
    else:
        net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def _top_k_indices(net, graph, k=TOP_K_VARS):
    with torch.no_grad():
        scores = net(graph.x_cons, graph.edge_index_cons_to_vars,
                     graph.edge_attr, graph.x_vars)
    _, idx = torch.topk(scores, k)
    return set(idx.tolist())


def _solve_scip_default(inst_path):
    m = scip.Model(); m.hideOutput(True); m.readProblem(inst_path)
    t0 = time.perf_counter(); m.optimize(); return time.perf_counter() - t0


def _solve_with_branching(inst_path, net):
    graph    = compute_mip_representation(inst_path)
    prio_set = _top_k_indices(net, graph)
    m = scip.Model(); m.hideOutput(True)
    m.setParam("display/verblevel", 0); m.readProblem(inst_path)
    for idx, var in enumerate(m.getVars()):
        m.chgVarBranchPriority(var, 2 if idx in prio_set else 1)
    t0 = time.perf_counter(); m.optimize(); return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance",        required=True)
    p.add_argument("--expname",         required=True)
    p.add_argument("--model",           required=True)
    p.add_argument("--multitask_model", required=True)
    args = p.parse_args()

    out_dir = os.path.join("results", args.expname)
    os.makedirs(out_dir, exist_ok=True)

    # Summary CSV path — append results across calls
    summary_csv = os.path.join(out_dir, "summary.csv")
    inst_name   = os.path.basename(args.instance)

    print(f"\n[{inst_name}]")
    print("[1/3] SCIP default ...")
    t_scip = _solve_scip_default(args.instance); print(f"      {t_scip:.3f}s")

    print("[2/3] GAT single-task ...")
    t_gat  = _solve_with_branching(args.instance, _load_model(args.model))
    print(f"      {t_gat:.3f}s")

    print("[3/3] GAT multi-task ...")
    t_mt   = _solve_with_branching(args.instance, _load_model(args.multitask_model))
    print(f"      {t_mt:.3f}s")

    # Append to summary CSV
    row = pd.DataFrame([{"instance": inst_name,
                         "scip_default": t_scip,
                         "gat_singletask": t_gat,
                         "gat_multitask": t_mt}])
    if os.path.exists(summary_csv):
        row.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        row.to_csv(summary_csv, index=False)

    # Print running average
    df_all = pd.read_csv(summary_csv)
    n = len(df_all)
    print(f"\n--- Running average over {n} instance(s) ---")
    print(f"  SCIP default  : {df_all['scip_default'].mean():.3f}s")
    print(f"  GAT single    : {df_all['gat_singletask'].mean():.3f}s")
    print(f"  GAT multi     : {df_all['gat_multitask'].mean():.3f}s")
    print(f"  Summary CSV   : {summary_csv}")


if __name__ == "__main__":
    main()
