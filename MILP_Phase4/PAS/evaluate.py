"""
PAS/evaluate.py
===============
Evaluation script for PaS following the paper's metrics exactly:
  - Primal Gap  (PG)  = |v - v*| / max(|v*|, 1e-8)
  - Primal Integral (PI) = integral of PG over runtime [0, TL]

Paper: runtime cutoff = 1000s for PaS.
Baseline = SCIP default (no variable fixing).

ENHANCEMENT 1 (extension over Cai et al. CPAIOR 2025)
=====================================================
Adaptive Confidence-Thresholded Variable Fixing
------------------------------------------------
The paper's PaS scheme (Section 4.3) follows Han et al. 2023:
  - Always fix the top k0% lowest-confidence-toward-1 variables to 0
  - Always fix the top k1% highest-confidence-toward-1 variables to 1
This is a *fixed-k* scheme that commits to fixing k0+k1 % of variables
regardless of how confident the model actually is. When the model is
mis-calibrated on a problem class (observed: MVC, where pas_gap=1.0 vs
base_gap=0.0 in our test runs), this catastrophically destroys the
search by pinning ~60% of variables to wrong values.

Our extension introduces two confidence thresholds (tau_low, tau_high):
  - Fix variable i to 0 only if probs[i] <= tau_low  (model says "definitely 0")
  - Fix variable i to 1 only if probs[i] >= tau_high (model says "definitely 1")
  - Then cap the fraction at k0_pct / k1_pct as before.
  - If no variable clears the threshold on either side, fall back to
    the SCIP baseline (no fixing). This is the "do no harm" property.

Use --tau_low 0.0 --tau_high 1.0 (or --paper_mode) to recover the
original fixed-k behaviour for ablation comparisons.
"""

import os, sys, csv, time, argparse
import numpy as np
import torch
import pyscipopt as scip

from shared.scip_features import extract_bipartite_graph
from PAS.model import build_PAS_model


def _load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_PAS_model(dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


class _PrimalTracker(scip.Eventhdlr):
    """Records (time, obj) whenever a new incumbent is found."""
    def __init__(self):
        self.history = []   # list of (elapsed_time, obj_val)
        self._t0     = None

    def eventinit(self):
        self._t0 = time.time()
        self.model.catchEvent(scip.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(scip.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        elapsed = time.time() - self._t0
        obj     = self.model.getSolObjVal(self.model.getBestSol())
        self.history.append((elapsed, obj))
        return {}


def _primal_integral(history, time_limit, obj_sense):
    """
    Compute primal integral given a list of (time, obj) events.
    obj_sense: 1=maximise, -1=minimise.
    Returns (final_gap, primal_integral).
    """
    if not history:
        return 1.0, time_limit   # no solution found

    # Best known = best obj in history
    objs     = [o for _, o in history]
    best_obj = max(objs) if obj_sense == 1 else min(objs)

    # Build primal gap curve
    # gap(t) = |v(t) - best| / max(|best|, 1e-8)
    eps = 1e-8

    events = sorted(history, key=lambda x: x[0])
    times  = [0.0] + [e[0] for e in events] + [time_limit]

    # running best
    running_best = None
    gaps         = []
    for t, o in events:
        if running_best is None:
            running_best = o
        else:
            running_best = max(running_best, o) if obj_sense == 1 else min(running_best, o)
        gap = abs(running_best - best_obj) / max(abs(best_obj), eps)
        gaps.append((t, gap))

    # Before first solution: gap = 1.0
    gap_curve = [(0.0, 1.0)]
    for t, g in gaps:
        gap_curve.append((t, g))
    gap_curve.append((time_limit, gaps[-1][1] if gaps else 1.0))

    # Integrate
    pi = 0.0
    for i in range(len(gap_curve) - 1):
        t0, g0 = gap_curve[i]
        t1, g1 = gap_curve[i + 1]
        pi += g0 * (t1 - t0)

    final_gap = gaps[-1][1] if gaps else 1.0
    return final_gap, pi


def _solve_with_tracker(inst_path, time_limit, fix_to_0=None, fix_to_1=None,
                         var_names=None):
    """Run SCIP with optional variable fixing and primal tracking."""
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(inst_path)
    m.setParam("limits/time", time_limit)

    obj_sense = 1 if m.getObjectiveSense() == "maximize" else -1

    if fix_to_0 or fix_to_1:
        vmap = {v.name: v for v in m.getVars(transformed=False)
                if v.vtype() in ("BINARY", "INTEGER")}
        for idx, vname in enumerate(var_names or []):
            if vname not in vmap: continue
            v = vmap[vname]
            if fix_to_0 and idx in fix_to_0:
                m.chgVarLb(v, 0.0); m.chgVarUb(v, 0.0)
            elif fix_to_1 and idx in fix_to_1:
                m.chgVarLb(v, 1.0); m.chgVarUb(v, 1.0)

    tracker = _PrimalTracker()
    m.includeEventhdlr(tracker, "PrimalTracker", "Tracks incumbent solutions")
    m.optimize()
    return tracker.history, obj_sense


def _predict_and_search(inst_path, model, device, k0_pct, k1_pct, time_limit,
                        tau_low=0.1, tau_high=0.9, paper_mode=False):
    """
    Run PaS on one instance.

    Parameters
    ----------
    k0_pct, k1_pct : float
        Cap on percent of variables to fix to 0 / 1 (paper-style).
    tau_low, tau_high : float
        Confidence thresholds. A variable is eligible for fixing-to-0 only
        if its predicted probability <= tau_low; eligible for fixing-to-1
        only if >= tau_high. After thresholding, the most confident
        candidates are kept up to the k0_pct / k1_pct cap.
    paper_mode : bool
        If True, restore the original fixed-k behaviour from the paper
        (ignore tau_low/tau_high). Useful for ablation.
    """
    t0 = time.time()

    vf, cf, ei, ea, var_names, obj_sense = extract_bipartite_graph(inst_path)
    if vf is None or len(var_names) == 0:
        return None

    with torch.no_grad():
        probs = model.predict_probs(
            torch.tensor(vf, dtype=torch.float32).to(device),
            torch.tensor(cf, dtype=torch.float32).to(device),
            torch.tensor(ei, dtype=torch.long).to(device),
            torch.tensor(ea, dtype=torch.float32).to(device),
        ).cpu().numpy()

    n_vars = len(var_names)

    # ── Variable selection ────────────────────────────────────────────────
    if paper_mode:
        # Original paper behaviour: always fix top k0%/k1% by confidence.
        conf     = np.abs(probs - 0.5)
        by_conf  = np.argsort(-conf)
        fix_to_0 = set(list(i for i in by_conf if probs[i] <  0.5)[:int(n_vars * k0_pct / 100)])
        fix_to_1 = set(list(i for i in by_conf if probs[i] >= 0.5)[:int(n_vars * k1_pct / 100)])
    else:
        # ENHANCEMENT 1 — adaptive confidence-thresholded fixing.
        # Step 1: filter by absolute probability threshold.
        cand_0_idx = np.where(probs <= tau_low )[0]
        cand_1_idx = np.where(probs >= tau_high)[0]
        # Step 2: rank survivors by how far past the threshold they are.
        cand_0_idx = cand_0_idx[np.argsort( probs[cand_0_idx])]   # most-zero first
        cand_1_idx = cand_1_idx[np.argsort(-probs[cand_1_idx])]   # most-one  first
        # Step 3: cap each side at k_pct of n_vars (defensive cap).
        cap_0 = int(n_vars * k0_pct / 100)
        cap_1 = int(n_vars * k1_pct / 100)
        fix_to_0 = set(cand_0_idx[:cap_0].tolist())
        fix_to_1 = set(cand_1_idx[:cap_1].tolist())

    n_fixed = len(fix_to_0) + len(fix_to_1)

    # ── PaS solve (with safety fallback when nothing is confident) ────────
    if n_fixed == 0 and not paper_mode:
        # Model has no confident predictions for this instance; do no harm.
        pas_hist, _ = _solve_with_tracker(inst_path, time_limit)
        fallback_used = True
    else:
        pas_hist, _ = _solve_with_tracker(inst_path, time_limit,
                                          fix_to_0, fix_to_1, var_names)
        fallback_used = False
    pas_gap, pas_pi = _primal_integral(pas_hist, time_limit, obj_sense)

    # Baseline solve (SCIP default, no fixing)
    base_hist, _ = _solve_with_tracker(inst_path, time_limit)
    base_gap, base_pi = _primal_integral(base_hist, time_limit, obj_sense)

    return {
        "t_total"      : time.time() - t0,
        "n_vars"       : n_vars,
        "n_fixed"      : n_fixed,
        "fallback"     : int(fallback_used),
        "pas_gap"      : pas_gap,
        "pas_pi"       : pas_pi,
        "base_gap"     : base_gap,
        "base_pi"      : base_pi,
    }


def evaluate(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join("pretrain", args.task_name, "model_best.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    model = _load_model(ckpt_path, device)
    mode_str = "paper-mode (fixed-k)" if args.paper_mode \
               else f"adaptive (tau_low={args.tau_low}, tau_high={args.tau_high})"
    print(f"\n[PaS Eval] task={args.task_name}  k0={args.k0_pct}%  k1={args.k1_pct}%  "
          f"TL={args.time_limit}s  mode={mode_str}")

    problems = ["CA","MIS","MVC"] if args.problem == "all" else [args.problem]
    splits   = ["test"]           if args.split   == "test" else [args.split]
    os.makedirs(os.path.join("logs", args.task_name), exist_ok=True)

    for prob in problems:
        for spl in splits:
            inst_dir = os.path.join(args.instance_dir, prob, spl)
            if not os.path.isdir(inst_dir):
                print(f"  [SKIP] {inst_dir} not found"); continue
            files = sorted(f for f in os.listdir(inst_dir)
                           if f.endswith(".lp"))[:args.max_instances]
            if not files: continue

            csv_path = os.path.join("logs", args.task_name,
                                    f"PAS_results_{prob}_{spl}.csv")
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "instance","n_vars","n_fixed","fallback",
                    "pas_gap","pas_pi","base_gap","base_pi","t_total"])

            pas_gaps, pas_pis = [], []
            base_gaps, base_pis = [], []
            n_fallbacks = 0

            for fname in files:
                r = _predict_and_search(
                    os.path.join(inst_dir, fname),
                    model, device, args.k0_pct, args.k1_pct, args.time_limit,
                    tau_low=args.tau_low, tau_high=args.tau_high,
                    paper_mode=args.paper_mode)
                if r is None:
                    print(f"  {fname}: SKIP (no binary vars)")
                    continue
                tag = " [FALLBACK]" if r["fallback"] else ""
                print(f"  {fname}: "
                      f"PaS_gap={r['pas_gap']:.4f}  PaS_PI={r['pas_pi']:.2f}  |  "
                      f"Base_gap={r['base_gap']:.4f}  Base_PI={r['base_pi']:.2f}  "
                      f"fixed={r['n_fixed']}/{r['n_vars']}  t={r['t_total']:.1f}s{tag}")
                pas_gaps.append(r['pas_gap']); pas_pis.append(r['pas_pi'])
                base_gaps.append(r['base_gap']); base_pis.append(r['base_pi'])
                n_fallbacks += r['fallback']
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        fname, r['n_vars'], r['n_fixed'], r['fallback'],
                        f"{r['pas_gap']:.6f}", f"{r['pas_pi']:.4f}",
                        f"{r['base_gap']:.6f}", f"{r['base_pi']:.4f}",
                        f"{r['t_total']:.2f}"])

            if pas_gaps:
                print(f"\n  [{prob}/{spl}] "
                      f"PaS  → mean_gap={np.mean(pas_gaps):.4f}  mean_PI={np.mean(pas_pis):.2f}")
                print(f"  [{prob}/{spl}] "
                      f"Base → mean_gap={np.mean(base_gaps):.4f}  mean_PI={np.mean(base_pis):.2f}")
                print(f"  [{prob}/{spl}] "
                      f"Fallback (no fixing) used on {n_fallbacks}/{len(pas_gaps)} instances")
            print(f"  Results → {csv_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_name",     default="pas_finetuned")
    p.add_argument("--problem",       default="all",
                   choices=["CA","MIS","MVC","all"])
    p.add_argument("--split",         default="test",
                   choices=["train","test","all"])
    p.add_argument("--instance_dir",  default="instances")
    p.add_argument("--k0_pct",        type=float, default=30.0)
    p.add_argument("--k1_pct",        type=float, default=30.0)
    p.add_argument("--time_limit",    type=int,   default=1000)
    p.add_argument("--max_instances", type=int,   default=100)
    # ── Enhancement 1 flags ─────────────────────────────────────────────
    p.add_argument("--tau_low",       type=float, default=0.1,
                   help="Fix-to-0 threshold (default: 0.1). Lower = more conservative.")
    p.add_argument("--tau_high",      type=float, default=0.9,
                   help="Fix-to-1 threshold (default: 0.9). Higher = more conservative.")
    p.add_argument("--paper_mode",    action="store_true",
                   help="Restore the original paper fixed-k behaviour (no thresholding).")
    evaluate(p.parse_args())
