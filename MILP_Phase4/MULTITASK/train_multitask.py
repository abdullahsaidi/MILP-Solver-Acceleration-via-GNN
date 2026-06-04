"""
MULTITASK/train_multitask.py
============================
Joint training of PAS + BACKDOOR + CONFIG on a single SharedEncoder.

ENHANCEMENT 2 (extension over Cai et al. CPAIOR 2025)
=====================================================
Uncertainty-Weighted Multi-Task Loss (Kendall, Gal & Cipolla, CVPR 2018)
------------------------------------------------------------------------
The paper (Section 4.2) combines per-task losses with a fixed weighted
sum: L = w_pas*L_pas + w_bd*L_bd + w_cfg*L_cfg, with w_pas=0.5,
w_bd=w_cfg=0.25 chosen by hand. This is problematic because the three
losses are on different scales (BCE for PaS vs NTXent contrastive for
Backdoor and Config), so the manual weights are essentially arbitrary.

We replace the fixed weights with learnable log-variance parameters
sigma_t per task, and train with the homoscedastic-uncertainty
weighting from Kendall et al. 2018:

   L_total = sum_t [ exp(-log_var_t) * L_t  +  log_var_t ]

The first term is the precision-weighted task loss (high uncertainty =
low weight); the second is a regularizer that prevents log_var from
diverging to infinity. The model learns its own loss balance, removing
the three magic-number hyperparameters (w_pas, w_backdoor, w_config).

Use --fixed_weights to recover the paper's behaviour for ablation.

CHANGE LOG (vs. submitted version)
------------------------------------
1. freeze_normalization() is now called each epoch on the shared encoder.
2. _append_row() helper removed — was a one-liner wrapper.
3. BACKDOOR_GATPolicy_multitask removed from BACKDOOR/GAT.py.
4. NEW: Uncertainty-weighted loss (Enhancement 2). Activated by default.

Documented deviations from the paper
--------------------------------------
* Paper Section 4: "we use three randomly initialised task-specific
  layers to enhance robustness." We keep single heads for simplicity.
* Loss weights now learned via uncertainty weighting. Pass
  --fixed_weights to use the paper's manually tuned values.
"""

import os, csv, argparse, random, time, math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning import losses as ml_losses
from pytorch_metric_learning.distances import DotProductSimilarity
import torch_geometric
from torch_scatter import scatter_mean

from shared.shared_encoder import SharedEncoder, EMB
from PAS.model import PaSHead
from PAS.MIPDataset import PAS_MultiTaskDataset, build_PAS_loader
from BACKDOOR.MIPDataset import BACKDOOR_GraphDataset
from CONFIG.MIPDataset import CONFIG_GraphDataset


class MultiTaskModel(nn.Module):
    """One SharedEncoder + three task-specific heads.

    Adds three learnable log-variance parameters (one per task) for
    uncertainty-based loss weighting (Enhancement 2). When fixed_weights
    is True, these parameters are still created but the training loop
    ignores them and uses the user-supplied static weights.
    """

    def __init__(self):
        super().__init__()
        self.encoder       = SharedEncoder()
        self.pas_head      = PaSHead(dropout=0.1)
        self.backdoor_head = nn.Sequential(
            nn.Linear(EMB, EMB), nn.ReLU(), nn.Linear(EMB, 1, bias=False)
        )
        self.config_head   = nn.Sequential(
            nn.Linear(EMB * 2, EMB), nn.ELU(), nn.Linear(EMB, 33, bias=False)
        )
        # ── ENHANCEMENT 2: learnable per-task log-variance ────────────────
        # Initialised at 0 → exp(-log_var)=1 and the regulariser term=0,
        # so the first epoch behaves like an unweighted sum. The optimiser
        # then learns the right balance.
        self.log_var_pas = nn.Parameter(torch.zeros(1))
        self.log_var_bd  = nn.Parameter(torch.zeros(1))
        self.log_var_cfg = nn.Parameter(torch.zeros(1))

    @staticmethod
    def uncertainty_weight(loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Kendall, Gal & Cipolla 2018, Eq. 10:
            L = exp(-log_var) * loss + log_var
        The first term is precision-weighted; the second prevents the
        model from inflating log_var to infinity to zero out the loss.
        """
        return torch.exp(-log_var) * loss + log_var

    def forward_pas(self, var_f, con_f, ei, ea):
        h_var, _ = self.encoder(var_f, con_f, ei, ea)
        return self.pas_head(h_var)

    def forward_backdoor(self, var_f, con_f, ei, ea):
        h_var, _ = self.encoder(var_f, con_f, ei, ea)
        return self.backdoor_head(h_var).squeeze(-1)

    def forward_config(self, var_f, con_f, ei, ea, var_batch, con_batch):
        h_var, h_con = self.encoder(var_f, con_f, ei, ea)
        return self.config_head(torch.cat([
            scatter_mean(h_var, var_batch, dim=0),
            scatter_mean(h_con, con_batch, dim=0),
        ], dim=-1))

    def freeze_normalization(self) -> bool:
        """Delegate to SharedEncoder. Returns True when fully frozen."""
        return self.encoder.freeze_normalization()


def _bce(logits, labels):
    return nn.BCEWithLogitsLoss()(logits, labels)


def _pad_collate(flat, batch_ids, pad=0.0):
    groups  = [flat[batch_ids == bid] for bid in torch.unique(batch_ids)]
    max_len = max(len(g) for g in groups)
    rows    = [torch.cat([g, torch.full((max_len - len(g),), pad,
                                        device=flat.device, dtype=flat.dtype)])
               for g in groups]
    return torch.stack(rows), max_len


def _triplet_loss(emb, ap, pi, an, ni, loss_fn, device):
    return loss_fn(emb, indices_tuple=(
        torch.tensor(ap, device=device), torch.tensor(pi, device=device),
        torch.tensor(an, device=device), torch.tensor(ni, device=device),
    ))


def _step_pas(model, batch, device):
    vf = batch["var_features"].to(device)
    cf = batch["con_features"].to(device)
    ei = batch["edge_index"].to(device)
    ea = batch["edge_attr"].to(device)
    return _bce(model.forward_pas(vf, cf, ei, ea), batch["labels"].to(device))


def _step_backdoor(model, batch, infonce, device):
    batch = batch.to(device)
    raw   = model.forward_backdoor(batch.x_vars, batch.x_cons,
                                    batch.edge_index_cons_to_vars, batch.edge_attr)
    mat, slen = _pad_collate(raw, batch.x_vars_batch)
    emb = torch.sigmoid(mat)
    ap, pi, an, ni = [], [], [], []
    nt = len(batch)
    for i in range(len(batch)):
        for vec in batch.pos_sample[i]:
            ap.append(i); pi.append(nt)
            t = torch.tensor(vec, dtype=torch.float32)
            # Inline concatenation — replaces the removed _append_row helper
            emb = torch.cat([emb, torch.cat([t, torch.zeros(slen - len(t))]).unsqueeze(0).to(device)])
            nt += 1
        for vec in batch.neg_sample[i]:
            an.append(i); ni.append(nt)
            t = torch.tensor(vec, dtype=torch.float32)
            emb = torch.cat([emb, torch.cat([t, torch.zeros(slen - len(t))]).unsqueeze(0).to(device)])
            nt += 1
    return _triplet_loss(emb, ap, pi, an, ni, infonce, device)


def _step_config(model, batch, infonce, device):
    batch = batch.to(device)
    emb   = model.forward_config(batch.x_vars, batch.x_cons,
                                  batch.edge_index_cons_to_vars, batch.edge_attr,
                                  batch.x_vars_batch, batch.x_cons_batch)
    ap, pi, an, ni = [], [], [], []
    nt = emb.shape[0]
    for i in range(len(batch)):
        for vec in batch.pos_sample[i]:
            ap.append(i); pi.append(nt)
            emb = torch.cat([emb, torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)])
            nt += 1
        for vec in batch.neg_sample[i]:
            an.append(i); ni.append(nt)
            emb = torch.cat([emb, torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)])
            nt += 1
    return _triplet_loss(emb, ap, pi, an, ni, infonce, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name",        default="multitask_v1")
    p.add_argument("--pas_dataset_dir",  default="dataset")
    p.add_argument("--pas_problems",     nargs="+", default=["CA", "MIS", "MVC"])
    p.add_argument("--bd_instance_dir",  default=None)
    p.add_argument("--bd_result_dir",    default=None)
    p.add_argument("--cfg_instance_dir", default=None)
    p.add_argument("--cfg_result_dir",   default=None)
    p.add_argument("--pretrain_model",   default=None)
    p.add_argument("--w_pas",            type=float, default=0.5)
    p.add_argument("--w_backdoor",       type=float, default=0.25)
    p.add_argument("--w_config",         type=float, default=0.25)
    p.add_argument("--fixed_weights",    action="store_true",
                   help="Use the paper's fixed weights instead of learned uncertainty weighting (Enhancement 2).")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--temp",             type=float, default=0.07)
    p.add_argument("--num_workers",      type=int,   default=0)
    args = p.parse_args()

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join("pretrain",   args.task_name)
    log_dir = os.path.join("train_logs", args.task_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.csv")
    print(f"\nMulti-Task Training | {args.task_name} | {DEVICE}\n")

    # --- Data loaders ---
    pas_tr = PAS_MultiTaskDataset(args.pas_problems, "train", args.pas_dataset_dir)
    pas_va = PAS_MultiTaskDataset(args.pas_problems, "test",  args.pas_dataset_dir)
    pas_tr_loader = build_PAS_loader(pas_tr, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    pas_va_loader = build_PAS_loader(pas_va, args.batch_size, shuffle=False, num_workers=args.num_workers)

    bd_tr_loader = bd_va_loader = None
    if args.bd_instance_dir and args.bd_result_dir:
        names = os.listdir(args.bd_result_dir)
        pairs = sorted([(os.path.join(args.bd_instance_dir, n),
                         os.path.join(args.bd_result_dir,   n)) for n in names])
        random.seed(0); random.shuffle(pairs)
        nv = int(0.2 * len(pairs))
        kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  follow_batch=["x_vars"])
        bd_tr_loader = torch_geometric.loader.DataLoader(
            BACKDOOR_GraphDataset(pairs[nv:]), shuffle=True,  **kw)
        bd_va_loader = torch_geometric.loader.DataLoader(
            BACKDOOR_GraphDataset(pairs[:nv]), shuffle=False, **kw)

    cfg_tr_loader = cfg_va_loader = None
    if args.cfg_instance_dir and args.cfg_result_dir:
        csv_files = [f for f in os.listdir(args.cfg_result_dir) if f.endswith(".csv")]
        cfg_pairs = sorted([
            (os.path.join(args.cfg_instance_dir, f[:-4] + ".lp"),
             os.path.join(args.cfg_result_dir,   f))
            for f in csv_files
            if os.path.exists(os.path.join(args.cfg_instance_dir, f[:-4] + ".lp"))
        ])
        nv  = int(0.2 * len(cfg_pairs))
        kw2 = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   follow_batch=["x_vars", "x_cons"])
        cfg_tr_loader = torch_geometric.loader.DataLoader(
            CONFIG_GraphDataset(cfg_pairs[nv:]),  shuffle=True,  **kw2)
        cfg_va_loader = torch_geometric.loader.DataLoader(
            CONFIG_GraphDataset(cfg_pairs[:nv]), shuffle=False, **kw2)

    # --- Model ---
    model = MultiTaskModel().to(DEVICE)
    if args.pretrain_model:
        ckpt = torch.load(args.pretrain_model, map_location=DEVICE)
        sd   = ckpt.get("model_state", ckpt)
        model.encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in sd.items()
             if k.startswith("encoder.")}, strict=False)
        print(f"Loaded encoder from {args.pretrain_model}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    infonce   = ml_losses.NTXentLoss(
        temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)
    best_val  = float("inf")

    weight_mode = "fixed (paper)" if args.fixed_weights else "uncertainty (Enhancement 2)"
    print(f"Loss-weighting mode: {weight_mode}")

    # ── Helper: build the per-task loss term for one mini-batch ──────────
    # In fixed mode, multiply by the user-supplied weight.
    # In uncertainty mode, wrap with exp(-log_var)*L + log_var.
    def _weight(loss_term, log_var, fixed_w):
        if args.fixed_weights:
            return fixed_w * loss_term
        return MultiTaskModel.uncertainty_weight(loss_term, log_var)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "pas_tr", "pas_va", "bd_tr", "bd_va",
             "cfg_tr", "cfg_va", "total_va",
             "log_var_pas", "log_var_bd", "log_var_cfg", "lr"])

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        pas_tr_loss = bd_tr_loss = cfg_tr_loss = 0.0
        pas_n = bd_n = cfg_n = 0

        for batch in pas_tr_loader:
            optimizer.zero_grad()
            raw_loss = _step_pas(model, batch, DEVICE)
            loss = _weight(raw_loss, model.log_var_pas, args.w_pas)
            loss.backward(); optimizer.step()
            pas_tr_loss += raw_loss.item(); pas_n += 1   # log raw loss for comparability

        if bd_tr_loader:
            for batch in bd_tr_loader:
                optimizer.zero_grad()
                raw_loss = _step_backdoor(model, batch, infonce, DEVICE)
                loss = _weight(raw_loss, model.log_var_bd, args.w_backdoor)
                loss.backward(); optimizer.step()
                bd_tr_loss += raw_loss.item(); bd_n += 1

        if cfg_tr_loader:
            for batch in cfg_tr_loader:
                optimizer.zero_grad()
                raw_loss = _step_config(model, batch, infonce, DEVICE)
                loss = _weight(raw_loss, model.log_var_cfg, args.w_config)
                loss.backward(); optimizer.step()
                cfg_tr_loss += raw_loss.item(); cfg_n += 1

        # ── Call freeze_normalization each epoch ──────────────────────
        model.freeze_normalization()

        model.eval()
        with torch.no_grad():
            pas_va_loss = (sum(_step_pas(model, b, DEVICE).item()
                               for b in pas_va_loader) / max(len(pas_va_loader), 1))
            bd_va_loss  = (sum(_step_backdoor(model, b, infonce, DEVICE).item()
                               for b in bd_va_loader) / max(len(bd_va_loader), 1)
                           if bd_va_loader else 0.0)
            cfg_va_loss = (sum(_step_config(model, b, infonce, DEVICE).item()
                               for b in cfg_va_loader) / max(len(cfg_va_loader), 1)
                           if cfg_va_loader else 0.0)

        # Snapshot the learned log-variances for logging
        lv_pas = model.log_var_pas.item()
        lv_bd  = model.log_var_bd.item()
        lv_cfg = model.log_var_cfg.item()

        # Validation total: in uncertainty mode, use exp(-log_var) precisions
        # (no log_var regulariser since we're not training); in fixed mode,
        # use the user-supplied weights as before.
        if args.fixed_weights:
            total_va = (args.w_pas * pas_va_loss
                        + args.w_backdoor * bd_va_loss
                        + args.w_config   * cfg_va_loss)
        else:
            total_va = (math.exp(-lv_pas) * pas_va_loss
                        + math.exp(-lv_bd)  * bd_va_loss
                        + math.exp(-lv_cfg) * cfg_va_loss)

        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch+1:04d}  "
              f"PaS={pas_tr_loss/max(pas_n,1):.4f}|{pas_va_loss:.4f}  "
              f"BD={bd_tr_loss/max(bd_n,1):.4f}|{bd_va_loss:.4f}  "
              f"CFG={cfg_tr_loss/max(cfg_n,1):.4f}|{cfg_va_loss:.4f}  "
              f"lvar=[{lv_pas:+.3f},{lv_bd:+.3f},{lv_cfg:+.3f}]  "
              f"total_val={total_va:.4f}  {time.time()-t0:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1,
                pas_tr_loss/max(pas_n,1), pas_va_loss,
                bd_tr_loss/max(bd_n,1),   bd_va_loss,
                cfg_tr_loss/max(cfg_n,1), cfg_va_loss,
                total_va, lv_pas, lv_bd, lv_cfg, lr_now,
            ])

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "args": vars(args)}
        if total_va < best_val:
            best_val = total_va
            torch.save(ckpt, os.path.join(out_dir, "model_best.pt"))
            print(f"  Best saved (total_val={best_val:.4f})")
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(out_dir, "model_final.pt"))

    torch.save(ckpt, os.path.join(out_dir, "model_final.pt"))
    print(f"\nMulti-task training complete. Best val={best_val:.4f}")
    if not args.fixed_weights:
        print(f"Final learned log-variances: "
              f"pas={model.log_var_pas.item():+.4f}  "
              f"bd={model.log_var_bd.item():+.4f}  "
              f"cfg={model.log_var_cfg.item():+.4f}")


if __name__ == "__main__":
    main()
