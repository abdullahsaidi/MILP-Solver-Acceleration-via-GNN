import os, csv, argparse, time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from PAS.model import build_PAS_model
from PAS.MIPDataset import PAS_Dataset, PAS_MultiTaskDataset, build_PAS_loader


def _loss(logits, labels):
    return nn.BCEWithLogitsLoss()(logits, labels)


def _train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        vf, cf = batch["var_features"].to(device), batch["con_features"].to(device)
        ei, ea = batch["edge_index"].to(device), batch["edge_attr"].to(device)
        lbl    = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = _loss(model(vf, cf, ei, ea), lbl)
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARN] NaN/Inf loss — skipping batch")
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        vf, cf = batch["var_features"].to(device), batch["con_features"].to(device)
        ei, ea = batch["edge_index"].to(device), batch["edge_attr"].to(device)
        lbl    = batch["labels"].to(device)
        total += _loss(model(vf, cf, ei, ea), lbl).item(); n += 1
    return total / max(n, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPaS Training | task={args.task_name} | device={device}\n")

    if len(args.problems) == 1:
        tr_ds = PAS_Dataset(args.problems[0], "train", args.dataset_dir)
        va_ds = PAS_Dataset(args.problems[0], "test",  args.dataset_dir)
    else:
        tr_ds = PAS_MultiTaskDataset(args.problems, "train", args.dataset_dir)
        va_ds = PAS_MultiTaskDataset(args.problems, "test",  args.dataset_dir)

    tr_loader = build_PAS_loader(tr_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    va_loader = build_PAS_loader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_PAS_model(dropout=args.dropout).to(device)

    # ── Load pretrained multitask encoder (for fine-tuning) ──────────────────
    if args.pretrain_model and os.path.exists(args.pretrain_model):
        ckpt_pt = torch.load(args.pretrain_model, map_location=device,
                             weights_only=False)
        sd = ckpt_pt.get("model_state", ckpt_pt)
        # Load only encoder weights — ignore head mismatches
        encoder_sd = {k.replace("encoder.", ""): v
                      for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"Loaded encoder from {args.pretrain_model}")
        if missing:    print(f"  Missing keys : {missing}")
        if unexpected: print(f"  Unexpected   : {unexpected}")

    # ── Freeze encoder if requested (paper Step 2: fine-tune head only) ──────
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — fine-tuning head only\n")
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)

    scheduler   = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch = 0
    best_val    = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} (epoch {start_epoch})\n")

    out_dir  = os.path.join("pretrain", args.task_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(start_epoch, args.epochs):
        tr_loss = _train_epoch(model, tr_loader, optimizer, device)
        va_loss = _eval_epoch(model, va_loader, device)
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:04d}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}  lr={lr_now:.2e}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, tr_loss, va_loss, lr_now])

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val, "args": vars(args)}
        if va_loss < best_val:
            best_val = va_loss
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, os.path.join(out_dir, "model_best.pt"))
            print(f"  Best saved (val={best_val:.4f})")
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(out_dir, "model_final.pt"))

    torch.save(ckpt, os.path.join(out_dir, "model_final.pt"))
    print(f"\nDone. Best val_loss={best_val:.4f}  ckpts -> {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--problems",        nargs="+", default=["CA","MIS","MVC"])
    p.add_argument("--task_name",       default="pas_finetuned")
    p.add_argument("--dataset_dir",     default="dataset")
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--resume",          default=None)
    p.add_argument("--pretrain_model",  default=None,
                   help="Path to multitask checkpoint to load encoder from")
    p.add_argument("--freeze_encoder",  action="store_true",
                   help="Freeze encoder and fine-tune head only (paper Step 2)")
    train(p.parse_args())
