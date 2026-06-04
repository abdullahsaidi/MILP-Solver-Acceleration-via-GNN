import os, random, time, argparse
import torch
import torch_geometric
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import DotProductSimilarity
from torch.utils.tensorboard import SummaryWriter

from CONFIG.GAT import CONFIG_GATPolicy as GATPolicy
from CONFIG.MIPDataset import CONFIG_GraphDataset as GraphDataset

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def run_epoch(model, loader, optimizer, infonce, device):
    model.train() if optimizer else model.eval()
    total, n = 0.0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = batch.to(device)
            emb   = model(batch.x_cons, batch.edge_index_cons_to_vars, batch.edge_attr,
                          batch.x_vars, batch.x_vars_batch, batch.x_cons_batch)

            ap, pi, an, ni = [], [], [], []
            nt = emb.shape[0]
            for i in range(len(batch)):
                for vec in batch.pos_sample[i]:
                    ap.append(i); pi.append(nt)
                    emb = torch.cat([emb, torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)]); nt += 1
                for vec in batch.neg_sample[i]:
                    an.append(i); ni.append(nt)
                    emb = torch.cat([emb, torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)]); nt += 1

            triplets = (torch.tensor(ap, dtype=torch.long, device=device),
                        torch.tensor(pi, dtype=torch.long, device=device),
                        torch.tensor(an, dtype=torch.long, device=device),
                        torch.tensor(ni, dtype=torch.long, device=device))
            loss = infonce(emb, indices_tuple=triplets)
            if optimizer:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); n += len(batch)

    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--taskName",        required=True)
    p.add_argument("--instance_dir",    required=True)
    p.add_argument("--result_dir",      required=True)
    p.add_argument("--pretrain_path",   default=None)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--patience",        type=int,   default=50)
    p.add_argument("--max_epochs",      type=int,   default=1000)
    p.add_argument("--temp",            type=float, default=0.07)
    p.add_argument("--fracdata",        type=float, default=1.0)
    args = p.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR     = args.lr or (2e-5 if args.pretrain_path else 1e-4)

    model_dir = os.path.join("pretrain",   args.taskName)
    log_dir   = os.path.join("train_logs", args.taskName)
    os.makedirs(model_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    writer   = SummaryWriter(log_dir)
    log_file = open(os.path.join(log_dir, "train.log"), "w")

    csv_files    = [f for f in os.listdir(args.result_dir) if f.endswith(".csv")]
    sample_files = [(os.path.join(args.instance_dir, f[:-4]), os.path.join(args.result_dir, f))
                    for f in csv_files if os.path.exists(os.path.join(args.instance_dir, f[:-4]))]
    if not sample_files:
        raise FileNotFoundError(f"No (instance, CSV) pairs found in {args.instance_dir} / {args.result_dir}")

    sample_files = sorted(sample_files)
    random.seed(348); random.shuffle(sample_files)
    sample_files = sample_files[:max(1, int(len(sample_files) * args.fracdata))]
    nv   = max(1, int(0.20 * len(sample_files)))
    kw   = dict(batch_size=args.batch_size, num_workers=0, follow_batch=["x_vars","x_cons"])
    trn_loader = torch_geometric.loader.DataLoader(GraphDataset(sample_files[nv:]), shuffle=True,  **kw)
    val_loader = torch_geometric.loader.DataLoader(GraphDataset(sample_files[:nv]), shuffle=False, **kw)
    print(f"Train: {len(sample_files)-nv}  Valid: {nv}")

    model = GATPolicy().to(DEVICE)
    if args.pretrain_path:
        ckpt = torch.load(args.pretrain_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"Loaded backbone from {args.pretrain_path}")
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "encoder" in name: param.requires_grad = False
        print(f"Frozen encoder, training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    infonce   = losses.NTXentLoss(temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    best_val, patience_n = float("inf"), 0

    for epoch in range(args.max_epochs):
        t0 = time.time()
        tr = run_epoch(model, trn_loader, optimizer, infonce, DEVICE)
        va = run_epoch(model, val_loader, None,      infonce, DEVICE)
        line = f"Epoch {epoch:4d} | train {tr:.5f} | valid {va:.5f} | {time.time()-t0:.1f}s"
        print(line); log_file.write(line + "\n"); log_file.flush()
        writer.add_scalars("CONFIG_loss", {"train": tr, "valid": va}, epoch)
        torch.save(model.state_dict(), os.path.join(model_dir, "model_last.pth"))

        if va < best_val:
            best_val, patience_n = va, 0
            torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pth"))
            print(f"  Saved best (val={best_val:.5f})")
        else:
            patience_n += 1
            if patience_n >= args.patience:
                print(f"Early stop at epoch {epoch}"); break

    writer.flush(); writer.close(); log_file.close()
    print(f"Done. Best -> {model_dir}/model_best.pth")


if __name__ == "__main__":
    main()
