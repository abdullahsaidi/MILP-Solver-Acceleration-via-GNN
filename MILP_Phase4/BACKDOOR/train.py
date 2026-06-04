import os, random, time, argparse
import numpy as np
import torch
import torch_geometric
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import DotProductSimilarity
from torch.utils.tensorboard import SummaryWriter

from BACKDOOR.GAT import BACKDOOR_GATPolicy as GATPolicy
from BACKDOOR.MIPDataset import BACKDOOR_GraphDataset as GraphDataset

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def _pad_collate(flat, batch_ids, pad=0.0):
    groups  = [flat[batch_ids == bid] for bid in torch.unique(batch_ids)]
    max_len = max(len(g) for g in groups)
    rows    = []
    for g in groups:
        p = torch.full((max_len - len(g),), pad, device=flat.device, dtype=flat.dtype)
        rows.append(torch.cat([g, p]))
    return torch.stack(rows), max_len


def _append_vec(emb, vec, tlen, device):
    t = torch.tensor(vec, dtype=torch.float32)
    return torch.cat([emb, torch.cat([t, torch.zeros(tlen - len(t))]).unsqueeze(0).to(device)])


def _run_epoch(epoch, model, loader, loss_fn, writer, optimizer=None, device="cpu"):
    model.train(optimizer is not None)
    tag = "BD_train" if optimizer else "BD_valid"
    total, n = 0.0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = batch.to(device)
            raw   = model(batch.x_cons, batch.edge_index_cons_to_vars,
                          batch.edge_attr, batch.x_vars)
            mat, slen = _pad_collate(raw, batch.x_vars_batch)
            emb = torch.sigmoid(mat)

            ap, pi, an, ni = [], [], [], []
            nt = len(batch)
            for i in range(len(batch)):
                for vec in batch.pos_sample[i]:
                    ap.append(i); pi.append(nt); emb = _append_vec(emb, vec, slen, device); nt += 1
                for vec in batch.neg_sample[i]:
                    an.append(i); ni.append(nt); emb = _append_vec(emb, vec, slen, device); nt += 1

            triplets = (torch.tensor(ap, device=device), torch.tensor(pi, device=device),
                        torch.tensor(an, device=device), torch.tensor(ni, device=device))
            loss = loss_fn(emb, indices_tuple=triplets)
            writer.add_scalar(tag, loss.item(), epoch)

            if optimizer:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); n += len(batch)

    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--taskName",      default="BACKDOOR_IP")
    p.add_argument("--pretrainModel", default=None)
    p.add_argument("--temp",          type=float, default=0.07)
    p.add_argument("--fracdata",      type=float, default=1.0)
    p.add_argument("--freeze",        type=int,   default=0)
    p.add_argument("--instance_dir",  required=True)
    p.add_argument("--result_dir",    required=True)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-5)
    p.add_argument("--max_epochs",    type=int,   default=9999)
    args = p.parse_args()

    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = f"pretrain/{args.taskName}/"
    log_dir   = f"train_logs/{args.taskName}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    writer  = SummaryWriter(log_dir)
    log_f   = open(os.path.join(log_dir, f"{args.taskName}_train.log"), "w")
    loss_fn = losses.NTXentLoss(temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)

    names = os.listdir(args.result_dir)
    pairs = sorted([(os.path.join(args.instance_dir, n), os.path.join(args.result_dir, n)) for n in names])
    random.seed(0); random.shuffle(pairs)
    pairs = pairs[:int(len(pairs) * args.fracdata)]
    nv    = int(0.20 * len(pairs))
    kw    = dict(batch_size=args.batch_size, num_workers=0, follow_batch=["x_vars"])
    trn_loader = torch_geometric.loader.DataLoader(GraphDataset(pairs[nv:]), shuffle=True,  **kw)
    val_loader = torch_geometric.loader.DataLoader(GraphDataset(pairs[:nv]), shuffle=False, **kw)
    print(f"Train: {len(pairs)-nv}  Valid: {nv}")

    model = GATPolicy().to(DEVICE)
    if args.pretrainModel:
        ckpt = torch.load(args.pretrainModel, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"Loaded pretrain from {args.pretrainModel}")
        if args.freeze == 1:
            for name, param in model.named_parameters():
                if "encoder" in name: param.requires_grad = False
            print("Frozen: encoder layers")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_val  = float("inf")

    for epoch in range(args.max_epochs):
        t0 = time.time()
        tr = _run_epoch(epoch, model, trn_loader, loss_fn, writer, optimizer, DEVICE)
        va = _run_epoch(epoch, model, val_loader,  loss_fn, writer, device=DEVICE)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}  train={tr:.4f}  valid={va:.4f}  {elapsed:.1f}s")
        log_f.write(f"@epoch{epoch}  train={tr:.6f}  valid={va:.6f}  time={elapsed:.2f}s\n")
        log_f.flush()
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pth"))
        torch.save(model.state_dict(), os.path.join(model_dir, "model_last.pth"))

    writer.close(); log_f.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
