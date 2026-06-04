import os, argparse
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_lp(filepath, lines):
    with open(filepath, "w") as f:
        f.writelines(lines)


def generate_ca_instance(n_items=100, n_bids=500, max_bundle_size=5,
                          add_dummy_bids=True, seed=0, filepath=None):
    rng   = np.random.RandomState(seed)
    items = list(range(n_items))
    bids, bids_per_item = [], [[] for _ in range(n_items)]

    for b in range(n_bids):
        bundle = list(rng.choice(items, size=rng.randint(1, max_bundle_size + 1), replace=False))
        price  = rng.uniform(1, 100) * len(bundle)
        bids.append((bundle, price))
        for item in bundle: bids_per_item[item].append(b)

    if add_dummy_bids:
        for item in items:
            if not bids_per_item[item]:
                bids.append(([item], rng.uniform(1, 20)))
                bids_per_item[item].append(len(bids) - 1)

    n_total = len(bids)
    lines   = ["\\Problem name: CA\n", "Maximize\n", " obj:"]
    lines.append(" ".join(f" {'+ ' if b>0 else ''}{price:.4f} x_{b}" for b, (_, price) in enumerate(bids)))
    lines.append("\nSubject To\n")
    for i in range(n_items):
        if len(bids_per_item[i]) > 1:
            lines.append(f" c_{i}: {' + '.join(f'x_{b}' for b in bids_per_item[i])} <= 1\n")
    lines.append("Binary\n")
    for b in range(n_total): lines.append(f" x_{b}\n")
    lines.append("End\n")
    if filepath: _write_lp(filepath, lines)
    return filepath


def _erdos_renyi(n_nodes, density, seed):
    rng   = np.random.RandomState(seed)
    edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes) if rng.random() < density]
    return edges


def generate_mis_instance(n_nodes=100, density=0.25, seed=0, filepath=None):
    edges = _erdos_renyi(n_nodes, density, seed)
    lines = ["\\Problem name: MIS\n", "Maximize\n", " obj:"]
    lines.append(" ".join(f" {'+ ' if i>0 else ''}x_{i}" for i in range(n_nodes)))
    lines.append("\nSubject To\n")
    for idx, (i, j) in enumerate(edges): lines.append(f" e_{idx}: x_{i} + x_{j} <= 1\n")
    lines.append("Binary\n")
    for i in range(n_nodes): lines.append(f" x_{i}\n")
    lines.append("End\n")
    if filepath: _write_lp(filepath, lines)
    return filepath


def generate_mvc_instance(n_nodes=100, density=0.25, seed=0, filepath=None):
    edges = _erdos_renyi(n_nodes, density, seed)
    lines = ["\\Problem name: MVC\n", "Minimize\n", " obj:"]
    lines.append(" ".join(f" {'+ ' if i>0 else ''}x_{i}" for i in range(n_nodes)))
    lines.append("\nSubject To\n")
    for idx, (i, j) in enumerate(edges): lines.append(f" e_{idx}: x_{i} + x_{j} >= 1\n")
    lines.append("Binary\n")
    for i in range(n_nodes): lines.append(f" x_{i}\n")
    lines.append("End\n")
    if filepath: _write_lp(filepath, lines)
    return filepath


PROBLEM_CONFIGS = {
    "CA":  {
        "train": {"n_instances": 200, "n_items": 100, "n_bids": 500,  "seed_offset": 0},
        "test":  {"n_instances": 100, "n_items": 120, "n_bids": 600,  "seed_offset": 10000},
        "generator": generate_ca_instance,
        "params": {"max_bundle_size": 5, "add_dummy_bids": True},
    },
    "MIS": {
        "train": {"n_instances": 200, "n_nodes": 100, "density": 0.25, "seed_offset": 0},
        "test":  {"n_instances": 100, "n_nodes": 120, "density": 0.25, "seed_offset": 10000},
        "generator": generate_mis_instance, "params": {},
    },
    "MVC": {
        "train": {"n_instances": 200, "n_nodes": 100, "density": 0.25, "seed_offset": 0},
        "test":  {"n_instances": 100, "n_nodes": 120, "density": 0.25, "seed_offset": 10000},
        "generator": generate_mvc_instance, "params": {},
    },
}


def generate_all(base_dir="instances", problems=None):
    if problems is None: problems = ["CA", "MIS", "MVC"]
    for prob in problems:
        cfg = PROBLEM_CONFIGS[prob]
        gen, extra = cfg["generator"], cfg["params"]
        for split in ["train", "test"]:
            sc      = cfg[split]
            out_dir = os.path.join(base_dir, prob, split)
            ensure_dir(out_dir)
            print(f"\n[{prob}/{split}] Generating {sc['n_instances']} instances -> {out_dir}")
            for i in range(sc["n_instances"]):
                fp = os.path.join(out_dir, f"{prob}_{split}_{i:04d}.lp")
                if os.path.exists(fp): continue
                seed = sc["seed_offset"] + i
                if prob == "CA":
                    gen(n_items=sc["n_items"], n_bids=sc["n_bids"], seed=seed, filepath=fp, **extra)
                else:
                    gen(n_nodes=sc["n_nodes"], density=sc["density"], seed=seed, filepath=fp, **extra)
                if (i + 1) % 50 == 0: print(f"  {i+1}/{sc['n_instances']}")
    print("\nAll instances generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="instances")
    parser.add_argument("--problems", nargs="+", default=["CA","MIS","MVC"])
    args = parser.parse_args()
    generate_all(base_dir=args.base_dir, problems=args.problems)
