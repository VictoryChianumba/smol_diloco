#!/usr/bin/env python3
"""Turn the K x outer_lr grid sweep into figures.

Reads experiments/grid/k{K}_lr{LR}/{config.json, metrics.jsonl} and produces:
  - grid_heatmap.png     final val loss over the (sync interval, outer LR) grid
  - sensitivity.png      final val loss vs outer LR, one line per sync interval
                         (shows the best outer LR shifts with K)
  - frontier_tuned.png   communication volume vs final loss at the BEST outer LR
                         per sync interval -- the clean efficiency frontier

Usage: python experiments/plot_grid.py [--grid experiments/grid] [--out experiments/figures]
"""
import argparse, json, glob, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    return cfg, rows


def final_val(rows):
    ev = [r for r in rows if r.get("val_loss") is not None]
    return ev[-1]["val_loss"] if ev else float("nan")


def load_grid(grid_dir):
    """Return {(K, lr): {'final': loss, 'comm_gb': gb, 'rows': rows}}."""
    out = {}
    for run_dir in glob.glob(os.path.join(grid_dir, "k*_lr*")):
        if not os.path.isfile(os.path.join(run_dir, "metrics.jsonl")):
            continue
        cfg, rows = load_run(run_dir)
        key = (cfg["local_steps"], cfg["outer_lr"])
        out[key] = {"final": final_val(rows), "comm_gb": rows[-1]["comm_bytes"] / 1e9, "rows": rows}
    return out


def axes(grid):
    Ks = sorted({k for k, _ in grid})
    lrs = sorted({lr for _, lr in grid})
    return Ks, lrs


def plot_heatmap(grid, out):
    Ks, lrs = axes(grid)
    M = np.array([[grid.get((k, lr), {}).get("final", np.nan) for lr in lrs] for k in Ks])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(lrs)), [str(lr) for lr in lrs])
    ax.set_yticks(range(len(Ks)), [str(k) for k in Ks])
    ax.set_xlabel("outer learning rate")
    ax.set_ylabel("sync interval K")
    ax.set_title("Final validation loss over (K, outer LR)")
    for i in range(len(Ks)):
        for j in range(len(lrs)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > np.nanmean(M) else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="final val loss")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close()


def plot_sensitivity(grid, out):
    Ks, lrs = axes(grid)
    plt.figure(figsize=(6.5, 4.5))
    for k in Ks:
        ys = [grid.get((k, lr), {}).get("final", np.nan) for lr in lrs]
        plt.plot(lrs, ys, marker="o", label=f"K={k}")
    plt.xlabel("outer learning rate")
    plt.ylabel("final validation loss")
    plt.title("Outer-LR sensitivity depends on sync interval")
    plt.legend(title="sync interval")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_frontier_tuned(grid, out):
    """Best outer LR per K: communication volume vs achievable loss."""
    Ks, _ = axes(grid)
    best_loss, gb, best_lr = [], [], []
    for k in Ks:
        cands = [(grid[(k, lr)]["final"], lr, grid[(k, lr)]["comm_gb"])
                 for (kk, lr) in grid if kk == k]
        loss, lr, g = min(cands, key=lambda t: t[0])
        best_loss.append(loss); best_lr.append(lr); gb.append(g)

    fig, ax1 = plt.subplots(figsize=(6.8, 4.5))
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("sync interval K")
    ax1.set_ylabel("best final val loss (tuned outer LR)", color="tab:blue")
    ax1.plot(Ks, best_loss, "o-", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    # widen the loss axis so the near-flat curve isn't visually exaggerated
    ax1.set_ylim(min(best_loss) - 0.6, max(best_loss) + 0.3)
    ax1.set_xticks(Ks); ax1.set_xticklabels([str(k) for k in Ks])
    for k, l, lr in zip(Ks, best_loss, best_lr):
        ax1.annotate(f"lr={lr}", (k, l), textcoords="offset points", xytext=(0, 8), fontsize=8, color="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("communication volume (GB)", color="tab:red")
    ax2.plot(Ks, gb, "s--", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Tuned communication vs convergence frontier")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close()


def print_table(grid):
    Ks, _ = axes(grid)
    print("\n**Best outer LR per sync interval (tuned frontier):**\n")
    print("| Sync interval K | Best outer LR | Final val loss | Comm (GB) |")
    print("|---:|---:|---:|---:|")
    for k in Ks:
        cands = [(grid[(k, lr)]["final"], lr, grid[(k, lr)]["comm_gb"]) for (kk, lr) in grid if kk == k]
        loss, lr, g = min(cands, key=lambda t: t[0])
        print(f"| {k} | {lr} | {loss:.3f} | {g:.3f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="experiments/grid")
    ap.add_argument("--out", default="experiments/figures")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    if not grid:
        raise SystemExit(f"no grid runs found under {args.grid}")
    os.makedirs(args.out, exist_ok=True)

    plot_heatmap(grid, os.path.join(args.out, "grid_heatmap.png"))
    plot_sensitivity(grid, os.path.join(args.out, "sensitivity.png"))
    plot_frontier_tuned(grid, os.path.join(args.out, "frontier_tuned.png"))
    print_table(grid)
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
