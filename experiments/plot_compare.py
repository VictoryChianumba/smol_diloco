#!/usr/bin/env python3
"""Compare two K × outer_lr grids (e.g. IID vs non-IID).

Reads grids from two directories (each with k{K}_lr{LR}/ subdirs containing
metrics.jsonl + config.json) and produces:
  - grid_compare.png    side-by-side heatmaps (shared color scale) + a delta
                        panel (B − A), so the impact of the regime change is visible
  - frontier_compare.png  tuned best-loss-per-K for each grid on the same axes

Also prints a markdown table comparing best outer LR per K across the two grids.

Usage:
  python experiments/plot_compare.py \
      --grid-a experiments/grid --label-a IID \
      --grid-b experiments/grid_noniid --label-b "non-IID" \
      --out experiments/figures
"""
import argparse, json, glob, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _final_val(rows):
    ev = [r for r in rows if r.get("val_loss") is not None]
    return ev[-1]["val_loss"] if ev else float("nan")


def load_grid(grid_dir):
    g = {}
    for d in glob.glob(os.path.join(grid_dir, "k*_lr*")):
        cfg = json.load(open(os.path.join(d, "config.json")))
        rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
        g[(cfg["local_steps"], cfg["outer_lr"])] = {
            "final": _final_val(rows),
            "comm_gb": rows[-1]["comm_bytes"] / 1e9,
        }
    return g


def axes(g):
    return sorted({k for k, _ in g}), sorted({lr for _, lr in g})


def matrix(g, Ks, lrs, field="final"):
    return np.array([[g.get((k, lr), {}).get(field, np.nan) for lr in lrs] for k in Ks])


def _annotate(ax, M, threshold=None):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if (threshold is not None and v > threshold) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=9)


def plot_grid_compare(gA, gB, label_a, label_b, out):
    Ks, lrs = axes(gA)
    A = matrix(gA, Ks, lrs)
    B = matrix(gB, Ks, lrs)
    D = B - A

    vmin = min(np.nanmin(A), np.nanmin(B))
    vmax = max(np.nanmax(A), np.nanmax(B))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, M, title in [(axs[0], A, f"{label_a} (final val loss)"),
                         (axs[1], B, f"{label_b} (final val loss)")]:
        im = ax.imshow(M, cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(lrs)), [str(lr) for lr in lrs])
        ax.set_yticks(range(len(Ks)), [str(k) for k in Ks])
        ax.set_xlabel("outer learning rate")
        ax.set_ylabel("sync interval K")
        ax.set_title(title)
        _annotate(ax, M, threshold=(vmin + vmax) / 2)
    fig.colorbar(im, ax=axs[:2], shrink=0.85, label="final val loss")

    # delta panel
    dmax = np.nanmax(np.abs(D))
    im2 = axs[2].imshow(D, cmap="RdBu_r", aspect="auto", vmin=-dmax, vmax=dmax)
    axs[2].set_xticks(range(len(lrs)), [str(lr) for lr in lrs])
    axs[2].set_yticks(range(len(Ks)), [str(k) for k in Ks])
    axs[2].set_xlabel("outer learning rate")
    axs[2].set_ylabel("sync interval K")
    axs[2].set_title(f"Δ ({label_b} − {label_a})")
    _annotate(axs[2], D, threshold=0)
    fig.colorbar(im2, ax=axs[2], shrink=0.85, label=f"{label_b} − {label_a}")

    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def best_per_K(g):
    Ks, _ = axes(g)
    out = {}
    for k in Ks:
        cands = [(g[(kk, lr)]["final"], lr, g[(kk, lr)]["comm_gb"]) for (kk, lr) in g if kk == k]
        loss, lr, gb = min(cands, key=lambda t: t[0])
        out[k] = (loss, lr, gb)
    return out


def plot_frontier_compare(gA, gB, label_a, label_b, out):
    bA = best_per_K(gA); bB = best_per_K(gB)
    Ks = sorted(bA)
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(Ks, [bA[k][0] for k in Ks], "o-", label=f"{label_a} (best per K)")
    plt.plot(Ks, [bB[k][0] for k in Ks], "s--", label=f"{label_b} (best per K)")
    plt.xscale("log", base=2)
    plt.xticks(Ks, [str(k) for k in Ks])
    plt.xlabel("sync interval K")
    plt.ylabel("best final val loss across outer LRs")
    plt.title("Tuned frontier: IID vs non-IID")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def print_table(gA, gB, label_a, label_b):
    bA = best_per_K(gA); bB = best_per_K(gB)
    Ks = sorted(bA)
    print(f"\n| K | {label_a} best lr | {label_a} loss | {label_b} best lr | {label_b} loss | Δ |")
    print("|---:|---:|---:|---:|---:|---:|")
    for k in Ks:
        lossA, lrA, _ = bA[k]; lossB, lrB, _ = bB[k]
        print(f"| {k} | {lrA} | {lossA:.3f} | {lrB} | {lossB:.3f} | {lossB-lossA:+.3f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-a", default="experiments/grid")
    ap.add_argument("--grid-b", default="experiments/grid_noniid")
    ap.add_argument("--label-a", default="IID")
    ap.add_argument("--label-b", default="non-IID")
    ap.add_argument("--out", default="experiments/figures")
    args = ap.parse_args()

    gA = load_grid(args.grid_a); gB = load_grid(args.grid_b)
    if not gA or not gB:
        raise SystemExit(f"empty grid: a={len(gA)} b={len(gB)}")
    os.makedirs(args.out, exist_ok=True)
    plot_grid_compare(gA, gB, args.label_a, args.label_b,
                      os.path.join(args.out, "grid_compare.png"))
    plot_frontier_compare(gA, gB, args.label_a, args.label_b,
                          os.path.join(args.out, "frontier_compare.png"))
    print_table(gA, gB, args.label_a, args.label_b)
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
