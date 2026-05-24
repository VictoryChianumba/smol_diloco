#!/usr/bin/env python3
"""Turn sweep logs into figures.

Reads experiments/results/k*/{config.json, metrics.jsonl} and produces:
  - convergence.png      val loss vs inner steps, one curve per sync interval
  - delta_norm.png       avg worker-delta norm vs round (drift between syncs)
  - frontier.png         the headline: communication volume vs final val loss

Usage: python experiments/plot_results.py [--results experiments/results] [--out experiments/figures]
"""
import argparse, json, glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    rows = []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return cfg, rows


def load_all(results_dir):
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, "k*"))):
        if os.path.isfile(os.path.join(run_dir, "metrics.jsonl")):
            cfg, rows = load_run(run_dir)
            runs.append((cfg, rows))
    # order by sync interval K
    runs.sort(key=lambda cr: cr[0]["local_steps"])
    return runs


def _evals(rows):
    """rows with a recorded val_loss."""
    return [r for r in rows if r.get("val_loss") is not None]


def plot_convergence(runs, out):
    plt.figure(figsize=(7, 4.5))
    for cfg, rows in runs:
        ev = _evals(rows)
        if not ev:
            continue
        xs = [r["inner_steps"] for r in ev]
        ys = [r["val_loss"] for r in ev]
        plt.plot(xs, ys, marker="o", ms=3, label=f"K={cfg['local_steps']}")
    plt.xlabel("inner steps per worker")
    plt.ylabel("validation loss")
    plt.title("Convergence vs sync interval (fixed compute budget)")
    plt.legend(title="sync interval")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_delta_norm(runs, out):
    plt.figure(figsize=(7, 4.5))
    for cfg, rows in runs:
        xs = [r["round"] for r in rows if r.get("avg_delta_norm") is not None]
        ys = [r["avg_delta_norm"] for r in rows if r.get("avg_delta_norm") is not None]
        if xs:
            plt.plot(xs, ys, label=f"K={cfg['local_steps']}")
    plt.xlabel("outer round")
    plt.ylabel("avg worker-delta L2 norm")
    plt.title("Local drift between syncs")
    plt.legend(title="sync interval")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_frontier(runs, out):
    """Communication volume vs final val loss -- the DiLoCo efficiency frontier."""
    Ks, final_loss, gb = [], [], []
    for cfg, rows in runs:
        ev = _evals(rows)
        if not ev:
            continue
        Ks.append(cfg["local_steps"])
        final_loss.append(ev[-1]["val_loss"])
        gb.append(rows[-1]["comm_bytes"] / 1e9)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("sync interval K (local steps between syncs)")
    ax1.set_ylabel("final validation loss", color="tab:blue")
    ax1.plot(Ks, final_loss, "o-", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(Ks)
    ax1.set_xticklabels([str(k) for k in Ks])
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("communication volume (GB)", color="tab:red")
    ax2.plot(Ks, gb, "s--", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Communication vs convergence frontier")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close()


def print_table(runs):
    """Markdown benchmark table, copy-pasteable into the README."""
    print("\n| Sync interval K | Rounds | Final val loss | Comm (GB) | Wall-clock (s) |")
    print("|---:|---:|---:|---:|---:|")
    for cfg, rows in runs:
        ev = _evals(rows)
        final = ev[-1]["val_loss"] if ev else float("nan")
        gb = rows[-1]["comm_bytes"] / 1e9
        wall = sum(r["wall_time_s"] for r in rows)
        print(f"| {cfg['local_steps']} | {len(rows)} | {final:.3f} | {gb:.3f} | {wall:.1f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="experiments/figures")
    args = ap.parse_args()

    runs = load_all(args.results)
    if not runs:
        raise SystemExit(f"no runs found under {args.results}")
    os.makedirs(args.out, exist_ok=True)

    plot_convergence(runs, os.path.join(args.out, "convergence.png"))
    plot_delta_norm(runs, os.path.join(args.out, "delta_norm.png"))
    plot_frontier(runs, os.path.join(args.out, "frontier.png"))
    print_table(runs)
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
