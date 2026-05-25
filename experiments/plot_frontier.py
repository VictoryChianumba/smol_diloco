#!/usr/bin/env python3
"""Plot the extended frontier sweep (run_frontier.sh) and locate the break.

Reads experiments/frontier/k*/{config.json, metrics.jsonl} and produces:
  - frontier_extended.png   communication volume vs final loss across K, with the
                            "break" region (loss jumps relative to the best K) shaded
  - convergence_extended.png  val loss vs inner steps, one curve per K

Also prints a markdown table and the detected break point.

Usage: python experiments/plot_frontier.py [--results experiments/frontier]
                                            [--out experiments/figures]
                                            [--break-frac 0.10]
"""
import argparse, json, glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    return cfg, rows


def load_all(results_dir):
    runs = []
    for d in glob.glob(os.path.join(results_dir, "k*")):
        if os.path.isfile(os.path.join(d, "metrics.jsonl")):
            runs.append(load_run(d))
    runs.sort(key=lambda cr: cr[0]["local_steps"])
    return runs


def _evals(rows):
    return [r for r in rows if r.get("val_loss") is not None]


MIN_ROUNDS = 4  # below this, the final loss is too under-resolved to trust


def summarize(runs):
    """list of dicts: K, final, best_in_run, comm_gb, rounds, rise, unstable, resolved."""
    out = []
    for cfg, rows in runs:
        ev = _evals(rows)
        finals = [r["val_loss"] for r in ev]
        final = finals[-1] if finals else float("nan")
        best_in_run = min(finals) if finals else float("nan")
        rounds = len(rows)
        out.append({
            "K": cfg["local_steps"],
            "final": final,
            "best_in_run": best_in_run,
            "comm_gb": rows[-1]["comm_bytes"] / 1e9,
            "rounds": rounds,
            "rise": final - best_in_run,          # within-run divergence
            "resolved": rounds >= MIN_ROUNDS,
        })
    return out


def find_break(summary, frac):
    """A run 'breaks' when its loss diverges within the run: the final loss rises
    off the run's own minimum by more than `frac`. This catches instability that a
    final-loss-vs-K curve hides, and ignores under-resolved (few-round) runs whose
    endpoint is just early-descent noise. Returns the smallest such K."""
    broken = [s for s in summary
              if s["resolved"] and s["best_in_run"] > 0
              and s["rise"] / s["best_in_run"] > frac]
    for s in summary:
        s["unstable"] = s in broken
    return (min((s["K"] for s in broken), default=None),
            [s for s in summary if not s["resolved"]])


def plot_frontier(runs, summary, break_K, out):
    Ks = [s["K"] for s in summary]
    # best-in-run loss is the fair convergence measure (final is noisy at high K)
    loss = [s["best_in_run"] for s in summary]
    gb = [s["comm_gb"] for s in summary]

    fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("sync interval K (local steps between syncs)")
    ax1.set_ylabel("best validation loss in run", color="tab:blue")
    ax1.plot(Ks, loss, "o-", color="tab:blue", zorder=3)
    # ring unstable runs (loss diverged within the run) and mark under-resolved ones
    for s in summary:
        if s.get("unstable"):
            ax1.scatter([s["K"]], [s["best_in_run"]], s=180, facecolors="none",
                        edgecolors="tab:red", linewidths=2, zorder=4)
        if not s["resolved"]:
            ax1.annotate(f"{s['rounds']} rounds\n(under-resolved)", (s["K"], s["best_in_run"]),
                         textcoords="offset points", xytext=(-2, 10), fontsize=7,
                         color="gray", ha="center")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(Ks); ax1.set_xticklabels([str(k) for k in Ks])
    if break_K is not None:
        ax1.axvspan(break_K / 1.4, Ks[-1] * 1.4, color="tab:red", alpha=0.08, zorder=0)
        ax1.annotate(f"frontier breaks\n(loss diverges within run @ K={break_K})",
                     (break_K, max(loss)), textcoords="offset points", xytext=(4, -28),
                     color="tab:red", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("communication volume (GB)", color="tab:red")
    ax2.plot(Ks, gb, "s--", color="tab:red", zorder=2)
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Extended frontier (outer_lr={runs[0][0]['outer_lr']}, fixed compute budget)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close()


def plot_convergence(runs, out):
    plt.figure(figsize=(7.2, 4.6))
    for cfg, rows in runs:
        ev = _evals(rows)
        if ev:
            plt.plot([r["inner_steps"] for r in ev], [r["val_loss"] for r in ev],
                     marker="o", ms=3, label=f"K={cfg['local_steps']}")
    plt.xlabel("inner steps per worker")
    plt.ylabel("validation loss")
    plt.title("Convergence across sync intervals (extended)")
    plt.legend(title="sync interval", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/frontier")
    ap.add_argument("--out", default="experiments/figures")
    ap.add_argument("--break-frac", type=float, default=0.10,
                    help="loss increase over best-K that counts as 'broken'")
    args = ap.parse_args()

    runs = load_all(args.results)
    if not runs:
        raise SystemExit(f"no runs found under {args.results}")
    os.makedirs(args.out, exist_ok=True)
    summary = summarize(runs)
    break_K, under_resolved = find_break(summary, args.break_frac)

    plot_frontier(runs, summary, break_K, os.path.join(args.out, "frontier_extended.png"))
    plot_convergence(runs, os.path.join(args.out, "convergence_extended.png"))

    print("\n| Sync interval K | Rounds | Best val loss | Final val loss | Within-run rise | Comm (GB) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for s in summary:
        flag = " ⚠" if s.get("unstable") else (" *" if not s["resolved"] else "")
        print(f"| {s['K']} | {s['rounds']} | {s['best_in_run']:.3f} | {s['final']:.3f} "
              f"| +{s['rise']:.3f}{flag} | {s['comm_gb']:.4f} |")
    print()
    if break_K is not None:
        s = next(x for x in summary if x["K"] == break_K)
        print(f"Break: K={break_K} -- loss diverges within the run "
              f"(rises from {s['best_in_run']:.3f} to {s['final']:.3f}, "
              f"+{int(s['rise']/s['best_in_run']*100)}%) as averaging fails to reconcile worker drift.")
    else:
        print(f"No within-run divergence detected up to K={summary[-1]['K']}.")
    if under_resolved:
        ks = ", ".join(f"K={s['K']} ({s['rounds']} rounds)" for s in under_resolved)
        print(f"Under-resolved at this budget (endpoint unreliable): {ks}.")
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
