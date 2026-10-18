"""Standalone figures for verified resource bounds and staged rule allocation."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path); a = p.parse_args()
    d = json.loads((a.evidence/"analysis.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    circuits = ["barenco_tof_3", "mod5_4", "adder_8", "gf2^8_mult"]
    ax = axes[0, 0]
    ratios = [d["components"]["totals"][c]["bounds"]["seconds"]/d["components"]["totals"][c]["reference"]["seconds"] for c in circuits]
    ax.bar(range(4), ratios, color="#427aa1")
    for i, r in enumerate(ratios):
        ax.text(i, r+.02, f"{r:.2f}", ha="center")
    ax.axhline(1., color="gray", lw=.8); ax.set_ylim(0, 1.12)
    ax.set_xticks(range(4), ["barenco3", "mod5", "adder8", "gf2^8"])
    ax.set_ylabel("Bounds / reference feasibility time")
    ax.set_title("A. Identical menus; cold previews, 3 repeats")
    ax = axes[0, 1]
    pairs = [r for r in d["common_prefixes"] if r["node"] == 22 and r["mode"] == "cost_bounds"]
    for j, policy in enumerate(("frozen", "online")):
        values = [r["same_steps_speedup"] for r in pairs if r["policy"] == policy]
        ax.scatter(np.arange(3)+(j-.5)*.15, values, marker="o" if j == 0 else "x",
                   color="#335c67" if j == 0 else "#9e2a2b", label=policy, s=45)
    ax.axhline(1., color="gray", lw=.8); ax.set_xticks(range(3), ["1409", "1423", "1451"])
    ax.set_xlabel("Seed"); ax.set_ylabel("Reference / bounds time for identical steps")
    ax.set_title("B. End-to-end; identical semantic prefixes"); ax.legend()
    modes = ["cost_bounds", "cost_staged", "cost_quota", "cost_staged_quota"]
    labels = ["bounds", "+ staged", "+ quota", "+ both"]
    ax = axes[1, 0]
    values = [next(r["median_accepted_per_second"] for r in d["summaries"] if r["node"] == 23 and r["mode"] == m) for m in modes]
    ax.bar(range(4), values, color=["#427aa1", "#718f59", "#e09f3e", "#9e2a2b"])
    for i, v in enumerate(values):
        ax.text(i, v+.15, f"{v:.2f}", ha="center")
    ax.set_ylim(0, max(values)*1.2); ax.set_xticks(range(4), labels)
    ax.set_ylabel("Median accepted rewrites / second")
    ax.set_title("C. Node23; 600 s, 6 trials per configuration")
    ax = axes[1, 1]
    for j, policy in enumerate(("frozen", "online")):
        for i, mode in enumerate(modes):
            rows = sorted((r for r in d["records"] if r["node"] == 23 and r["mode"] == mode and r["policy"] == policy), key=lambda r:r["seed"])
            ys = [r["best_gates"] for r in rows]; x = i+(j-.5)*.24
            ax.scatter(x+np.linspace(-.04, .04, len(ys)), ys,
                       marker="o" if j == 0 else "x", color="#335c67" if j == 0 else "#9e2a2b",
                       label=policy if i == 0 else None, s=40)
            ax.plot([x-.08, x+.08], [statistics.median(ys)]*2, color="black", lw=1.)
    ax.set_xticks(range(4), labels); ax.set_ylabel("Best total NAM gates (lower is better)")
    ax.set_title("D. barenco_tof_3; 3 seeds per policy"); ax.legend()
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(a.evidence/"resource-cost.pdf")
    fig.savefig(a.evidence/"resource-cost.png", dpi=180)


if __name__ == "__main__":
    main()
