"""Export the rule benefit/cost evidence as a standalone research figure."""
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
    data = json.loads((a.evidence/"analysis.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    names = ["matching", "feasibility", "rewrite_verify", "score", "ppo", "inference"]
    labels = ["View + proposals", "Budget previews", "Rewrite + verify", "Extract + cleanup", "PPO", "Inference"]
    values = [data["history"]["timer_percent"][k] for k in names]
    ax.barh(labels[::-1], values[::-1], color="#427aa1")
    for i, v in enumerate(values[::-1]):
        ax.text(v+.4, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 47); ax.set_xlabel("Share of summed search wall time (%)")
    ax.set_title("A. Historical 26-circuit workload")
    rows = data["components"]; circuits = ["barenco_tof_3", "mod5_4", "adder_8", "gf2^8_mult"]
    profiles = ["native", "macros", "no-a", "full"]
    index = {(r["circuit"], r["profile"]): r for r in rows}
    ax = axes[0, 1]; width = .18; colors = ["#335c67", "#718f59", "#e09f3e", "#9e2a2b"]
    for i, profile in enumerate(profiles):
        ratios = [index[c, profile]["total_ms"]/index[c, "native"]["total_ms"] for c in circuits]
        ax.bar(np.arange(4)+(i-1.5)*width, ratios, width, label=profile, color=colors[i])
    ax.axhline(1., color="gray", lw=.8); ax.set_xticks(range(4), ["barenco3", "mod5", "adder8", "gf2^8"])
    ax.set_ylabel("Proposal + cold preview time / native")
    ax.set_ylim(0, 1.25*max(index[c, p]["total_ms"]/index[c, "native"]["total_ms"]
                          for c in circuits for p in profiles))
    ax.set_title("B. Identical states; warmed proposals")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax = axes[1, 0]
    values = [index[c, "full"]["a_total_percent"] for c in circuits]
    ax.bar(range(4), values, color="#9e2a2b")
    for i, v in enumerate(values):
        ax.text(i, v+.06, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(range(4), ["barenco3", "mod5", "adder8", "gf2^8"])
    ax.set_ylim(0, max(values)*1.25)
    ax.set_ylabel("A matching / (proposal + preview) (%)")
    ax.set_title("C. Direct nonlinear A matching cost")
    ax = axes[1, 1]
    if "four_layer_campaign" in data and data["four_layer_campaign"]["trials"] == 24:
        rows = data["four_layer_campaign"]["rows"]
        for j, policy in enumerate(("frozen", "online")):
            for i, profile in enumerate(profiles):
                subset = sorted((r for r in rows if r["mode"] == "rules_"+profile.replace("-", "_")
                                 and r["policy"] == policy), key=lambda r: r["seed"])
                x = i+(j-.5)*.24
                ys = [r["gates"] for r in subset]
                ax.scatter([x+v for v in np.linspace(-.035, .035, len(ys))], ys, marker="o" if j == 0 else "x",
                           color="#335c67" if j == 0 else "#9e2a2b", s=35,
                           label=policy if i == 0 else None)
                ax.plot([x-.07, x+.07], [statistics.median(ys)]*2, color="black", lw=1)
        ax.set_xticks(range(4), profiles); ax.set_ylabel("Best total NAM gates (lower is better)")
        ax.set_title("D. barenco_tof_3; 600 s, 3 seeds per policy")
        ax.legend(fontsize=9)
    else:
        ax.axis("off"); ax.text(.1, .5, "Training results await complete replay audit")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(a.evidence/"rule-analysis.pdf")
    fig.savefig(a.evidence/"rule-analysis.png", dpi=180)


if __name__ == "__main__":
    main()
