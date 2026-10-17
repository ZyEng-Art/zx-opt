"""All-seed search curves and PPO diagnostics for the general ZX runner."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=a.refresh)
    campaign = json.loads((a.campaign/"campaign.json").read_text())
    summary = json.loads((a.campaign/"summary.json").read_text())
    assert summary["completed"] == summary["expected"] == len(summary["rows"])
    assert not any(r["status"] == "failed" for r in summary["rows"])
    budget = campaign["seconds"]; grid = np.linspace(0, budget, 401)
    curves = {}; trials = []; learning = []
    def mean(updates, key):
        values = [u["metrics"].get(key) for u in updates]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    for row in summary["rows"]:
        folder = a.campaign/row["folder"]
        protocol = json.loads((folder/"protocol.json").read_text())
        assert not protocol["tracked_dirty"]
        events = [json.loads(l) for l in (folder/"events.jsonl").read_text().splitlines()]
        curve = np.full(len(grid), protocol["root_cost"]["gates"], dtype=float)
        incumbent = protocol["root_cost"]["gates"]; previous = 0.; area = 0.
        statuses = Counter(e["status"] for e in events)
        for e in events:
            if e["status"] == "accepted":
                area += (e["seconds"]-previous)*incumbent
                previous = e["seconds"]; incumbent = min(incumbent, e["cost"]["gates"])
                curve[grid >= e["seconds"]] = np.minimum(curve[grid >= e["seconds"]], incumbent)
        area += (budget-previous)*incumbent
        curves.setdefault(row["arm"], []).append(curve)
        updates = [json.loads(l) for l in (folder/"updates.jsonl").read_text().splitlines()]
        trial = dict(row, mean_anytime_gates=area/budget, statuses=dict(statuses),
                     git_commit=protocol["git_commit"], input_sha256=protocol["input_sha256"],
                     events_sha256=hashlib.sha256((folder/"events.jsonl").read_bytes()).hexdigest(),
                     invalid_extractions=row.get("completion", {}).get("unique_invalid_qasm", 0),
                     unscored_fraction=row["unscored"]/max(1, row["accepted"]))
        if updates:
            count = max(1, len(updates)//4)
            keys = ["value_loss", "policy_loss", "conditional_normalized_entropy", "value_explained_variance_after",
                    "approx_kl", "clip_fraction", "nonzero_reward_fraction", "positive_reward_fraction"]
            trial["learning"] = {k: dict(first=mean(updates[:count], k), last=mean(updates[-count:], k)) for k in keys}
            trial["learning"]["optimizer_steps"] = sum(u["optimizer_steps"] for u in updates)
            learning.append((row["arm"], row["seed"], updates))
        trials.append(trial)
    fig, ax = plt.subplots(figsize=(9, 5))
    aggregate = []
    for arm, values in sorted(curves.items()):
        group = [r for r in trials if r["arm"] == arm]
        arrays = np.array(values)
        line = ax.plot(grid, arrays.mean(0), label=arm)[0]
        for c in arrays:
            ax.plot(grid, c, color=line.get_color(), alpha=.18, lw=.65)
        aggregate.append(dict(arm=arm, seeds=[r["seed"] for r in group],
                              final_gates=[r["best"]["cost"]["gates"] for r in group],
                              final_cx=[r["best"]["cost"]["cx"] for r in group],
                              mean_anytime_gates=float(np.mean([r["mean_anytime_gates"] for r in group])),
                              transitions=[r["transitions"] for r in group], updates=[r["updates"] for r in group],
                              full_budget_runs=sum(r["total_loop_seconds"] >= .99*budget for r in group),
                              unscored=sum(r["unscored"] for r in group), recovered=sum(r["recovered"] for r in group),
                              invalid_extractions=sum(r["invalid_extractions"] for r in group),
                              rejections=sum(r["statuses"].get("rejected", 0) for r in group)))
    ax.set(xlabel="Search wall time (seconds)", ylabel="Best total NAM gates", title="barenco_tof_3: all seeds and arms")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(a.output/("anytime."+suffix), dpi=170)
    plt.close(fig)
    if learning:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        metrics = ["value_loss", "conditional_normalized_entropy", "value_explained_variance_after",
                   "policy_loss", "approx_kl", "clip_fraction"]
        labels = ["Value MSE", "Conditional normalized entropy", "Value explained variance", "Policy loss", "Approximate KL", "Clip fraction"]
        for arm, seed, updates in learning:
            for ax, key in zip(axes.flatten(), metrics):
                ax.plot([u["seconds"] for u in updates], [u["metrics"].get(key) for u in updates], label=f"{arm} s{seed}")
        for ax, label in zip(axes.flatten(), labels):
            ax.set(xlabel="Search wall time (seconds)", ylabel=label); ax.grid(alpha=.2)
        axes[0, 0].set_yscale("log"); axes[0, 1].set_ylim(0, 1.05)
        axes[0, 0].legend(fontsize=7); fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(a.output/("learning."+suffix), dpi=170)
        plt.close(fig)
    paired = []
    for base in ("macros", "jpv"):
        for seed in campaign["seeds"]:
            f = next((r for r in trials if r["arm"] == base+"_frozen" and r["seed"] == seed), None)
            o = next((r for r in trials if r["arm"] == base+"_online" and r["seed"] == seed), None)
            if f and o:
                paired.append(dict(base=base, seed=seed, online_minus_frozen_anytime=o["mean_anytime_gates"]-f["mean_anytime_gates"],
                                   online_final=o["best"]["cost"], frozen_final=f["best"]["cost"]))
    best = min(trials, key=lambda r: (r["best"]["cost"]["gates"], r["best"]["cost"]["cx"]))
    shutil.copyfile(a.campaign/best["folder"]/"best.qasm", a.output/"best.qasm")
    result = dict(status="passed", budget_seconds=budget, aggregate=aggregate, trials=trials, paired=paired,
                  scope="single target circuit, paired random initialization, no generalization/global optimality claim",
                  best_source=dict(arm=best["arm"], seed=best["seed"]))
    (a.output/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["All arms and seeds retained; lower mean anytime gates is better."]
    for r in aggregate:
        lines.append(f"{r['arm']}: gates={r['final_gates']}, CX={r['final_cx']}, anytime={r['mean_anytime_gates']:.6f}, updates={r['updates']}, unscored={r['unscored']}, recovered={r['recovered']}")
    lines.append("Loss/EV/entropy and fixed-state policy probes diagnose learning; only paired search outcomes assess search benefit.")
    (a.output/"findings.txt").write_text("\n".join(lines)+"\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
