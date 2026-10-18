"""Plot closed audited training diagnostics alongside frozen-weight utility."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_anchor import require_audit_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    campaign = json.loads((args.campaign/"campaign.json").read_text())
    audit = json.loads((args.campaign/"audit/verification.json").read_text())
    finished = json.loads((args.campaign/"finished.json").read_text())
    assert audit["status"] == finished["status"] == "passed"
    assert audit["trials"] == finished["trials"] == len(campaign["plan"])
    weights = json.loads(args.weights.read_text()); assert weights["status"] == "passed"
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"global32": "#666666", "anchor8_resource": "#b45309", "anchor8_resource_lr1e4": "#0369a1"}
    styles = {137: "-", 163: "--", 197: ":"}
    diagnostics = []; shown = set()

    def smooth(values, window=50):
        return [statistics.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]

    for spec in campaign["plan"]:
        if spec["policy"] != "online":
            continue
        folder = args.campaign/spec["folder"]
        require_audit_receipt(folder, spec, audit)
        path = folder/"updates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        times = [r["seconds"]/60 for r in rows]
        mode = spec["mode"]; label = mode if mode not in shown else None; shown.add(mode)
        axes[0].plot(times, smooth([r["metrics"]["value_loss"] for r in rows]),
                     color=colors[mode], linestyle=styles[spec["seed"]], label=label, alpha=.8)
        entropies = [r["metrics"].get("normalized_anchor_entropy") for r in rows]
        if all(x is not None for x in entropies):
            axes[1].plot(times, smooth(entropies), color=colors[mode], linestyle=styles[spec["seed"]], alpha=.8)
        metrics = {}
        for key in ("value_loss", "normalized_anchor_entropy", "value_explained_variance_after"):
            values = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
            if values:
                metrics[key] = dict(first100_median=statistics.median(values[:100]),
                                    last100_median=statistics.median(values[-100:]), minimum=min(values), maximum=max(values))
        diagnostics.append(dict(mode=mode, seed=spec["seed"], updates=len(rows), metrics=metrics,
                                updates_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pairs = Counter((p["initial"]["gates"], p["trained"]["gates"]) for p in weights["pairs"])
    for (before, after), count in sorted(pairs.items()):
        color = "#15803d" if after < before else "#be123c" if after > before else "#777777"
        axes[2].scatter(before, after, s=120*count, color=color, alpha=.65)
        axes[2].text(before, after, str(count), ha="center", va="center", color="white")
    axes[2].plot([36.5, 40.5], [36.5, 40.5], color="black", linestyle="--", linewidth=.8)
    axes[0].set(title="Critic MSE (50-update mean)", xlabel="Training minutes", ylabel="Value loss")
    axes[1].set(title="Anchor entropy (50-update mean)", xlabel="Training minutes", ylabel="Normalized entropy")
    axes[2].set(title="Frozen-weight searches: 2 wins / 8 ties / 17 losses", xlabel="Initial weights: best gates", ylabel="Trained weights: best gates")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("barenco_tof_3: loss/entropy changes do not establish useful learning", fontsize=12)
    fig.tight_layout(); fig.savefig(args.output/"training-and-utility.png", dpi=160); plt.close(fig)
    result = dict(status="completed_audited_training_descriptive_diagnostics", source_commit=campaign["source_commit"],
                  training=diagnostics, weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Training-set diagnostics and separate frozen model-only utility searches on the same circuit. Lines differ by training seed, controls are related; no convergence/generalization claim. Single-action conditional menus have zero entropy without implying policy collapse.")
    (args.output/"diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(training_runs=len(diagnostics), weight_pairs=len(weights["pairs"]))))


if __name__ == "__main__":
    main()
