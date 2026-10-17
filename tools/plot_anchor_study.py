"""Plots from completed, audited on-target anchor-search trials."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = dict(global32="Global32", anchor8="Anchor8", anchor32="Anchor32",
              anchor8_nodes="Node prior", anchor8_lr1e3="LR 0.001",
              anchor8_nodes_lr1e3="Node prior + LR 0.001")
COLORS = dict(zip(LABELS, plt.get_cmap("tab10").colors))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def band(ax, xs, series, label, color, step=False):
    ys = np.asarray(series)
    useful = np.any(np.isfinite(ys), axis=0)
    if not useful.any():
        return
    values = ys[:, useful]
    median = np.nanmedian(values, axis=0)
    lo, hi = np.nanquantile(values, [.25, .75], axis=0)
    ax.plot(xs[useful], median, color=color, label=label, linewidth=1.5,
            drawstyle="steps-post" if step else "default")
    ax.fill_between(xs[useful], lo, hi, color=color, alpha=.12, linewidth=0,
                    step="post" if step else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    root, out = a.campaign.resolve(), a.output.resolve()
    assert json.loads((root/"finished.json").read_text())["status"] == "passed"
    assert json.loads((root/"audit/verification.json").read_text())["status"] == "passed"
    campaign = json.loads((root/"campaign.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    sources = {}

    def record(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    for path in (root/"campaign.json", root/"finished.json", root/"audit/verification.json"):
        record(path)
    artifacts = []
    for circuit in sorted({r["circuit"] for r in campaign["plan"]}):
        trials = [r for r in campaign["plan"] if r["circuit"] == circuit]
        modes = sorted({r["mode"] for r in trials}, key=lambda mode: list(LABELS).index(mode))
        seeds = sorted({r["seed"] for r in trials})
        grid = np.linspace(0, campaign["seconds"], 201)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
        for ax, policy in zip(axs, ("frozen", "online")):
            for mode in modes:
                series = []
                for spec in trials:
                    if spec["mode"] != mode or spec["policy"] != policy:
                        continue
                    folder = root/spec["folder"]
                    protocol = json.loads((folder/"protocol.json").read_text())
                    history = read_rows(folder/"best-history.jsonl")
                    times = np.asarray([0.]+[r["seconds"] for r in history])
                    gates = np.asarray([protocol["root_cost"]["gates"]]+[r["cost"]["gates"] for r in history])
                    series.append(gates[np.searchsorted(times, grid, side="right")-1])
                    record(folder/"protocol.json"); record(folder/"best-history.jsonl")
                if series:
                    band(ax, grid, series, LABELS[mode], COLORS[mode], step=True)
            ax.set_title(policy.capitalize())
            ax.set_xlabel("Search wall time (s)"); ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axs[0].set_ylabel("Best total NAM gates")
        scope = "1 seed; single runs" if len(seeds) == 1 else str(len(seeds))+" independent seeds; median and interquartile range"
        fig.suptitle(circuit+" | "+scope, fontsize=11)
        path = out/(circuit+"-gates.png"); fig.savefig(path, dpi=180); plt.close(fig); artifacts.append(path.name)

        metrics = [("value_loss", "Value loss"), ("value_explained_variance_after", "Value explained variance"),
                   ("approx_kl", "PPO approximate KL"), ("positive_reward_fraction", "Positive reward fraction"),
                   ("anchor_entropy", "Anchor entropy (includes fixed prior)"),
                   ("conditional_action_entropy", "Local action entropy")]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
        for ax, (key, label) in zip(axs.flat, metrics):
            for mode in modes:
                series = []
                for spec in trials:
                    if spec["mode"] != mode or spec["policy"] != "online":
                        continue
                    path = root/spec["folder"]/"updates.jsonl"; updates = read_rows(path); record(path)
                    usable = [(r["seconds"], r["metrics"].get(key)) for r in updates
                              if r["metrics"].get(key) is not None]
                    if not usable:
                        continue
                    times, values = map(np.asarray, zip(*usable))
                    interpolated = np.interp(grid, times, values)
                    interpolated[(grid < times[0]) | (grid > times[-1])] = np.nan
                    series.append(interpolated)
                if series:
                    band(ax, grid, series, LABELS[mode], COLORS[mode])
            ax.set_title(label, fontsize=10); ax.set_xlabel("Search wall time (s)")
            ax.grid(alpha=.2)
            if key == "approx_kl":
                ax.set_yscale("symlog", linthresh=1e-6)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.suptitle(circuit+" | PPO diagnostics; median and interquartile range", fontsize=11)
        path = out/(circuit+"-ppo.png"); fig.savefig(path, dpi=180); plt.close(fig); artifacts.append(path.name)
    manifest = dict(source_commit=campaign["source_commit"], figures=artifacts, input_files=sources,
                    matplotlib=matplotlib.__version__, numpy=np.__version__,
                    caveats=["Each circuit is trained independently; no held-out evaluation",
                             "Faster or lower-loss training does not establish improvement over frozen search",
                             "Prior-weighted anchor entropy is not directly comparable to uniform-prior entropy",
                             "No extrapolation of PPO curves before the first or after the final update",
                             "Only final audited trials are plotted; the campaign archive checksum is verified separately"])
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(status="passed", figures=artifacts)))


if __name__ == "__main__":
    main()
