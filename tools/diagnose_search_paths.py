"""Measure gate excursions and restart-pool representation diversity from saved runs.

This is observational: a hypothetical early restart changes subsequent random
choices, so these counts do not predict the outcome of a modified search.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def diagnose(folder, extra, checkpoint_pool=False):
    protocol = json.loads((folder/"protocol.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    best = protocol["root_cost"]["gates"]
    counts = Counter(); statuses = Counter(); excess = Counter(); maxima = {}; improvements = []
    representations = {}; files = [folder/"protocol.json", folder/"report.json", folder/"events.jsonl"]
    with (folder/"events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line); statuses[row["status"]] += 1
            if row.get("status") not in ("accepted", "anchor_noop", "accepted_unscored", "rejected"):
                continue
            counts["transitions"] += 1
            trajectory = row["trajectory"]
            cost = row.get("cost")
            if cost is not None:
                delta = cost["gates"]-best
                excess[delta] += 1
                maxima[trajectory] = max(maxima.get(trajectory, 0), delta)
                if delta > extra:
                    counts["above_incumbent_plus_extra"] += 1
                    counts["above_bound_"+row["status"]] += 1
                if row.get("graph_key"):
                    key = row["graph_key"]
                    previous = representations.get(key, {})
                    representations[key] = dict(view=row.get("view_key") or previous.get("view"),
                                                qasm=row.get("sha256") or previous.get("qasm"))
            new_best = row.get("best", {}).get("gates", best)
            if new_best < best:
                improvements.append(dict(transition=row["transition"], trajectory=trajectory,
                                         before=best, after=new_best,
                                         trajectory_prefix_max_excess=maxima.get(trajectory, 0)))
            best = new_best
    result = dict(folder=str(folder), policy=protocol["config"]["policy"], seed=protocol["config"]["seed"],
                  selection=protocol.get("selection"), final_best_gates=report["best"]["cost"]["gates"],
                  hypothetical_extra_gates=extra, counts=dict(counts), statuses=dict(statuses),
                  gate_excess_histogram=dict(sorted(excess.items())), strict_improvements=improvements,
                  improving_trajectory_prefixes_exceeding_bound=sum(
                      row["trajectory_prefix_max_excess"] > extra for row in improvements))
    if checkpoint_pool:
        # Only use this option for our own locally produced trusted checkpoints.
        import torch
        path = folder/"search-state.pt"; files.append(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        states = list(saved["pool"].scored.values())
        mapped = [representations[state.key] for state in states if state.key in representations]
        result["checkpoint_pool"] = dict(
            checkpoint_transitions=saved["counts"][0], size=len(states),
            raw_keys=len({state.key for state in states}), mapped=len(mapped),
            unique_known_views=len({row["view"] for row in mapped if row["view"]}),
            unique_known_qasm=len({row["qasm"] for row in mapped if row["qasm"]}),
            known_qasm=sum(row["qasm"] is not None for row in mapped),
            gate_histogram=dict(Counter(state.cost["gates"] for state in states)),
            node_histogram=dict(Counter(len(state.diagram.nodes) for state in states)),
            summary=saved["pool"].summary())
    result["files_sha256"] = {str(path): digest(path) for path in files}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--extra-gates", type=int, default=6)
    p.add_argument("--checkpoint-pool", action="store_true")
    a = p.parse_args()
    rows = [diagnose(path.parent, a.extra_gates, a.checkpoint_pool)
            for path in sorted(a.campaign.glob("*/*/report.json"))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        scope="Closed reports only; observational diagnostics before full study audits. Prefixes start at a buffer restart, not the root. No counterfactual outcome claim. Checkpoint pools may predate final reports.",
        trials=len(rows), rows=rows), indent=2)+"\n")
    print(json.dumps(dict(trials=len(rows), output=str(a.output),
                         transitions=sum(row["counts"].get("transitions", 0) for row in rows),
                         above_bound=sum(row["counts"].get("above_incumbent_plus_extra", 0) for row in rows),
                         strict_improvements=sum(len(row["strict_improvements"]) for row in rows),
                         improving_prefixes_above_bound=sum(row["improving_trajectory_prefixes_exceeding_bound"] for row in rows))))


if __name__ == "__main__":
    main()
