"""Summarize logged learning and trajectory behavior without inferring convergence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = ("value_loss", "value_explained_variance_after", "post_approx_kl",
           "normalized_anchor_entropy", "conditional_normalized_entropy",
           "forced_action_fraction", "positive_reward_fraction", "return_std")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder, fraction):
    protocol = json.loads((folder/"protocol.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    updates = [json.loads(line) for line in (folder/"updates.jsonl").read_text().splitlines()]
    assert report["updates"] == len(updates)
    budget = protocol["config"]["seconds"]
    if budget <= 0:
        raise ValueError("These windows require a positive fixed wall budget")
    early = [r for r in updates if r["seconds"] <= fraction*budget]
    late = [r for r in updates if r["seconds"] >= (1-fraction)*budget]

    def metrics(rows):
        result = {}
        for key in METRICS:
            values = [r["metrics"].get(key) for r in rows]
            values = [v for v in values if v is not None]
            assert all(math.isfinite(v) for v in values)
            result[key] = dict(samples=len(values), median=statistics.median(values) if values else None)
        return result

    bins = {"early": Counter(), "late": Counter()}
    families = {name: Counter() for name in bins}
    history = {row["transition"]: row for row in
               (json.loads(line) for line in (folder/"best-history.jsonl").read_text().splitlines())
               if row["strict_gate_improvement"]}
    trajectory = None; graphs = set(); circuits = set(); transitions = 0
    neutral = peak_neutral = 0; neutral_thresholds = Counter(); best_prefixes = []
    with (folder/"events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line); transitions += 1
            if event["trajectory"] != trajectory:
                trajectory = event["trajectory"]
                graphs = {event["parent_graph_key"]}; circuits = set()
                neutral = peak_neutral = 0
            if event["transition"] in history:
                best_prefixes.append(dict(transition=event["transition"], gates=event["cost"]["gates"],
                                          trajectory_step=event["trajectory_step"],
                                          preceding_neutral_steps=neutral, peak_neutral_steps_in_prefix=peak_neutral))
            if event["status"] != "excluded_after_deadline":
                neutral = (neutral+1 if event["status"] in ("accepted", "anchor_noop")
                           and event["cost"] is not None and event["reward"] == 0 else 0)
                peak_neutral = max(peak_neutral, neutral)
                if neutral in (16, 32, 64):
                    neutral_thresholds[neutral] += 1
            group = ("early" if event["seconds"] <= fraction*budget else
                     "late" if event["seconds"] >= (1-fraction)*budget else None)
            counted = event["status"] != "excluded_after_deadline"
            if group is not None and counted:
                b = bins[group]; b["transitions"] += 1
                b[event["status"]] += 1
                b["positive_reward"] += int(event["reward"] > 0)
                b["negative_reward"] += int(event["reward"] < 0)
                b["zero_reward"] += int(event["reward"] == 0)
                families[group][event["action"]["family"]] += 1
                if event["status"] in ("accepted", "accepted_unscored"):
                    b["accepted_rewrite"] += 1
                    b["repeated_raw_graph_in_trajectory"] += int(event["graph_key"] in graphs)
                    if event.get("sha256"):
                        b["repeated_completion_in_trajectory"] += int(event["sha256"] in circuits)
            if event["status"] in ("accepted", "accepted_unscored"):
                graphs.add(event["graph_key"])
                if event.get("sha256"):
                    circuits.add(event["sha256"])
    assert transitions == report["transitions"]
    return dict(folder=str(folder), source_commit=protocol["git_commit"],
                policy=protocol["config"]["policy"], learning_rate=protocol["config"]["learning_rate"],
                best=report["best"]["cost"], updates=len(updates), transitions=transitions,
                early_update_count=len(early), late_update_count=len(late),
                early_metrics=metrics(early), late_metrics=metrics(late),
                neutral_run_threshold_hits=dict(neutral_thresholds),
                strict_gate_improvement_trajectory_prefixes=best_prefixes,
                trajectory_windows={k: dict(counts=dict(bins[k]), families=dict(families[k])) for k in bins},
                files={name: digest(folder/name) for name in
                       ("protocol.json", "report.json", "updates.jsonl", "events.jsonl", "best-history.jsonl")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--window-fraction", type=float, default=.2)
    p.add_argument("--allow-unaudited", action="store_true",
                   help="Explicit descriptive preview of closed training reports; no passed-study claim")
    a = p.parse_args()
    if not 0 < a.window_fraction <= .5:
        raise ValueError("Window fraction must be in (0,.5]")
    campaign = json.loads((a.campaign/"campaign.json").read_text())
    summary = json.loads((a.campaign/"summary.json").read_text())
    assert summary["completed"] == summary["expected"] == len(campaign["plan"])
    assert len(summary["rows"]) == summary["expected"]
    assert not any(row.get("status") == "failed" for row in summary["rows"])
    audited = False; audit_sha = None
    if (a.campaign/"finished.json").exists() and (a.campaign/"audit/verification.json").exists():
        finished = json.loads((a.campaign/"finished.json").read_text())
        audit = json.loads((a.campaign/"audit/verification.json").read_text())
        audited = (finished["status"] == audit["status"] == "passed" and
                   finished["trials"] == audit["trials"] == summary["expected"])
        if audited:
            audit_sha = digest(a.campaign/"audit/verification.json")
            # Bind descriptive data to the completed full-replay receipts.
            receipts = {(r["report_sha256"], r["events_sha256"]) for r in audit["records"]
                        if r["status"] == "passed"}
            assert len(receipts) == summary["expected"]
            assert all((digest(a.campaign/r["folder"]/"report.json"),
                        digest(a.campaign/r["folder"]/"events.jsonl")) in receipts for r in summary["rows"])
    if not audited and not a.allow_unaudited:
        raise ValueError("Wait for the full study audit or explicitly request an unaudited descriptive preview")
    records = [summarize(a.campaign/row["folder"], a.window_fraction) for row in summary["rows"]]
    result = dict(status="descriptive_summary" if audited else "unaudited_descriptive_preview",
                  study_audit_passed=audited, source_commit=campaign["source_commit"],
                  study_audit_sha256=audit_sha, window_fraction=a.window_fraction, records=records,
                  scope="Medians of logged PPO metrics in the first/last wall-time windows. Repetition counts use exact raw records or already verified completion hashes within each trajectory. Equal completions can enable different future rewrites; repetition alone is not a valid pruning rule. Lower loss or entropy is not evidence of useful learning or convergence. Frozen weight probes and gate-count controls remain necessary.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status=result["status"], trials=len(records), study_audit_passed=audited)))


if __name__ == "__main__":
    main()
