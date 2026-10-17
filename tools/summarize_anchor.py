"""Gate-count comparisons, useful throughput, paired prefixes and PPO traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def require_audit_receipt(folder, spec, verification):
    report_sha = hashlib.sha256((folder/"report.json").read_bytes()).hexdigest()
    events_sha = hashlib.sha256((folder/"events.jsonl").read_bytes()).hexdigest()
    matches = [r for r in verification["records"]
               if r["report_sha256"] == report_sha and r["events_sha256"] == events_sha
               and r["arm"] == spec["arm"] and r["seed"] == spec["seed"] and r["status"] == "passed"]
    if len(matches) != 1:
        raise ValueError("Report/events do not have one matching completed audit receipt: "+str(folder))
    receipt = matches[0]
    for field, name in (("protocol_sha256", "protocol.json"), ("best_history_sha256", "best-history.jsonl"),
                        ("initial_model_sha256", "initial.pt"), ("final_model_sha256", "final.pt")):
        if receipt.get(field) and hashlib.sha256((folder/name).read_bytes()).hexdigest() != receipt[field]:
            raise ValueError("Changed audited artifact: "+str(folder/name))
    for name, expected in receipt.get("improvement_model_receipts", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Improvement snapshot must be local to its trial")
        if hashlib.sha256((folder/relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed audited policy snapshot: "+str(folder/relative))
    return receipt


def summarize(root, output, baseline_mode=None):
    campaign = json.loads((root/"campaign.json").read_text())
    modes = {r["mode"] for r in campaign["plan"]}
    baseline_mode = baseline_mode or ("global32" if "global32" in modes else "anchor8")
    if baseline_mode not in modes:
        raise ValueError("Comparison baseline missing: "+baseline_mode)
    if not (root/"finished.json").is_file():
        raise RuntimeError("Only summarize a completed audited campaign")
    verification = json.loads((root/"audit/verification.json").read_text())
    assert verification["status"] == "passed"
    finished = json.loads((root/"finished.json").read_text())
    assert finished["status"] == "passed"
    assert finished["trials"] == verification["trials"] == len(verification["records"]) == len(campaign["plan"])
    records, trials = [], {}
    for spec in campaign["plan"]:
        folder = root/spec["folder"]
        require_audit_receipt(folder, spec, verification)
        report = json.loads((folder/"report.json").read_text())
        protocol = json.loads((folder/"protocol.json").read_text())
        assert not protocol["tracked_dirty"]
        assert protocol["git_commit"] == campaign["source_commit"]
        assert protocol["config"]["seed"] == spec["seed"]
        assert protocol["config"]["policy"] == spec["policy"]
        assert hashlib.sha256((folder/"input.qasm").read_bytes()).hexdigest() == protocol["input_sha256"]
        updates = rows(folder/"updates.jsonl")
        events = rows(folder/"events.jsonl")
        noops = sum(e["status"] == "anchor_noop" for e in events)
        gated = [e for e in events if e["status"] == "accepted"]
        first_best = (0. if report["root_cost"]["gates"] == report["best"]["cost"]["gates"] else
                      next((e["seconds"] for e in gated if e["cost"]["gates"] == report["best"]["cost"]["gates"]), None))
        t = report["timers"]; wall = report["total_loop_seconds"]
        row = dict(**spec, input_gates=protocol["input_gates"], root_gates=report["root_cost"]["gates"],
                   best_gates=report["best"]["cost"]["gates"], first_best_seconds=first_best,
                   wall_seconds=wall, transitions=report["transitions"], accepted=report["accepted"], noops=noops,
                   accepted_per_second=report["accepted"]/wall, noop_fraction=noops/max(1, len(events)),
                   updates=report["updates"], matching_and_views_seconds=t.get("matching", 0)+t.get("views", 0),
                   feasibility_seconds=t.get("feasibility", 0), rewrite_verify_seconds=t.get("rewrite_verify", 0),
                   score_seconds=t.get("score", 0), neural_seconds=t.get("anchor_inference", 0)+t.get("inference", 0)+t.get("ppo", 0),
                   previews=(report.get("feasibility") or {}).get("previews", 0),
                   unique_valid_qasm=report["completion"]["unique_valid_qasm"],
                   invalid_extractions=report["completion"]["unique_invalid_qasm"],
                   verification=report["verification_method"])
        if updates:
            for key in ("value_loss", "approx_kl", "value_explained_variance_after", "anchor_entropy", "conditional_action_entropy"):
                values = [u["metrics"].get(key) for u in updates if u["metrics"].get(key) is not None]
                row[key+"_last"] = values[-1] if values else None
            assert all(u["mean_loss"] is not None for u in updates)
        records.append(row)
        trials[spec["circuit"], spec["mode"], spec["seed"], spec["policy"]] = (row, protocol, events)
    pairs, learning, speeds = [], [], []
    for (circuit, mode, seed, policy), (online, po, eo) in trials.items():
        if policy != "online":
            continue
        frozen_key = (circuit, mode, seed, "frozen")
        if frozen_key in trials:
            frozen, pf, ef = trials[frozen_key]
            assert po["initial_tensor_sha256"] == pf["initial_tensor_sha256"]
            prefix = min(po["config"]["rollout"], len(eo), len(ef))
            keys = ("parent_graph_key", "restart_graph_key", "restart_source", "action", "available",
                    "candidate_sha256", "selection", "context", "anchor", "status", "cost", "reward", "terminal")
            for f, o in zip(ef[:prefix], eo[:prefix]):
                assert {k: f.get(k) for k in keys} == {k: o.get(k) for k in keys}
            pairs.append(dict(circuit=circuit, mode=mode, seed=seed, identical_prefix=prefix,
                              same_initialization=True))
            learning.append(dict(circuit=circuit, mode=mode, seed=seed,
                                 online_gates=online["best_gates"], frozen_gates=frozen["best_gates"],
                                 outcome="win" if online["best_gates"] < frozen["best_gates"] else
                                         "loss" if online["best_gates"] > frozen["best_gates"] else "tie"))
        baseline = trials.get((circuit, baseline_mode, seed, policy))
        if mode != baseline_mode and baseline:
            base = baseline[0]
            speeds.append(dict(circuit=circuit, mode=mode, seed=seed, policy=policy,
                               accepted_speedup=online["accepted_per_second"]/base["accepted_per_second"],
                               gate_delta=online["best_gates"]-base["best_gates"]))
    grouped = {}
    for mode in sorted({r["mode"] for r in records}):
        learning_rows = [r for r in learning if r["mode"] == mode]
        speed_rows = [r for r in speeds if r["mode"] == mode and r["policy"] == "online"]
        grouped[mode] = dict(online_vs_frozen={k: sum(r["outcome"] == k for r in learning_rows) for k in ("win", "tie", "loss")},
                             median_online_accepted_speedup=statistics.median(r["accepted_speedup"] for r in speed_rows) if speed_rows else None,
                             online_vs_baseline={k: sum((r["gate_delta"] < 0 if k == "win" else r["gate_delta"] > 0 if k == "loss" else r["gate_delta"] == 0)
                                                     for r in speed_rows) for k in ("win", "tie", "loss")})
        if baseline_mode == "global32":
            grouped[mode]["online_vs_global"] = grouped[mode]["online_vs_baseline"]
    output.mkdir(parents=True, exist_ok=False)
    fieldnames = list(dict.fromkeys(k for r in records for k in r))
    with (output/"trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    result = dict(status="passed", source_commit=campaign["source_commit"], trials=len(records), baseline_mode=baseline_mode,
                  grouped=grouped, paired_prefixes=pairs, learning=learning, performance=speeds,
                  records=records, caveats=["Finite on-target search; no generalization claim",
                  "Accepted throughput excludes noops but includes legal neutral rewrites",
                  "Different proposal distributions; speed does not isolate matching implementation alone",
                  "PPO loss and nonzero parameter changes alone do not establish learning benefit",
                  "Timing includes setup-excluded search budget, all selector/matcher/previews/verification/scoring/PPO work",
                  "Learning comparison uses total gate count only; CX/depth ties do not count as wins"],
                  evidence=dict(verification_sha256=hashlib.sha256((root/"audit/verification.json").read_bytes()).hexdigest()))
    (output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--baseline-mode")
    a = p.parse_args()
    r = summarize(a.campaign, a.output, a.baseline_mode)
    print(json.dumps(dict(trials=r["trials"], grouped=r["grouped"])))
