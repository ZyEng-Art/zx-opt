"""Compare audited online/frozen searches at equal sampled-action counts.

This descriptive comparison complements the primary equal-wall-budget result.
Different policies can have different action costs and restart distributions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_anchor import require_audit_receipt


def best_prefix(folder, limit, root_gates):
    best = root_gates
    first_best = 0
    with (folder/"events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["transition"] > limit:
                break
            if event["status"] == "accepted" and event["cost"]["gates"] < best:
                best = event["cost"]["gates"]
                first_best = event["transition"]
    return dict(gates=best, first_best_transition=first_best)


def outcome(online, frozen):
    return "win" if online < frozen else "loss" if online > frozen else "tie"


def compare(root):
    campaign = json.loads((root/"campaign.json").read_text())
    finished = json.loads((root/"finished.json").read_text())
    audit_path = root/"audit/verification.json"
    audit = json.loads(audit_path.read_text())
    if not (finished["status"] == audit["status"] == "passed"
            and finished["trials"] == audit["trials"] == len(audit["records"]) == len(campaign["plan"])):
        raise ValueError("Whole campaign has not passed its complete replay audit")
    index = {(s["circuit"], s["mode"], s["seed"], s["policy"]): s for s in campaign["plan"]}
    if len(index) != len(campaign["plan"]):
        raise ValueError("Duplicate comparison cases")
    pairs = []
    for key, online in index.items():
        if key[-1] != "online":
            continue
        frozen = index[key[:-1]+("frozen",)]
        reports = []; protocols = []
        for spec in (online, frozen):
            folder = root/spec["folder"]
            require_audit_receipt(folder, spec, audit)
            report = json.loads((folder/"report.json").read_text())
            protocol = json.loads((folder/"protocol.json").read_text())
            if protocol["config"].get("shared_buffer"):
                raise ValueError("External circuit imports are outside this comparison")
            if protocol["tracked_dirty"] or protocol["git_commit"] != campaign["source_commit"]:
                raise ValueError("Unsealed source")
            reports.append(report); protocols.append(protocol)
        if not (protocols[0]["initial_tensor_sha256"] == protocols[1]["initial_tensor_sha256"]
                and protocols[0]["input_sha256"] == protocols[1]["input_sha256"]
                and reports[0]["root_cost"] == reports[1]["root_cost"]):
            raise ValueError("Different initial search conditions")
        common = min(r["transitions"] for r in reports)
        prefixes = [best_prefix(root/spec["folder"], common, report["root_cost"]["gates"])
                    for spec, report in zip((online, frozen), reports)]
        final = [r["best"]["cost"]["gates"] for r in reports]
        pairs.append(dict(circuit=key[0], mode=key[1], seed=key[2],
                          common_sampled_actions=common,
                          online_transitions=reports[0]["transitions"], frozen_transitions=reports[1]["transitions"],
                          online_prefix=prefixes[0], frozen_prefix=prefixes[1],
                          equal_steps_outcome=outcome(prefixes[0]["gates"], prefixes[1]["gates"]),
                          online_final_gates=final[0], frozen_final_gates=final[1],
                          equal_wall_outcome=outcome(*final)))
    return dict(status="passed", source_commit=campaign["source_commit"], pairs=pairs,
                by_mode={mode: {kind: dict(Counter(p[kind] for p in pairs if p["mode"] == mode))
                                for kind in ("equal_steps_outcome", "equal_wall_outcome")}
                         for mode in sorted({p["mode"] for p in pairs})},
                verification_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="Completed audited on-target trials only. Common sampled-action prefix includes noops/rejections and excludes unaccepted deadline candidates. Equal steps do not equal compute. This complements the equal-wall primary result; related frozen controls introduce dependence. No generalization or convergence claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status=result["status"], by_mode=result["by_mode"])))
