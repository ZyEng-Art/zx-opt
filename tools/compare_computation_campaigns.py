"""Compare fully audited wall-budget campaigns differing only in named source files."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_anchor import require_audit_receipt


TRACE_FIELDS = ("transition", "trajectory", "trajectory_step", "parent_graph_key", "restart_graph_key",
                "restart_source", "action", "available", "candidate_sha256", "selection", "context", "anchor",
                "status", "graph_key", "cost", "sha256", "reward", "terminal", "auxiliary", "buffer", "best",
                "unscored_depth_before", "unscored_depth", "admitted")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root):
    campaign = json.loads((root/"campaign.json").read_text())
    finished = json.loads((root/"finished.json").read_text())
    audit = json.loads((root/"audit/verification.json").read_text())
    assert finished["status"] == audit["status"] == "passed"
    assert finished["trials"] == audit["trials"] == len(audit["records"]) == len(campaign["plan"])
    records = {}
    for spec in campaign["plan"]:
        folder = root/spec["folder"]
        require_audit_receipt(folder, spec, audit)
        protocol = json.loads((folder/"protocol.json").read_text())
        assert protocol["git_commit"] == campaign["source_commit"] and not protocol["tracked_dirty"]
        assert sha(folder/"input.qasm") == protocol["input_sha256"]
        report = json.loads((folder/"report.json").read_text())
        assert report["verification"] == "passed" and report["status"] == "budget_reached"
        key = (spec["circuit"], spec["mode"], spec["policy"], spec["seed"])
        assert key not in records
        records[key] = (folder, protocol, report)
    return campaign, records


def matching_prefix(left, right):
    count = 0; difference = None
    with (left/"events.jsonl").open() as a, (right/"events.jsonl").open() as b:
        for l, r in zip(a, b):
            old, new = json.loads(l), json.loads(r)
            if "excluded_after_deadline" in (old["status"], new["status"]):
                break
            changed = [key for key in TRACE_FIELDS if old.get(key) != new.get(key)]
            if changed:
                difference = dict(transition=old["transition"], fields=changed,
                                  reference_status=old["status"], updated_status=new["status"])
                break
            count += 1
    return count, difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path); p.add_argument("updated", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--changed-source", nargs="+", required=True)
    a = p.parse_args()
    old_campaign, old = load(a.reference); new_campaign, new = load(a.updated)
    assert old.keys() == new.keys()
    pairs = []
    for key, (left, lp, lr) in sorted(old.items()):
        right, rp, rr = new[key]
        assert lp["input_sha256"] == rp["input_sha256"]
        assert lp["initial_tensor_sha256"] == rp["initial_tensor_sha256"]
        assert lp["config"] == rp["config"], key
        assert lp["sources"].keys() == rp["sources"].keys()
        changed = {name for name in lp["sources"] if lp["sources"][name] != rp["sources"][name]}
        assert changed == set(a.changed_source), (key, changed)
        prefix, difference = matching_prefix(left, right)
        accepted = [(r["accepted"]/r["total_loop_seconds"]) for r in (lr, rr)]
        transitions = [(r["transitions"]/r["total_loop_seconds"]) for r in (lr, rr)]
        pairs.append(dict(circuit=key[0], mode=key[1], policy=key[2], seed=key[3],
                          reference_gates=lr["best"]["cost"]["gates"], updated_gates=rr["best"]["cost"]["gates"],
                          reference_accepted_per_second=accepted[0], updated_accepted_per_second=accepted[1],
                          accepted_speedup=accepted[1]/accepted[0], transitions_speedup=transitions[1]/transitions[0],
                          identical_trace_prefix=prefix, first_trace_difference=difference,
                          reference_transitions=lr["transitions"], updated_transitions=rr["transitions"],
                          reference_protocol_sha256=sha(left/"protocol.json"), updated_protocol_sha256=sha(right/"protocol.json")))
    groups = {}
    for policy in sorted({row["policy"] for row in pairs}):
        rows = [r for r in pairs if r["policy"] == policy]
        speeds = [r["accepted_speedup"] for r in rows]
        groups[policy] = dict(pairs=len(rows), median_accepted_speedup=statistics.median(speeds),
                              minimum_accepted_speedup=min(speeds), maximum_accepted_speedup=max(speeds),
                              gate_wins=sum(r["updated_gates"] < r["reference_gates"] for r in rows),
                              gate_ties=sum(r["updated_gates"] == r["reference_gates"] for r in rows),
                              gate_losses=sum(r["updated_gates"] > r["reference_gates"] for r in rows))
    result = dict(status="completed_audited_computation_comparison", pairs=pairs, by_policy=groups,
                  reference_source=old_campaign["source_commit"], updated_source=new_campaign["source_commit"],
                  changed_sources=a.changed_source,
                  reference_audit_sha256=sha(a.reference/"audit/verification.json"),
                  updated_audit_sha256=sha(a.updated/"audit/verification.json"),
                  scope="Same planned cases, input, exact initial tensor digest, search config and wall budget; source changes restricted to the named files. All accepted rewrites/circuits/model files have complete audit receipts. Accepted throughput includes neutral legal rewrites. Whole trace prefixes are compared separately; any divergence is reported rather than hidden. Small on-target computation comparison, not evidence of useful learning/generalization.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status=result["status"], by_policy=groups)))


if __name__ == "__main__":
    main()
