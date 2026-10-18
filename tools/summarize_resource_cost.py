"""Source-bound end-to-end resource optimizations and rule allocation results."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_anchor import summarize
from tools.summarize_rule_ablation import event_prefix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--node22", type=Path, required=True)
    p.add_argument("--node23", type=Path, required=True)
    p.add_argument("--component", type=Path, required=True)
    p.add_argument("--native-control", type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    components = json.loads(a.component.read_text())
    records = []; cohorts = {}; prefixes = []; summaries = []; training = []; learning_steps = []
    studies = [("node22", 22, a.node22, "cost_reference"), ("node23", 23, a.node23, "cost_bounds")]
    if a.native_control is not None:
        studies.append(("native-control", 22, a.native_control, "rules_native"))
    for cohort, node, root, baseline in studies:
        result = summarize(root, a.output/cohort, baseline)
        plan = json.loads((root/"campaign.json").read_text())
        initial = {}
        for spec in plan["plan"]:
            folder = root/spec["folder"]
            protocol = json.loads((folder/"protocol.json").read_text()); config = protocol["config"]
            assert config["selection"] == "global"
            assert config["rule_profile"] == ("native" if spec["mode"] == "rules_native" else "full")
            assert config["feasibility_mode"] == ("reference" if spec["mode"] in ("cost_reference", "rules_native") else "bounds")
            assert config["rule_schedule"] == ("staged" if "staged" in spec["mode"] else "static")
            assert config["candidate_allocation"] == ("native-heavy" if "quota" in spec["mode"] else "balanced")
            assert config["seconds"] == plan["seconds"]
            key = spec["circuit"], spec["seed"]
            assert initial.setdefault(key, protocol["initial_tensor_sha256"]) == protocol["initial_tensor_sha256"]
            for name, expected in components["runtime_sources"].items():
                assert protocol["sources"][name] == expected
        assert result["source_commit"] == components["source_commit"]
        local = []
        for row in result["records"]:
            folder = root/row["folder"]; report = json.loads((folder/"report.json").read_text())
            f = report["feasibility"]; vertices = []; phases = Counter()
            with (folder/"events.jsonl").open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["status"] in ("accepted", "accepted_unscored"):
                        vertices.append(event["vertices"])
                    if event["status"] != "excluded_after_deadline":
                        schedule = event.get("rule_schedule")
                        if schedule:
                            phases["expanded" if schedule["expanded"] else "native"] += 1
                            phases["fallback"] += schedule["native_empty_fallback"]
            row.update(node=node, cohort=cohort, bound_accepts=f.get("bound_accepts", 0),
                       proposed=f.get("proposed", 0), feasibility_cache_hits=f.get("cache_hits", 0),
                       previews_per_transition=row["previews"]/row["transitions"],
                       median_vertices=statistics.median(vertices),
                       expanded_schedule_steps=phases["expanded"], native_schedule_steps=phases["native"],
                       native_empty_fallbacks=phases["fallback"],
                       best_qasm_sha256=sha(folder/"best.qasm"), report_sha256=sha(folder/"report.json"),
                       updates_sha256=sha(folder/"updates.jsonl"))
            sampled = report["proposal_counts"]
            row["sampled_native"] = sum(v for k,v in sampled.items() if k in {"LC", "PV", "PVB", "PVG", "ID", "GF"})
            row["sampled_jpv"] = sum(v for k,v in sampled.items() if ":" in k)
            row["sampled_macro"] = sum(sampled.values())-row["sampled_native"]-row["sampled_jpv"]
            local.append(row); records.append(row)
            updates = [json.loads(line) for line in (folder/"updates.jsonl").read_text().splitlines()]
            if updates:
                entry = dict(node=node, cohort=cohort, mode=row["mode"], seed=row["seed"], updates=len(updates))
                for metric in ("value_loss", "value_explained_variance_after", "normalized_entropy", "approx_kl", "return_mean"):
                    values = [u["metrics"][metric] for u in updates if u["metrics"].get(metric) is not None]
                    assert all(math.isfinite(v) for v in values)
                    if values:
                        entry[metric+"_first25"] = statistics.mean(values[:25])
                        entry[metric+"_last25"] = statistics.mean(values[-25:])
                training.append(entry)
        modes = sorted({r["mode"] for r in local})
        for seed in sorted({r["seed"] for r in local}):
            for policy in ("frozen", "online"):
                rows = [next(r for r in local if r["seed"] == seed and r["policy"] == policy and r["mode"] == m) for m in modes]
                limit = max(1, (min(r["transitions"] for r in rows)//32-1)*32)
                values = {r["mode"]:event_prefix(root/r["folder"], limit) for r in rows}
                identical = len({v["semantics_sha256"] for v in values.values()}) == 1
                if cohort == "node22":
                    assert identical, (seed, policy, "Bounds altered the semantic prefix")
                for row in rows:
                    v = values[row["mode"]]
                    prefixes.append(dict(node=node, cohort=cohort, seed=seed, policy=policy, mode=row["mode"],
                        common_steps=limit, best_gates=v["best"], seconds=v["seconds"],
                        semantics_sha256=v["semantics_sha256"], identical_semantics=identical,
                        same_steps_speedup=values[baseline]["seconds"]/v["seconds"]))
        for mode in modes:
            rows = [r for r in local if r["mode"] == mode]
            for seed in sorted({r["seed"] for r in rows}):
                pair = {r["policy"]:r for r in rows if r["seed"] == seed}
                limit = max(1, (min(r["transitions"] for r in pair.values())//32-1)*32)
                costs = {pol:event_prefix(root/r["folder"], limit)["best"] for pol,r in pair.items()}
                difference = costs["online"]-costs["frozen"]
                learning_steps.append(dict(node=node, cohort=cohort, mode=mode, seed=seed, common_steps=limit,
                    frozen_gates=costs["frozen"], online_gates=costs["online"],
                    outcome="win" if difference < 0 else "loss" if difference > 0 else "tie"))
            base = {(r["seed"],r["policy"]):r for r in local if r["mode"] == baseline}
            differences = [r["best_gates"]-base[r["seed"],r["policy"]]["best_gates"] for r in rows]
            summaries.append(dict(node=node, cohort=cohort, mode=mode, trials=len(rows),
                frozen_gates="/".join(str(r["best_gates"]) for r in rows if r["policy"] == "frozen"),
                online_gates="/".join(str(r["best_gates"]) for r in rows if r["policy"] == "online"),
                mean_best_gates=statistics.mean(r["best_gates"] for r in rows), best_gates=min(r["best_gates"] for r in rows),
                median_accepted_per_second=statistics.median(r["accepted_per_second"] for r in rows),
                median_paired_speedup=statistics.median(r["accepted_per_second"]/base[r["seed"],r["policy"]]["accepted_per_second"] for r in rows),
                mean_previews_per_transition=statistics.mean(r["previews_per_transition"] for r in rows),
                wins=sum(v < 0 for v in differences), ties=sum(v == 0 for v in differences), losses=sum(v > 0 for v in differences)))
        cohorts[cohort] = dict(node=node, source_commit=result["source_commit"], trials=result["trials"],
            baseline_mode=baseline, evidence=result["evidence"], learning=result["learning"],
            config_sha256=sha(root/"campaign.json"))
    output = dict(cohorts=cohorts, records=records, summaries=summaries, common_prefixes=prefixes,
        training=training, learning_common_steps=learning_steps, components=components,
        component_sha256=sha(a.component), analyzer_sha256=sha(Path(__file__)),
        caveats=["Same target only, 600-second methodological ablation, not a three-hour plateau rerun",
                 "Compare performance within a node, not across nodes",
                 "Scheduling/allocation change search distribution; resource bounds must preserve it",
                 "Loss reduction and parameter updates alone do not establish useful learning"])
    (a.output/"analysis.json").write_text(json.dumps(output, indent=2)+"\n")
    for name, rows in (("trials", records), ("summary", summaries), ("common-prefixes", prefixes),
                       ("training", training), ("learning-common-steps", learning_steps)):
        write_csv(a.output/(name+".csv"), rows)
    print(json.dumps(dict(trials=len(records), strict_prefix_pairs=6, summaries=summaries)))


if __name__ == "__main__":
    main()
