"""Combine descriptive history, paired component timings and audited ablations."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_anchor import require_audit_receipt

NATIVE = {"LC", "PV", "PVB", "PVG", "ID", "GF"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(family):
    return "native" if family in NATIVE else "jpv" if ":" in family else "macros"


def history(evidence):
    old = {}; rows = []; selected = Counter(); accepted = Counter(); timers = Counter(); feasibility = Counter(); wall = 0.
    for node in (22, 23):
        for row in json.loads((ROOT/"reports/anchor-first-20261018/original-nam26"/("node"+str(node)+".json")).read_text())["records"]:
            old[row["circuit"]] = row
            selected.update(row["report"]["proposal_counts"])
            accepted.update(row["report"]["accepted_counts"])
            timers.update(row["report"]["timers"]); wall += row["report"]["total_loop_seconds"]
            feasibility.update({k: row["report"]["feasibility"].get(k, 0) for k in
                                ("previews", "calls", "proposed", "eligible", "resource_masks", "cache_hits")})
        rows += json.loads((evidence/("usage-node"+str(node)+".json")).read_text())["rows"]
    groups = defaultdict(Counter); families = defaultdict(Counter)
    for family, n in selected.items():
        families[family]["sampled"] += n; groups[group(family)]["sampled"] += n
    for family, n in accepted.items():
        families[family]["accepted"] += n; groups[group(family)]["accepted"] += n
    for row in rows:
        original = old[row["circuit"]]
        assert all(row[k] == original[k] for k in ("protocol_sha256", "report_sha256"))
        for f, n in row["best_ancestry_families"].items():
            families[f]["best_ancestry_actions"] += n; groups[group(f)]["best_ancestry_actions"] += n
        for g in ("native", "macros", "jpv"):
            groups[g]["circuits_with_best_ancestry"] += any(group(f) == g for f in row["best_ancestry_families"])
        for e in row["strict_best_improvements"]:
            for target in (groups[group(e["family"])], families[e["family"]]):
                target["strict_best_final_action_events"] += 1
                target["strict_best_final_action_gate_delta"] += e["improvement"]
        for f, d in row["direct"].items():
            for k in ("scored_parent_and_child", "up", "equal", "down", "unscored"):
                families[f][k] += d.get(k, 0)
    return dict(circuits=len(rows), sampled=sum(selected.values()), accepted=sum(accepted.values()),
        groups={k: dict(v) for k, v in groups.items()}, families={k: dict(v) for k, v in families.items()},
        summed_search_seconds=wall, timers=dict(timers),
        feasibility=dict(feasibility),
        feasibility_cache_hit_fraction=feasibility["cache_hits"]/feasibility["proposed"],
        timer_percent={k: 100*v/wall for k, v in timers.items()},
        caveat="Final improvement actions and path presence describe observed traces, not necessary causal rule benefits. Matching includes view normalization.")


def components(evidence):
    result = []
    for path in sorted(evidence.glob("*.strict-warm-profile.json")):
        data = json.loads(path.read_text()); n = data["states"]*data["repeats"]
        assert sha(evidence/(data["circuit"]+".pack.json")) == data["pack_sha256"]
        for profile, value in data["totals"].items():
            prep = value["prepare_seconds"]; feas = value["feasibility_seconds"]
            a = sum(v for k, v in (value["jpv"] or {}).get("profile_seconds", {}).items() if k.startswith("A:"))
            result.append(dict(circuit=data["circuit"], profile=profile, states=data["states"], repeats=data["repeats"],
                prepare_ms=1000*prep/n, feasibility_ms=1000*feas/n, total_ms=1000*(prep+feas)/n,
                a_ms=1000*a/n, a_prepare_percent=100*a/prep if prep else 0,
                a_total_percent=100*a/(prep+feas) if prep+feas else 0,
                candidates=value["candidates"], eligible=value["eligible"],
                raw_profile_sha256=sha(path), runtime_sources=data["runtime_sources"],
                profiler_sha256=data["profiler_sha256"]))
    return result


def event_prefix(folder, limit):
    """Hash event semantics, excluding wall-time/cache implementation counters."""
    keys = ("parent_graph_key", "restart_graph_key", "restart_source", "action", "available",
            "candidate_sha256", "selection", "context", "status", "graph_key", "cost", "reward", "terminal")
    digest = hashlib.sha256(); best = json.loads((folder/"protocol.json").read_text())["root_cost"]["gates"]
    end = 0.; count = 0
    with (folder/"events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["transition"] > limit:
                break
            if event["status"] == "excluded_after_deadline":
                continue
            digest.update(json.dumps({k: event.get(k) for k in keys}, sort_keys=True).encode())
            count += 1; end = event["seconds"]
            if event["status"] == "accepted":
                gates = event["cost"]["gates"]
                best = min(best, gates)
    return dict(transitions=count, seconds=end, best=best, semantics_sha256=digest.hexdigest())


def campaign(root, strict_a=False):
    plan = json.loads((root/"campaign.json").read_text()); audit_path = root/"audit/verification.json"
    audit = json.loads(audit_path.read_text()); finished = json.loads((root/"finished.json").read_text())
    assert audit["status"] == finished["status"] == "passed"
    assert len(plan["plan"]) == audit["trials"] == finished["trials"] == len(audit["records"])
    rows = []; index = {}; initial = {}
    for spec in plan["plan"]:
        folder = root/spec["folder"]; require_audit_receipt(folder, spec, audit)
        protocol = json.loads((folder/"protocol.json").read_text()); report = json.loads((folder/"report.json").read_text())
        assert protocol["git_commit"] == plan["source_commit"] and not protocol["tracked_dirty"]
        assert protocol["config"]["seconds"] == plan["seconds"]
        assert protocol["config"]["rule_profile"] == spec["mode"][6:].replace("_", "-")
        key = (spec["circuit"], spec["seed"])
        assert initial.setdefault(key, protocol["initial_tensor_sha256"]) == protocol["initial_tensor_sha256"]
        row = dict(**spec, gates=report["best"]["cost"]["gates"], steps=report["transitions"],
            accepted=report["accepted"], seconds=report["total_loop_seconds"],
            accepted_per_second=report["accepted"]/report["total_loop_seconds"], timers=report["timers"],
            a_sampled=sum(v for k,v in report["proposal_counts"].items() if k.startswith("A:")),
            initial_tensor_sha256=protocol["initial_tensor_sha256"],
            report_sha256=sha(folder/"report.json"), protocol_sha256=sha(folder/"protocol.json"))
        vertices = []; edges = []; cache_hits = 0; neutral = comparable = 0
        with (folder/"events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["status"] not in ("accepted", "accepted_unscored"):
                    continue
                vertices.append(event["vertices"]); edges.append(event["edges"])
                cache_hits += bool(event.get("cache_hit"))
                if event.get("cost") is not None and event.get("parent_cost") is not None:
                    comparable += 1
                    neutral += event["cost"]["gates"] == event["parent_cost"]["gates"]
        row.update(median_vertices=statistics.median(vertices), median_edges=statistics.median(edges),
                   cost_cache_hit_fraction=cache_hits/len(vertices),
                   neutral_gate_fraction=neutral/comparable if comparable else None)
        rows.append(row); index[spec["seed"], spec["policy"], spec["mode"]] = row
    common = []; modes = sorted({r["mode"] for r in rows})
    for seed in sorted({r["seed"] for r in rows}):
        for policy in ("frozen", "online"):
            subset = [index[seed, policy, m] for m in modes]
            # Leave at least one rollout before the wall-time tail out of prefix comparisons.
            limit = max(1, (min(r["steps"] for r in subset)//32-1)*32)
            prefixes = {r["mode"]: event_prefix(root/r["folder"], limit) for r in subset}
            paired = len({v["semantics_sha256"] for v in prefixes.values()}) == 1
            common.append(dict(seed=seed, policy=policy, common_steps=limit, prefixes=prefixes,
                identical_semantics=paired,
                full_over_no_a_time=(prefixes["rules_full"]["seconds"]/prefixes["rules_no_a"]["seconds"]
                    if strict_a and paired else None)))
    return dict(source_commit=plan["source_commit"], trials=len(rows), rows=rows, common_prefixes=common,
        audit_sha256=sha(audit_path), budget_seconds=plan["seconds"],
        scope="Fixed-budget on-target rule analysis; separate from per-circuit three-hour patience protocol")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path); p.add_argument("--campaign", type=Path)
    p.add_argument("--strict-a-campaign", type=Path)
    a = p.parse_args()
    result = dict(history=history(a.evidence), components=components(a.evidence), analyzer_sha256=sha(Path(__file__)))
    if a.campaign:
        result["four_layer_campaign"] = campaign(a.campaign)
    if a.strict_a_campaign:
        result["strict_a_campaign"] = campaign(a.strict_a_campaign, True)
    (a.evidence/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    for key in ("components",):
        rows = result[key]
        if rows:
            with (a.evidence/(key+".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "runtime_sources"])
                writer.writeheader(); writer.writerows({k:v for k,v in r.items() if k != "runtime_sources"} for r in rows)
    print(json.dumps({"history_circuits":result["history"]["circuits"], "component_rows":len(result["components"])}))


if __name__ == "__main__":
    main()
