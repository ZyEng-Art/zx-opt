"""Descriptive rule usage, direct deltas and first-creation best ancestries.

Immediate gate decreases can undo earlier increases. Presence on a best path
does not establish necessity: neither statistic replaces a rule ablation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def analyze(trial):
    protocol = json.loads((trial/"protocol.json").read_text())
    report = json.loads((trial/"report.json").read_text())
    root = protocol["root_graph"]; best = report["best"]["graph"]
    parents = {}; direct = defaultdict(Counter); improvements = []
    current = report["root_cost"]["gates"]; digest = hashlib.sha256()
    total = 0; accepted = 0
    with (trial/"events.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line); row = json.loads(line); total += 1
            if row["status"] not in ("accepted", "accepted_unscored"):
                continue
            accepted += 1
            child = row["graph_key"]; parent = row["parent_graph_key"]
            family = row["action"]["family"]; mode = row["action"]["mode"]
            cost = row["cost"]; previous = row["parent_cost"]
            c = None if cost is None else cost["gates"]
            p = None if previous is None else previous["gates"]
            edge = dict(parent=parent, child=child, family=family, mode=mode,
                        parent_gates=p, gates=c, transition=row["transition"], seconds=row["seconds"])
            if child != root:
                parents.setdefault(child, edge)
            d = direct[family]; d["accepted"] += 1
            if c is None:
                d["unscored"] += 1
            elif p is not None:
                d["scored_parent_and_child"] += 1
                d["down" if c < p else "up" if c > p else "equal"] += 1
                d["sum_parent_minus_child"] += p-c
            if c is not None and c < current:
                improvements.append(dict(**edge, previous_best=current, improvement=current-c))
                current = c
    assert accepted == report["accepted"]
    assert current == report["best"]["cost"]["gates"]
    path = []; cursor = best; seen = set()
    while cursor != root:
        assert cursor not in seen; seen.add(cursor)
        edge = parents[cursor]; path.append(edge); cursor = edge["parent"]
    path.reverse()
    return dict(circuit=protocol["circuit"], source_commit=protocol["git_commit"],
        events_sha256=digest.hexdigest(), events=total, accepted=accepted,
        protocol_sha256=hashlib.sha256((trial/"protocol.json").read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256((trial/"report.json").read_bytes()).hexdigest(),
        root_gates=report["root_cost"]["gates"], best_gates=current,
        direct={k: dict(v) for k, v in direct.items()}, strict_best_improvements=improvements,
        best_ancestry=path, best_ancestry_families=dict(Counter(e["family"] for e in path)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args()
    rows = [analyze(path.parent) for path in sorted(a.study.glob("*/jpv_online_s11/report.json"))]
    assert len(rows) == 13
    with a.output.open("x") as stream:
        json.dump(dict(scope=__doc__, rows=rows), stream, indent=2); stream.write("\n")


if __name__ == "__main__":
    main()
