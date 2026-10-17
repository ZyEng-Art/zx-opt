"""Replay archived ZX states and measure extraction sensitivity to vertex order.

This diagnostic is independent of live search. It never supplies initial states
or checkpoints to a campaign. Every extraction is checked on fixed named wires.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyzx as zx
from qiskit import qasm2
import torch

from zxopt.dedup import StructuralIndex
from zxopt.kernel import Kernel
from zxopt.legacy import environment, oracle, ROOT
from zxopt.rules import Action
from zxopt.verify import graph_matrix, matrix_error, qiskit_matrix_zx_order
from runtime import graph_key


def paths(events, targets):
    root = events[0]["parent_graph_key"]
    parents = {}
    for event in events:
        if event["status"] == "accepted" and event["graph_key"] != root:
            parents.setdefault(event["graph_key"], event)
    result = []
    for target in targets:
        chain, seen, key = [], set(), target["graph_key"]
        while key != root:
            if key in seen:
                raise AssertionError("Archived ancestry cycle")
            seen.add(key)
            event = parents[key]
            chain.append({k: event[k] for k in ("action", "legacy_identifier",
                         "parent_graph_key", "graph_key", "cost", "transition")})
            key = event["parent_graph_key"]
        result.append(dict(state=target["state"], graph_key=target["graph_key"],
                           original_cost=target["cost"], steps=list(reversed(chain))))
    return root, result


def rebuild(graph, order):
    """Retain the ordered boundaries, phases, edge types, scalar and layout.

    Reconstruct only the physical diagram; phase-teleportation bookkeeping is
    intentionally reset. The scalar is copied directly without JSON rounding.
    """
    new = zx.Graph()
    mapping = {v: new.add_vertex(graph.type(v), graph.qubit(v), graph.row(v),
                               graph.phase(v), graph.is_ground(v)) for v in order}
    for edge in graph.edges():
        a, b = graph.edge_st(edge)
        new.add_edge(new.edge(mapping[a], mapping[b]), graph.edge_type(edge))
    new.set_inputs(tuple(mapping[v] for v in graph.inputs()))
    new.set_outputs(tuple(mapping[v] for v in graph.outputs()))
    new.scalar = graph.scalar.copy()
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=int, help="Compare cheapest and most expensive observed representations of this class")
    p.add_argument("--best-states", type=int, default=8)
    p.add_argument("--variants", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261017)
    args = p.parse_args()
    if args.variants < 1 or args.best_states < 1:
        raise ValueError("Expected positive probe bounds")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "candidates").mkdir()
    torch.set_num_threads(1)
    protocol = json.loads((args.run / "protocol.json").read_text())
    random.seed(protocol["config"]["seed"])
    np.random.seed(protocol["config"]["seed"])
    torch.manual_seed(protocol["config"]["seed"])
    input_file = ROOT / "inputs/barenco_tof_3.qasm"
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if protocol["input_sha256"] != digest:
        raise AssertionError("Unexpected input circuit")
    events = [json.loads(line) for line in (args.run / "events.jsonl").read_text().splitlines()]
    accepted = [e for e in events if e["status"] == "accepted"]
    if args.state is not None:
        group = [e for e in accepted if e["state"] == args.state]
        targets = [min(group, key=lambda e: (e["cost"]["gates"], e["cost"]["cx"])),
                   max(group, key=lambda e: (e["cost"]["gates"], e["cost"]["cx"]))]
    else:
        targets, seen = [], set()
        for event in sorted(accepted, key=lambda e: (e["cost"]["gates"], e["cost"]["cx"], e["transition"])):
            if event["state"] in seen:
                continue
            targets.append(event)
            seen.add(event["state"])
            if len(targets) == args.best_states:
                break
    root_key, witnesses = paths(events, targets)
    witness = dict(source_run=str(args.run), source_commit=protocol["git_commit"],
                   input_sha256=digest, root_graph_key=root_key, witnesses=witnesses,
                   scope="Offline diagnostic only; never seeded into search")
    (args.output / "witness.json").write_text(json.dumps(witness, indent=2) + "\n")
    env = environment(input_file)
    kernel, scorer = Kernel(env), oracle()
    root = env.graph.clone()
    if graph_key(root) != root_key:
        raise AssertionError("Archived representation identity differs in this environment")
    reference = qiskit_matrix_zx_order(qasm2.loads(input_file.read_text()))
    rng = random.Random(args.seed)
    rows, global_best, max_error = [], None, 0.
    started = time.perf_counter()
    for i, trial in enumerate(witnesses):
        graph = root.clone()
        for step in trial["steps"]:
            if graph_key(graph) != step["parent_graph_key"]:
                raise AssertionError("Parent replay mismatch: " + str(step["transition"]))
            a = step["action"]
            graph = kernel.step(graph, Action(a["kind"], tuple(a["vertices"]),
                                Fraction(a["phase"]), tuple(a["subset"])), step["legacy_identifier"])
            if graph_key(graph) != step["graph_key"]:
                raise AssertionError("Child replay mismatch: " + str(step["transition"]))
        _, original, _ = scorer.evaluate(graph)
        if original["cost"] != trial["original_cost"]:
            raise AssertionError("Archived extraction cost differs in this environment")
        original_error = max(matrix_error(reference, graph_matrix(graph)),
                             matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(original["qasm"]))))
        if original_error >= 2e-8:
            raise AssertionError("Original replay operator mismatch")
        max_error = max(max_error, original_error)
        (args.output / "candidates" / (original["sha256"] + ".qasm")).write_text(original["qasm"])
        index = StructuralIndex()
        index.intern(graph)
        records, best = [], original
        for j in range(args.variants):
            order = list(graph.vertices())
            rng.shuffle(order)
            variant = rebuild(graph, order)
            if index.intern(variant)[2]:
                raise AssertionError("Reconstruction changed physical graph structure")
            graph_error = matrix_error(reference, graph_matrix(variant))
            if graph_error >= 2e-8:
                raise AssertionError("Reconstructed graph operator mismatch")
            _, candidate, _ = scorer.evaluate(variant)
            qasm_error = matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(candidate["qasm"])))
            if qasm_error >= 2e-8:
                raise AssertionError("Extracted circuit operator mismatch")
            max_error = max(max_error, graph_error, qasm_error)
            (args.output / "candidates" / (candidate["sha256"] + ".qasm")).write_text(candidate["qasm"])
            records.append(dict(variant=j, cost=candidate["cost"], sha256=candidate["sha256"], order=order))
            if (candidate["cost"]["gates"], candidate["cost"]["cx"]) < (best["cost"]["gates"], best["cost"]["cx"]):
                best = candidate
        if global_best is None or (best["cost"]["gates"], best["cost"]["cx"]) < (global_best["cost"]["gates"], global_best["cost"]["cx"]):
            global_best = best
        row = dict(state=trial["state"], source_graph_key=trial["graph_key"],
                   replay_steps=len(trial["steps"]), original_cost=original["cost"],
                   best_cost=best["cost"], best_sha256=best["sha256"],
                   variants=args.variants, records=records,
                   gates_histogram={str(n): sum(r["cost"]["gates"] == n for r in records)
                                    for n in sorted({r["cost"]["gates"] for r in records})})
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "records"}), flush=True)
    (args.output / "best.qasm").write_text(global_best["qasm"])
    report = dict(status="passed", scope=witness["scope"], input_sha256=digest,
                  source_run=str(args.run), source_commit=protocol["git_commit"],
                  probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  events_sha256=hashlib.sha256((args.run/"events.jsonl").read_bytes()).hexdigest(),
                  seed=args.seed, variants_per_state=args.variants, max_operator_error=max_error,
                  total_seconds=time.perf_counter()-started, best_cost=global_best["cost"], rows=rows,
                  caveat="Reconstruction changes numbering, insertion order and teleportation metadata together; no isolated causal claim about numbering alone")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
