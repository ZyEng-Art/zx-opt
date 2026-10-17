"""Compare exact graph exports against a saved pre-optimization adapter."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_adapter", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("trials", type=Path, nargs="+")
    p.add_argument("--samples-per-trial", type=int, default=100)
    a = p.parse_args()
    if a.samples_per_trial < 1:
        raise ValueError("A positive sample count is required")
    import torch
    from qiskit import qasm2
    import zxopt.complete.adapter as adapter
    from zxopt.complete import Diagram
    from zxopt.scalable_verify import reference_for, check_graph
    torch.set_num_threads(1)
    # Execute only the saved function, sharing the current unchanged helpers.
    # This isolates layout changes; the import-hoisting gain is not timed here.
    tree = ast.parse(a.reference_adapter.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "to_pyzx"]
    if len(functions) != 1:
        raise ValueError("Reference adapter must contain exactly one to_pyzx function")
    namespace = dict(vars(adapter))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(a.reference_adapter), "exec"), namespace)
    reference_export = namespace["to_pyzx"]
    rng = random.Random(619); reference_seconds = indexed_seconds = 0.; max_error = 0.; cases = []
    total = 0
    for folder in a.trials:
        protocol = json.loads((folder/"protocol.json").read_text())
        text = (folder/"input.qasm").read_text(); n = qasm2.loads(text).num_qubits
        operator = reference_for(text, protocol["config"].get("verification", "matrix"))
        keys = dict.fromkeys([protocol["root_graph"]])
        with (folder/"events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["status"] in ("accepted", "accepted_unscored"):
                    keys[event["graph_key"]] = None
        selected = sorted(rng.sample(list(keys), min(len(keys), a.samples_per_trial)))
        for key in selected:
            raw = Diagram.from_record(json.loads(gzip.decompress((folder/"objects"/(key+".json.gz")).read_bytes())))
            assert raw.digest() == key
            functions = [("reference", reference_export), ("indexed", adapter.to_pyzx)]
            if total % 2:
                functions.reverse()
            results = {}
            for name, function in functions:
                started = time.perf_counter(); graph = function(raw, n, evaluate_expressions=True)
                elapsed = time.perf_counter()-started
                if name == "reference": reference_seconds += elapsed
                else: indexed_seconds += elapsed
                results[name] = graph
            assert results["reference"].to_json() == results["indexed"].to_json(), key
            error = check_graph(operator, results["indexed"]); assert error < 2e-8
            max_error = max(max_error, error); total += 1
        cases.append(dict(trial=str(folder.resolve()), sampled=len(selected), available=len(keys),
                          events_sha256=sha(folder/"events.jsonl"), graph_keys=selected))
    result = dict(status="passed", sampled_graphs=total, reference_seconds=reference_seconds,
                  indexed_seconds=indexed_seconds, export_speedup=reference_seconds/indexed_seconds,
                  max_graph_operator_error=max_error, cases=cases,
                  reference_adapter_sha256=sha(a.reference_adapter),
                  current_adapter_sha256=sha(ROOT/"zxopt/complete/adapter.py"),
                  current_ring_sha256=sha(ROOT/"zxopt/complete/ring.py"),
                  scope="Alternating-order export-only benchmark on recorded raw diagrams. Exact full exported JSON including IDs/edges/phases/rows/qubits/boundaries/scalar; independent fixed-wire operator checks outside timer. Both export functions share current other helpers, so import-hoisting gains are excluded. Not a training/full-audit speedup claim.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
