"""Compare exact native audit paths on a reservoir sample of recorded rewrites."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--samples", type=int, default=200)
    a = p.parse_args(); folder = a.trial.resolve()
    import torch
    from qiskit import qasm2
    from tools.audit_general import replay_native_rooted
    from zxopt.complete import Diagram
    from zxopt.general_kernel import GeneralKernel, HybridAction, Views
    from zxopt.legacy import environment
    from zxopt.scalable_verify import reference_for
    torch.set_num_threads(1)
    rng = random.Random(29); selected = []; seen = 0
    with (folder/"events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["status"] != "accepted" or event["action"]["mode"] != "legacy":
                continue
            seen += 1
            if len(selected) < a.samples:
                selected.append(event)
            else:
                index = rng.randrange(seen)
                if index < a.samples:
                    selected[index] = event
    selected.sort(key=lambda e: e["transition"])
    protocol = json.loads((folder/"protocol.json").read_text())
    text = (folder/"input.qasm").read_text(); n = qasm2.loads(text).num_qubits
    reference = reference_for(text, protocol["config"].get("verification", "matrix"))
    views = Views(n, reference); kernel = GeneralKernel(environment(folder/"input.qasm"), jpv=False)
    reference_seconds = rooted_seconds = 0.; kinds = {}
    for i, event in enumerate(selected):
        raw = Diagram.from_record(json.loads(gzip.decompress((folder/"objects"/(event["parent_graph_key"]+".json.gz")).read_bytes())))
        view = views.get(raw); action = HybridAction.from_record(event["action"])
        def full(): return kernel.step(raw, view, action)
        def rooted(): return replay_native_rooted(kernel, raw, view, action, event.get("anchor"))
        calls = [("reference", full), ("rooted", rooted)]
        if i % 2: calls.reverse()
        results = {}
        for name, function in calls:
            start = time.perf_counter(); result = function(); elapsed = time.perf_counter()-start
            if name == "reference": reference_seconds += elapsed
            else: rooted_seconds += elapsed
            results[name] = result
        left, lr = results["reference"]; right, rr = results["rooted"]
        assert left.record() == right.record() and lr == rr
        assert left.digest() == event["graph_key"] and lr == event["auxiliary"]
        assert views.get(right).error < 2e-8
        kinds[action.family] = kinds.get(action.family, 0)+1
    result = dict(status="passed", trial=str(folder), sampled_native_rewrites=len(selected),
                  all_native_rewrites=seen, families=kinds, reference_seconds=reference_seconds,
                  rooted_seconds=rooted_seconds, speedup=reference_seconds/rooted_seconds if rooted_seconds else None,
                  transitions=[e["transition"] for e in selected],
                  events_sha256=hashlib.sha256((folder/"events.jsonl").read_bytes()).hexdigest(),
                  scope="One-host alternating-order native-replay benchmark only. Graph loading/normalization/operator verification excluded from timing; they remain checked. Same child records/HH receipts and recorded graph hashes; not a training/full-audit speedup claim.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "transitions"}))


if __name__ == "__main__":
    main()
