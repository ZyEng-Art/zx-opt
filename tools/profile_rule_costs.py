"""Paired rule generation/preview timing on identical recorded ZX states.

Pack mode needs only the standard library. Profiling neither trains nor scores
new circuits. Cold feasibility masks avoid confounding rule time with cache hits.
"""
import argparse
from collections import Counter, defaultdict
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


def pack(trial, output, count):
    protocol = json.loads((trial/"protocol.json").read_text())
    report = json.loads((trial/"report.json").read_text())
    assert sha(trial/"input.qasm") == protocol["input_sha256"]
    keys = [protocol["root_graph"]]; seen = set(keys)
    with (trial/"events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] in ("accepted", "accepted_unscored") and row["graph_key"] not in seen:
                keys.append(row["graph_key"]); seen.add(row["graph_key"])
    chosen = [keys[round(i*(len(keys)-1)/max(1, count-1))] for i in range(count)]
    best = report["best"].get("graph", report["best"].get("graph_key", report["best"].get("key")))
    if best and best not in chosen:
        chosen[-1] = best
    states = []
    for key in dict.fromkeys(chosen):
        path = trial/"objects"/(key+".json.gz")
        states.append(dict(key=key, record=json.loads(gzip.decompress(path.read_bytes())),
                           object_sha256=sha(path)))
    value = dict(circuit=protocol["circuit"], source_commit=protocol["git_commit"],
        protocol_sha256=sha(trial/"protocol.json"), report_sha256=sha(trial/"report.json"),
        events_sha256=sha(trial/"events.jsonl"), input_sha256=protocol["input_sha256"],
        input_qasm=(trial/"input.qasm").read_text(), config=protocol["config"],
        root_graph=protocol["root_graph"], states=states,
        sampling="root and equidistant unique accepted state indices, last replaced by best if absent")
    with output.open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")


def profile(pack_path, output, repeats, warmup=False):
    import torch
    from zxopt.complete import Diagram
    from zxopt.general_kernel import GeneralKernel, Views
    from zxopt.general_feasibility import Feasibility
    from zxopt.legacy import environment
    from zxopt.legacy import ROOT as code_root
    from zxopt.scalable_verify import reference_for
    torch.set_num_threads(1)
    data = json.loads(pack_path.read_text()); config = data["config"]
    assert hashlib.sha256(data["input_qasm"].encode()).hexdigest() == data["input_sha256"]
    # environment needs a real QASM path; preserve the exact packed input bytes.
    import tempfile
    rows = []; warmups = []; totals = {}; previews = defaultdict(Counter)
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp)/"input.qasm"; input_path.write_text(data["input_qasm"])
        reference = reference_for(data["input_qasm"], config.get("verification", "matrix"))
        env = environment(input_path)
        kernels = {p: GeneralKernel(env, config["seed"], True, config["candidate_cap"],
            config["matching_work"], True, config["total_candidate_cap"], p)
            for p in ("native", "macros", "no-a", "full")}
        for p, k in kernels.items():
            if k.jpv:
                k.jpv.profile = True; k.jpv.stats.update(profile_seconds={}, profile_calls={})
            original = k.preview
            def timed(raw, view, action, original=original, p=p):
                start = time.perf_counter()
                try:
                    return original(raw, view, action)
                finally:
                    previews[p][action.family+" seconds"] += time.perf_counter()-start
                    previews[p][action.family+" calls"] += 1
            k.preview = timed
        root = next(Diagram.from_record(s["record"]) for s in data["states"] if s["key"] == data["root_graph"])
        raw_limit = len(root.nodes)+config["max_growth"]
        edge_limit = len(root.edges)+config["max_edge_growth"]
        view_limit = Views(len(root.ports)//2, reference).get(root).graph.num_vertices()+config["max_growth"]
        rng = random.Random(911)
        for state in data["states"]:
            raw = Diagram.from_record(state["record"]); assert raw.digest() == state["key"]
            if warmup:
                view = Views(len(raw.ports)//2, reference).get(raw)
                for p, k in kernels.items():
                    if k.jpv:
                        k.jpv.profile = False
                    start = time.perf_counter(); k.prepare(raw, view, view_limit)
                    warmups.append(dict(state=state["key"], profile=p, seconds=time.perf_counter()-start))
                    if k.jpv:
                        k.jpv.profile = True
            for repeat in range(repeats):
                start = time.perf_counter(); view = Views(len(raw.ports)//2, reference).get(raw)
                view_seconds = time.perf_counter()-start
                if view.graph is None:
                    raise ValueError("Recorded benchmark state has no normalized view")
                order = list(kernels); rng.shuffle(order)
                for p in order:
                    k = kernels[p]; start = time.perf_counter()
                    actions = k.prepare(raw, view, view_limit)
                    prepared = time.perf_counter()-start
                    start = time.perf_counter()
                    feasible, _ = Feasibility(raw_limit, edge_limit, capacity=0).prepare(raw, view, actions, k)
                    rows.append(dict(state=state["key"], vertices=len(raw.nodes), edges=len(raw.edges),
                        repeat=repeat, profile=p, view_seconds=view_seconds, prepare_seconds=prepared,
                        feasibility_seconds=time.perf_counter()-start, candidates=len(actions), eligible=len(feasible),
                        families=dict(Counter(a.family for a in actions))))
        for p, k in kernels.items():
            subset = [r for r in rows if r["profile"] == p]
            totals[p] = dict(prepare_seconds=sum(r["prepare_seconds"] for r in subset),
                feasibility_seconds=sum(r["feasibility_seconds"] for r in subset),
                candidates=sum(r["candidates"] for r in subset),
                eligible=sum(r["eligible"] for r in subset),
                jpv=None if k.jpv is None else k.jpv.stats,
                preview_families=dict(previews[p]))
    result = dict(circuit=data["circuit"], input_sha256=data["input_sha256"], pack_sha256=sha(pack_path),
        source_commit=data["source_commit"], profiler_sha256=sha(Path(__file__)),
        runtime_sources={str(p): sha(code_root/p) for p in
            (Path("zxopt/general_kernel.py"), Path("zxopt/jpv_actions.py"), Path("zxopt/general_feasibility.py"))},
        repeats=repeats, states=len(data["states"]),
        warmups=warmups,
        timing_scope="identical saved states, full bounded menus, cold feasibility cache; no extraction, PPO or end-to-end speed claim",
        order="fixed seeded randomized interleave", rows=rows, totals=totals)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["pack", "profile"])
    p.add_argument("input", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--states", type=int, default=6); p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", action="store_true", help="One untimed preparation per profile/state, retained separately")
    a = p.parse_args()
    if a.states < 2 or a.repeats < 1:
        raise ValueError("At least two states and one repeat required")
    if a.mode == "pack":
        pack(a.input, a.output, a.states)
    else:
        profile(a.input, a.output, a.repeats, a.warmup)


if __name__ == "__main__":
    main()
