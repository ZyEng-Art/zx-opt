"""Compare identical cold-cache resource menus against full rewrite previews."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--packs", nargs="+", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if a.repeats < 1:
        raise ValueError("Positive repeat count required")
    import torch
    from zxopt.complete import Diagram
    from zxopt.general_kernel import GeneralKernel, Views
    from zxopt.general_feasibility import Feasibility, signature
    from zxopt.legacy import environment
    from zxopt.scalable_verify import reference_for
    torch.set_num_threads(1)
    records = []; rng = random.Random(20261018)
    for pack in a.packs:
        data = json.loads(pack.read_text()); config = data["config"]
        assert hashlib.sha256(data["input_qasm"].encode()).hexdigest() == data["input_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.qasm"; path.write_text(data["input_qasm"])
            env = environment(path); reference = reference_for(data["input_qasm"], config.get("verification", "matrix"))
            kernel = GeneralKernel(env, config["seed"], True, config["candidate_cap"], config["matching_work"],
                                   True, config["total_candidate_cap"])
            views = Views(len(env.graph.inputs()), reference)
            root = next(Diagram.from_record(s["record"]) for s in data["states"] if s["key"] == data["root_graph"])
            view_limit = views.get(root).graph.num_vertices()+config["max_growth"]
            for state in data["states"]:
                raw = Diagram.from_record(state["record"]); assert raw.digest() == state["key"]
                view = views.get(raw)
                for repeat in range(a.repeats):
                    menu = kernel.prepare(raw, view, view_limit)
                    order = ["reference", "bounds"]; rng.shuffle(order)
                    measured = {}; selected = {}; previewed = {}
                    for mode in order:
                        mask = Feasibility(len(root.nodes)+config["max_growth"], len(root.edges)+config["max_edge_growth"],
                                           capacity=0, mode=mode)
                        start = time.perf_counter()
                        eligible, previews = mask.prepare(raw, view, menu, kernel)
                        measured[mode] = dict(seconds=time.perf_counter()-start, stats=mask.summary())
                        selected[mode] = [v.record() for v in eligible]; previewed[mode] = previews
                    assert selected["reference"] == selected["bounds"], (data["circuit"], state["key"])
                    parent = raw.digest()
                    for action in menu:
                        bound = kernel.resource_upper_bound(raw, view, action, parent)
                        actual = previewed["reference"].get(signature(raw, action, parent))
                        if bound is not None and actual is not None:
                            child, _ = actual
                            assert len(child.nodes) <= bound[0] and len(child.edges) <= bound[1]
                    records.append(dict(circuit=data["circuit"], pack_sha256=sha(pack), state=state["key"],
                        repeat=repeat, candidates=len(menu), eligible=len(selected["reference"]),
                        menu_sha256=hashlib.sha256(json.dumps([v.record() for v in menu], sort_keys=True).encode()).hexdigest(),
                        families=dict(Counter(v.family for v in menu)), measured=measured, identical_eligible_menu=True))
    totals = {}
    for circuit in sorted({r["circuit"] for r in records}):
        rows = [r for r in records if r["circuit"] == circuit]
        totals[circuit] = {}
        for mode in ("reference", "bounds"):
            counts = Counter()
            for row in rows:
                counts.update({k:v for k,v in row["measured"][mode]["stats"].items() if isinstance(v, (float, int))})
            totals[circuit][mode] = dict(seconds=sum(r["measured"][mode]["seconds"] for r in rows),
                                          calls=len(rows), stats=dict(counts))
    result = dict(source_commit=__import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        profiler_sha256=sha(Path(__file__)), runtime_sources={p:sha(ROOT/p) for p in
            ("zxopt/general_kernel.py", "zxopt/general_feasibility.py")},
        scope="Identical full-rule bounded menus, randomized interleaving, cold caches, no extraction/PPO; all eligible menus must match exactly",
        records=records, totals=totals)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(dict(calls=len(records), circuits=len(totals), eligible_menus_identical=True)))


if __name__ == "__main__":
    main()
