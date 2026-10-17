"""Same-graph exact matrix comparison against the frozen scan-based verifier."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zxopt.complete import Diagram
from zxopt.complete.adapter import to_pyzx
from zxopt.legacy import environment
from zxopt.verify import graph_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--reference-commit", default="b82ec47")
    p.add_argument("--trial", type=Path)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", a.reference_commit+":zxopt/verify.py"], text=True)
    namespace = {}; exec(compile(source, "frozen-scan-verifier.py", "exec"), namespace)
    original = namespace["graph_matrix"]
    graphs = [(n, environment(ROOT/"inputs"/(n+".qasm")).graph) for n in ("barenco_tof_3", "mod5_4", "tof_3", "tof_4")]
    if a.trial:
        for i, path in enumerate(sorted((a.trial/"objects").glob("*.json.gz"))[:16]):
            raw = Diagram.from_record(json.loads(gzip.decompress(path.read_bytes())))
            graphs.append(("saved_"+str(i), to_pyzx(raw, len(raw.ports)//2, evaluate_expressions=True)))
    records = []
    for name, graph in graphs:
        expected = original(graph); actual = graph_matrix(graph)
        if not np.array_equal(actual, expected):
            raise AssertionError("Changed arithmetic result on "+name+": "+str(np.max(np.abs(actual-expected))))
        timings = [[], []]
        for repeat in range(a.repeats):
            for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                fn = (original, graph_matrix)[index]
                started = time.perf_counter(); fn(graph); timings[index].append(time.perf_counter()-started)
        old, new = map(statistics.median, timings)
        records.append(dict(name=name, vertices=graph.num_vertices(), edges=graph.num_edges(),
                            bitwise_equal=True, scan_seconds=old, indexed_seconds=new, speedup=old/new))
    result = dict(status="passed", reference_commit=a.reference_commit,
                  reference_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  graphs=len(records), median_speedup=statistics.median(r["speedup"] for r in records), records=records,
                  scope="Same graph, same elimination order, same bitwise matrix; microbenchmark, not end-to-end speedup")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
