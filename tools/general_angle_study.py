"""Matched-budget conditional/general-angle trials on independent targets.

One campaign per mode/target; no model or archive transfer. Every seed and arm
is retained. The manifest records the global worker bound before execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zxopt.search import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=600.)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    p.add_argument("--workers-per-campaign", type=int, default=6)
    p.add_argument("--parallel-campaigns", type=int, default=8)
    p.add_argument("--preflight-seconds", type=float, default=300.)
    a = p.parse_args()
    if min(a.seconds, a.workers_per_campaign, a.parallel_campaigns) <= 0:
        raise ValueError("Positive budgets required")
    inputs = [f.resolve() for f in a.inputs]
    if len({f.stem for f in inputs}) != len(inputs):
        raise ValueError("Unique target names required")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    tasks = [(mode, f) for mode in ("conditional", "general") for f in inputs]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="")
    atomic_json(out/"study.json", dict(
        seconds_per_run=a.seconds, seeds=a.seeds, arms=["jpv_frozen", "jpv_online"],
        modes={"conditional": ["--allow-arbitrary-angles"], "general": ["--general-zx"]},
        targets=[dict(name=f.stem, path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in inputs],
        max_cpu_workers=a.parallel_campaigns*a.workers_per_campaign, gpu=False,
        pairing="Within each mode: frozen/online identical initialization and initial rollout. Across modes: different action embeddings, no identical-weight claim.",
        scope="Independent on-target search; no held-out generalization test, no model/state transfer."))

    def call(command, log):
        with log.open("x") as stream:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def preflight(task):
        mode, f = task; folder = out/(mode+"-"+f.stem+"-preflight")
        flags = ["--general-zx"] if mode == "general" else ["--allow-arbitrary-angles"]
        call(["-m", "zxopt.general_search", str(folder), "--input", str(f),
              "--jpv", "--policy", "online", "--seed", str(a.seeds[0]),
              "--seconds", str(a.preflight_seconds), "--max-steps", "128", *flags],
             out/(folder.name+".log"))
        report = json.loads((folder/"report.json").read_text())
        if report["transitions"] < 128 or report["updates"] < 1:
            raise RuntimeError("Preflight incomplete: "+folder.name)
        return dict(mode=mode, circuit=f.stem, transitions=report["transitions"],
                    updates=report["updates"], seconds=report["total_loop_seconds"])

    with ThreadPoolExecutor(max_workers=a.parallel_campaigns) as pool:
        checks = list(pool.map(preflight, tasks))
    atomic_json(out/"preflight.json", dict(status="passed", records=checks))

    def campaign(task):
        mode, f = task; folder = out/(mode+"-"+f.stem)
        flags = ["--general-zx"] if mode == "general" else ["--allow-arbitrary-angles"]
        call(["-m", "zxopt.general_campaign", str(folder), "--input", str(f),
              "--seconds", str(a.seconds), "--workers", str(a.workers_per_campaign),
              "--seeds", *map(str, a.seeds), "--arms", "jpv_frozen", "jpv_online", *flags],
             out/(folder.name+".log"))
        report = json.loads((folder/"report.json").read_text())
        if report["status"] != "passed":
            raise RuntimeError("Campaign failed: "+folder.name)
        summary = json.loads((folder/"summary.json").read_text())
        return [dict(row, mode=mode, target=f.stem, folder=folder.name+"/"+row["folder"])
                for row in summary["rows"]]

    rows = []; expected = len(tasks)*2*len(a.seeds)
    with ThreadPoolExecutor(max_workers=a.parallel_campaigns) as pool:
        for future in as_completed([pool.submit(campaign, task) for task in tasks]):
            rows.extend(future.result())
            atomic_json(out/"summary.json", dict(completed=len(rows), expected=expected, rows=rows))
            print(json.dumps(dict(completed=len(rows), expected=expected)), flush=True)
    atomic_json(out/"report.json", dict(status="passed", runs=len(rows), expected=expected))


if __name__ == "__main__":
    main()
