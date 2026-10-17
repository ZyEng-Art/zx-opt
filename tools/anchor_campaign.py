"""Matched on-target wall-budget trials for global versus anchor-first search."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zxopt.search import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--circuits", nargs="+", default=["barenco_tof_3", "mod5_4", "tof_5"])
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    p.add_argument("--policies", nargs="+", choices=["frozen", "online"], default=["frozen", "online"])
    p.add_argument("--modes", nargs="+", choices=["global32", "anchor8", "anchor32", "anchor8_nodes",
                                                 "anchor8_lr1e3", "anchor8_nodes_lr1e3"],
                   default=["global32", "anchor8"])
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--audit-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in a.circuits:
        path = ROOT/"inputs/quarl-nam-raw"/(name+".qasm")
        if not path.is_file():
            raise ValueError("Missing paper NAM input: "+name)
        cases.append(dict(name=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty and not a.allow_dirty:
        raise RuntimeError("Commit source before a formal comparison")
    plan = [dict(circuit=c, mode=m, policy=pol, seed=s,
                 arm=m+"_"+pol, folder=c+"/"+m+"_"+pol+"_s"+str(s))
            for s in a.seeds for c in a.circuits for m in a.modes for pol in a.policies]
    atomic_json(out/"campaign.json", dict(source_commit=revision, tracked_dirty=bool(dirty),
                cases=cases, plan=plan, seconds=a.seconds, workers=a.workers,
                scope="independent training/search on each target; no generalization claim; count noops separately",
                started_unix=time.time()))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="", XPU_VISIBLE_DEVICES="")

    def one(spec):
        folder = out/spec["folder"]; folder.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "zxopt.general_search", str(folder),
                   "--input", str(ROOT/"inputs/quarl-nam-raw"/(spec["circuit"]+".qasm")),
                   "--general-zx", "--policy", spec["policy"], "--seed", str(spec["seed"]),
                   "--selection", "global" if spec["mode"] == "global32" else "anchor",
                   "--anchor-candidate-cap", "32" if spec["mode"] == "anchor32" else "8",
                   "--anchor-prior", "nodes" if "_nodes" in spec["mode"] else "uniform",
                   "--learning-rate", ".001" if spec["mode"].endswith("lr1e3") else ".0003",
                   "--seconds", str(a.seconds), "--rollout", "32", "--total-candidate-cap", "32",
                   "--verification", "adaptive", "--extraction-timeout", "20", "--rewrite-timeout", "30",
                   "--checkpoint-seconds", "300"]
        if a.max_steps is not None:
            command += ["--max-steps", str(a.max_steps)]
        try:
            with (out/(spec["folder"].replace("/", "__")+".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((folder/"report.json").read_text())
            return dict(**spec, status="passed", report=report)
        except Exception as exc:
            return dict(**spec, status="failed", error=repr(exc), artifacts_retained=True)

    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one, spec) for spec in plan]):
            records.append(future.result())
            atomic_json(out/"summary.json", dict(expected=len(plan), completed=len(records), rows=records))
            r = records[-1]
            print(json.dumps(dict(folder=r["folder"], status=r["status"],
                                  best=r.get("report", {}).get("best"))), flush=True)
    if any(r["status"] != "passed" for r in records):
        raise RuntimeError("Failed trials preserved; do not report a successful comparison")
    # Reuse the event/QASM/model auditor. Campaign summary rows need report fields
    # at their top level for this existing auditing interface.
    rows = [dict(**r["report"], arm=r["arm"], seed=r["seed"], folder=r["folder"]) for r in records]
    atomic_json(out/"summary.json", dict(expected=len(plan), completed=len(records), rows=rows))
    command = [sys.executable, str(ROOT/"tools/audit_general.py"), str(out), str(out/"audit"),
               "--workers", str(a.audit_workers)]
    if a.allow_dirty:
        command.append("--allow-dirty")
    with (out/"audit.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(out/"finished.json", dict(status="passed", trials=len(records), finished_unix=time.time()))


if __name__ == "__main__":
    main()
