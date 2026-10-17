"""Paired general-ZX search campaign, with complete-kernel proposals on/off."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from .legacy import ROOT
from .search import atomic_json

ARMS = {name+"_"+policy: (["--jpv"] if name == "jpv" else []) + ["--policy", policy]
        for name in ("macros", "jpv") for policy in ("uniform", "frozen", "online")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--input", type=Path, default=ROOT/"inputs/barenco_tof_3.qasm")
    p.add_argument("--allow-arbitrary-angles", action="store_true")
    p.add_argument("--general-zx", action="store_true")
    p.add_argument("--seconds", type=float, default=1200.)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    p.add_argument("--arms", nargs="+", choices=list(ARMS))
    args = p.parse_args()
    if args.seconds <= 0 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Invalid budget/workers/seeds")
    selected = args.arms or (["jpv_uniform", "jpv_frozen", "jpv_online"] if args.general_zx else list(ARMS))
    if args.general_zx and any(not arm.startswith("jpv_") for arm in selected):
        raise ValueError("--general-zx requires jpv arms; macros-only controls use a separate campaign")
    if len(set(selected)) != len(selected):
        raise ValueError("Repeated arms")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    tasks = [(arm, seed) for arm in selected for seed in args.seeds]
    random.Random(20261017).shuffle(tasks)
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted((ROOT/"zxopt").rglob("*.py"))}
    atomic_json(out/"campaign.json", dict(suite="jpv", arms={k: ARMS[k] for k in selected}, seeds=args.seeds,
                seconds=args.seconds, workers=args.workers, order=tasks, sources=sources,
                circuit=args.input.stem, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                allow_arbitrary_angles=args.allow_arbitrary_angles,
                general_zx=args.general_zx,
                scope="independent target; existing six macros in all arms, extra JPV proposals only in jpv arms; identical general graph policy/cost/pools",
                initialization="Frozen/online same random init, no pretrained weights or historical witness"))
    def one(task):
        arm, seed = task; folder = out/(arm+"_s"+str(seed))
        command = [sys.executable, "-m", "zxopt.general_search", str(folder), "--seed", str(seed),
                   "--seconds", str(args.seconds), "--input", str(args.input.resolve())] + ARMS[arm]
        if args.allow_arbitrary_angles:
            command.append("--allow-arbitrary-angles")
        if args.general_zx:
            command.append("--general-zx")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
        with (out/(folder.name+".log")).open("w") as f:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            return dict(arm=arm, seed=seed, folder=folder.name, status="failed", returncode=result.returncode)
        report = json.loads((folder/"report.json").read_text())
        return dict(arm=arm, seed=seed, folder=folder.name, **report)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, task) for task in tasks]):
            row = future.result(); rows.append(row)
            atomic_json(out/"summary.json", dict(rows=sorted(rows, key=lambda r: (r["arm"], r["seed"])),
                                                completed=len(rows), expected=len(tasks)))
            print(json.dumps(dict(arm=row["arm"], seed=row["seed"], status=row["status"],
                                  best=row.get("best"), transitions=row.get("transitions"))), flush=True)
    for name, digest in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise AssertionError("Source changed during campaign: "+name)
    failed = sum(row["status"] == "failed" for row in rows)
    atomic_json(out/"report.json", dict(status="failed" if failed else "passed", runs=len(rows),
                                       all_arms_retained=True, source_unchanged=True, failure_count=failed))
    if failed:
        raise RuntimeError("Campaign has failed runs; preserve and inspect logs")


if __name__ == "__main__":
    main()
