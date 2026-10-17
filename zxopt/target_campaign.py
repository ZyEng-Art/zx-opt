"""Independent per-circuit training/search; no transferred model or state pool."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .legacy import ROOT
from .search import atomic_json
from .general_campaign import ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--allow-arbitrary-angles", action="store_true")
    p.add_argument("--general-zx", action="store_true")
    p.add_argument("--seconds", type=float, default=1200.)
    p.add_argument("--workers-per-target", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    p.add_argument("--arms", nargs="+", choices=list(ARMS),
                   default=None)
    p.add_argument("--preflight-seconds", type=float, default=300.)
    args = p.parse_args()
    args.arms = args.arms or (["jpv_uniform", "jpv_frozen", "jpv_online"] if args.general_zx
                             else ["macros_uniform", "jpv_uniform", "jpv_frozen", "jpv_online"])
    inputs = [f.resolve() for f in args.inputs]
    if len({f.stem for f in inputs}) != len(inputs) or args.workers_per_target < 1:
        raise ValueError("Require unique target names and positive worker budget")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    targets = [dict(circuit=f.stem, input=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in inputs]
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted((ROOT/"zxopt").rglob("*.py"))}
    atomic_json(out/"campaign.json", dict(suite="other_circuits", targets=targets, sources=sources,
                seconds=args.seconds, seeds=args.seeds, arms=args.arms,
                allow_arbitrary_angles=args.allow_arbitrary_angles,
                general_zx=args.general_zx,
                workers_per_target=args.workers_per_target, max_total_workers=len(inputs)*args.workers_per_target,
                scope="Each target trained from scratch; no model/state transfer; all targets/arms/seeds retained"))

    def preflight(f):
        folder = out/(f.stem+"-preflight")
        command = [sys.executable, "-m", "zxopt.general_search", str(folder), "--input", str(f),
                   "--jpv", "--policy", "online", "--seed", str(args.seeds[0]),
                   "--seconds", str(args.preflight_seconds), "--max-steps", "128"]
        if args.allow_arbitrary_angles:
            command.append("--allow-arbitrary-angles")
        if args.general_zx:
            command.append("--general-zx")
        with (out/(f.stem+"-preflight.log")).open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((folder/"report.json").read_text())
        if report["transitions"] < 128 or report["updates"] < 1:
            raise RuntimeError("Target preflight did not finish one rollout: "+f.stem)
        return dict(circuit=f.stem, transitions=report["transitions"], updates=report["updates"],
                    seconds=report["total_loop_seconds"], status="passed")
    with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        checks = list(pool.map(preflight, inputs))
    atomic_json(out/"preflight.json", dict(status="passed", records=checks))

    def one(f):
        folder = out/f.stem
        command = [sys.executable, "-m", "zxopt.general_campaign", str(folder), "--input", str(f),
                   "--seconds", str(args.seconds), "--workers", str(args.workers_per_target),
                   "--seeds", *map(str, args.seeds), "--arms", *args.arms]
        if args.allow_arbitrary_angles:
            command.append("--allow-arbitrary-angles")
        if args.general_zx:
            command.append("--general-zx")
        with (out/(f.stem+".log")).open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((folder/"summary.json").read_text())
        return [dict(row, target=f.stem, folder=f.stem+"/"+row["folder"]) for row in summary["rows"]]
    expected = len(inputs)*len(args.arms)*len(args.seeds); rows = []
    with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        for result in as_completed([pool.submit(one, f) for f in inputs]):
            rows.extend(result.result())
            atomic_json(out/"summary.json", dict(completed=len(rows), expected=expected,
                        rows=sorted(rows, key=lambda r: (r["target"], r["arm"], r["seed"]))))
    for name, digest in sources.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    atomic_json(out/"report.json", dict(status="passed", runs=len(rows), targets=len(inputs),
                all_arms_retained=True, source_unchanged=True))


if __name__ == "__main__":
    main()
