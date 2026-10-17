"""Same-circuit factorial pilot; every arm/seed is retained."""
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

ARMS = {
    "legacy_uniform": [],
    "legacy_dedup": ["--dedup"],
    "expanded_uniform": ["--expanded"],
    "expanded_dedup": ["--expanded", "--dedup"],
    "expanded_frozen": ["--expanded", "--dedup", "--policy", "frozen"],
    "expanded_online": ["--expanded", "--dedup", "--policy", "online"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--seconds", type=float, default=180.)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    tasks = [(name, seed) for name in ARMS for seed in args.seeds]
    random.Random(20261017).shuffle(tasks)
    snapshot = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (ROOT/"zxopt").glob("*.py")}
    atomic_json(out/"campaign.json", dict(arms=ARMS, seeds=args.seeds, seconds=args.seconds,
                workers=args.workers, order=tasks, sources=snapshot,
                scope="Only barenco_tof_3; all arms/seeds reported; no historical witness seeded",
                caveat="New verification/timing harness, so legacy_uniform is not a reprint of the old v49 timing"))

    def one(task):
        name, seed = task
        folder = out/(name+"_s"+str(seed))
        command = [sys.executable, "-m", "zxopt.search", str(folder), "--seed", str(seed),
                   "--seconds", str(args.seconds)] + ARMS[name]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0")
        with (out/(name+"_s"+str(seed)+".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            return dict(arm=name, seed=seed, status="failed", returncode=result.returncode, folder=folder.name)
        report = json.loads((folder/"report.json").read_text())
        events = [json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
        accepted = [e for e in events if e["status"] == "accepted"]
        first = {str(target): next((e["seconds"] for e in accepted if e["cost"]["gates"] <= target), None)
                 for target in (39,38,37,36,35)}
        row = dict(arm=name, seed=seed, folder=folder.name, **report,
                   first_passage_seconds=first,
                   duplicate_successors=sum(not e["structurally_new"] for e in accepted),
                   raw_graph_keys=len({e["graph_key"] for e in accepted}))
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, task) for task in tasks]):
            row = future.result(); rows.append(row)
            atomic_json(out/"summary.json", dict(rows=sorted(rows,key=lambda r:(r["arm"],r["seed"])),
                        completed=len(rows), expected=len(tasks)))
            print(json.dumps(dict(arm=row["arm"],seed=row["seed"],status=row["status"],
                                  best=row.get("best"),transitions=row.get("transitions"))), flush=True)
    for path, digest in snapshot.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise AssertionError("Source changed during campaign: "+path)
    failed = [r for r in rows if r["status"] == "failed"]
    atomic_json(out/"report.json", dict(status="failed" if failed else "passed", runs=len(rows),
                source_unchanged=True, all_arms_retained=True, failure_count=len(failed)))
    if failed: raise RuntimeError("Campaign contains failed runs; inspect logs")


if __name__ == "__main__": main()
