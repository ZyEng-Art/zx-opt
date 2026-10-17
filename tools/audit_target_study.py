"""Replay, operator/source/model audit and fixed-state probes for every target."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zxopt.search import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--workers-per-target", type=int, default=12)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    study = json.loads((args.study/"campaign.json").read_text())
    summary = json.loads((args.study/"summary.json").read_text())
    assert summary["completed"] == summary["expected"]

    def one(target):
        name = target["circuit"]; campaign = args.study/name; out = args.output/name
        commands = [["tools/audit_general.py", str(campaign), str(out), "--workers", str(args.workers_per_target)],
                    ["tools/audit_sources.py", str(campaign), str(out/"source-audit.json")],
                    ["tools/probe_general_policy.py", str(campaign), str(out/"policy-probe.json")]]
        with (args.output/(name+".log")).open("w") as log:
            for command in commands:
                subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((out/"verification.json").read_text())
        assert result["status"] == "passed"
        return dict(circuit=name, **result)
    with ThreadPoolExecutor(max_workers=len(study["targets"])) as pool:
        records = list(pool.map(one, study["targets"]))
    result = dict(status="passed", targets=records, trials=sum(r["trials"] for r in records),
                  accepted_rewrites=sum(r["accepted_rewrites"] for r in records),
                  saved_qasm=sum(r["saved_qasm"] for r in records))
    assert result["trials"] == summary["expected"]
    atomic_json(args.output/"verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "trials", "accepted_rewrites", "saved_qasm")}))


if __name__ == "__main__":
    main()
