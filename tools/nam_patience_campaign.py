"""Run assigned paper NAM targets independently until total-gate stagnation."""
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
    p.add_argument("--node", choices=[22, 23], type=int, required=True)
    p.add_argument("--patience", type=float, default=10800)
    p.add_argument("--workers", type=int, default=13)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/"docs/nam-patience-plan.json").read_text())
    names = plan["assignments"][str(a.node)]
    manifest = json.loads((ROOT/"docs/quarl-nam-inputs.json").read_text())
    cases = {c["name"]: c for c in manifest["cases"]}
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="", XPU_VISIBLE_DEVICES="")
    flags = ["--general-zx", "--policy", "online", "--seed", "11", "--verification", "adaptive",
             "--total-candidate-cap", "32", "--extraction-timeout", "20", "--rewrite-timeout", "30"]
    atomic_json(out/"campaign.json", dict(plan=plan, node=a.node, cases=names, flags=flags,
                patience=a.patience, overall_time_cap=None, gpu=False, source_commit=subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()))

    def run(command, log):
        with log.open("x") as stream:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def check_input(name):
        path = ROOT/"inputs/quarl-nam-raw"/(name+".qasm")
        if hashlib.sha256(path.read_bytes()).hexdigest() != cases[name]["raw"]["sha256"]:
            raise ValueError("Input hash mismatch: "+name)
        return path

    def preflight(name):
        path = check_input(name); folder = out/(name+"-preflight")
        command = ["-m", "zxopt.general_search", str(folder), "--input", str(path), *flags,
                   "--seconds", "600", "--max-steps", "4", "--rollout", "4", "--checkpoint-seconds", "30"]
        try:
            run(command, out/(name+"-preflight.log"))
            report = json.loads((folder/"report.json").read_text())
            assert report["transitions"] == 4 and report["updates"] == 1
            return dict(name=name, status="passed", transitions=4, updates=1, report=report)
        except Exception as exc:
            return dict(name=name, status="failed", reason=repr(exc))

    checks = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(preflight, n) for n in names]):
            checks.append(future.result())
            atomic_json(out/"preflight.json", dict(expected=len(names), completed=len(checks), records=checks))
    if any(r["status"] != "passed" for r in checks):
        raise RuntimeError("Preflight failures retained; no formal jobs started on this node")

    def one(name):
        folder = out/name; folder.mkdir()
        job = folder/"jpv_online_s11"; path = check_input(name)
        command = ["-m", "zxopt.general_search", str(job), "--input", str(path), *flags,
                   "--seconds", "0", "--no-improvement-seconds", str(a.patience),
                   "--checkpoint-seconds", "300", "--rollout", "32"]
        atomic_json(folder/"campaign.json", dict(circuit=name, seconds=None, seeds=[11],
                    arms={"jpv_online": ["--general-zx", "--policy", "online"]},
                    patience=a.patience, input_sha256=cases[name]["raw"]["sha256"]))
        try:
            run(command, folder/"worker.log")
            report = json.loads((job/"report.json").read_text())
            assert report["status"] == "no_improvement_timeout"
            assert report["finished_unix"]-report["patience"]["last_improvement_unix"] >= a.patience
            row = dict(arm="jpv_online", seed=11, folder=job.name, **report)
            atomic_json(folder/"summary.json", dict(completed=1, expected=1, rows=[row]))
            for tool, tail in (("audit_general.py", [str(folder), str(folder/"audit"), "--workers", "1"]),
                               ("audit_sources.py", [str(folder), str(folder/"audit/source-audit.json")]),
                               ("probe_general_policy.py", [str(folder), str(folder/"audit/policy-probe.json")])):
                run([str(ROOT/"tools"/tool), *tail], folder/(tool+".log"))
            return dict(name=name, status="passed", stop_reason=report["status"],
                        gates=report["best"]["cost"], transitions=report["transitions"],
                        updates=report["updates"], patience=report["patience"], verification_method=report["verification_method"])
        except Exception as exc:
            return dict(name=name, status="failed", reason=repr(exc), artifacts_retained=True)

    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one, n) for n in names]):
            records.append(future.result())
            atomic_json(out/"summary.json", dict(expected=len(names), completed=len(records), records=records))
    passed = len(records) == len(names) and all(r["status"] == "passed" for r in records)
    atomic_json(out/"report.json", dict(status="passed" if passed else "failed", records=records,
                finished_unix=time.time(), all_assigned_cases_retained=True))
    if not passed:
        raise RuntimeError("Failed cases retained; inspect logs")


if __name__ == "__main__":
    main()
