"""Host-side controller: run a committed bundle in an exact owned CPU container.

No GPU use. Restore the container to stopped only if this controller started the
same container incarnation. Save failure state and preserve every output folder.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--container-id", required=True)
    p.add_argument("--python", default="/workspace/env310/bin/python")
    p.add_argument("--seconds", type=int, default=180)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--suite", choices=["expansion", "representation", "frontier", "jpv", "targets"], default="expansion")
    args = p.parse_args()
    if not args.task_root.is_absolute() or not all(c in "0123456789abcdef" for c in args.commit):
        raise ValueError("Expected absolute task directory and hexadecimal commit")
    short = args.commit[:12]
    trials = 36 if args.suite == "targets" else (4 if args.suite == "frontier" else 6)*3
    bundle = args.task_root/("zx-opt-"+short+".bundle")
    dest = args.task_root/("zx-opt-"+short)
    mount = "/workspace/"+dest.name
    status_path = args.task_root/("zx-opt-"+short+"-controller.json")
    def run(*command): return subprocess.run(list(command), check=True)
    def inspect(): return json.loads(subprocess.check_output(["docker","inspect",args.container]))[0]
    def record(**row):
        row["time"] = datetime.datetime.now().astimezone().isoformat()
        tmp = status_path.with_suffix(".tmp"); tmp.write_text(json.dumps(row,indent=2)); tmp.replace(status_path)
    info = inspect()
    if info["Id"] != args.container_id or info["State"]["Running"]:
        raise RuntimeError("Require the exact stopped, user-owned container")
    if dest.exists(): raise FileExistsError("Preserve previous output: "+str(dest))
    run("git","clone",str(bundle),str(dest))
    commit = subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
    if commit != args.commit: raise RuntimeError("Bundle HEAD mismatch")
    run("docker","start",args.container)
    started = inspect()["State"]["StartedAt"]
    base = ["docker","exec","-e","OPENBLAS_NUM_THREADS=1","-e","OMP_NUM_THREADS=1",
            "-e","MKL_NUM_THREADS=1","-e","PYTHONHASHSEED=0","-e","CUDA_VISIBLE_DEVICES=",
            "-e","PYTHONPATH=/workspace/pyzx-298693bed2608e84fcb44a672282f873321c70b7",
            "-e","GIT_CONFIG_COUNT=1","-e","GIT_CONFIG_KEY_0=safe.directory",
            "-e","GIT_CONFIG_VALUE_0="+mount,"-w",mount,args.container,args.python]
    record(status="running",commit=commit,container=args.container_id,started_at=started,gpu=False,
           seconds_per_run=args.seconds,trials=trials,workers=args.workers,suite=args.suite,
           bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
    status = "failed"
    try:
        run(*base,"-m","unittest","discover","-s","tests","-v")
        if args.suite == "targets":
            inputs=["inputs/tof_3.qasm", "inputs/mod5_4.qasm", "inputs/tof_4.qasm"]
            per_target=max(1,args.workers//len(inputs))
            (dest/"runs").mkdir(exist_ok=True)
            (dest/"runs/environment.json").write_bytes(subprocess.check_output(base+["tools/environment.py"]))
            run(*base,"-m","zxopt.target_campaign","runs/pilot","--inputs",*inputs,
                "--seconds",str(args.seconds),"--workers-per-target",str(per_target))
            run(*base,"tools/audit_target_study.py","runs/pilot","runs/audit",
                "--workers-per-target",str(per_target))
        elif args.suite == "jpv":
            run(*base,"-m","zxopt.general_search","runs/preflight","--jpv",
                "--policy","online","--seconds","180","--max-steps","256")
            preflight = json.loads((dest/"runs/preflight/report.json").read_text())
            if preflight["transitions"] < 256 or preflight["updates"] < 2:
                raise RuntimeError("JPV preflight must finish 256 steps and two PPO updates")
            run(*base,"-m","zxopt.general_campaign","runs/pilot","--seconds",str(args.seconds),
                "--workers",str(args.workers))
            run(*base,"tools/audit_general.py","runs/pilot","runs/audit","--workers",str(min(args.workers,18)))
            run(*base,"tools/audit_sources.py","runs/pilot","runs/source-audit.json")
            run(*base,"tools/probe_general_policy.py","runs/pilot","runs/policy-probe.json")
        else:
            preflight = ["--expanded"] if args.suite == "expansion" else ["--dedup-scope", "serialized"]
            run(*base,"-m","zxopt.search","runs/preflight","--dedup",*preflight,
                "--policy","online","--seconds","30","--max-steps","128")
            run(*base,"-m","zxopt.campaign","runs/pilot","--seconds",str(args.seconds),
                "--workers",str(args.workers),"--suite",args.suite)
        status = "passed"
    finally:
        latest = inspect(); restored = False
        if latest["Id"] == args.container_id and latest["State"]["Running"] and latest["State"]["StartedAt"] == started:
            run("docker","stop","--timeout","20",args.container); restored=True
        record(status=status,commit=commit,container=args.container_id,gpu=False,
               restored_stopped=restored,output=str(dest/"runs/pilot"),suite=args.suite,
               seconds_per_run=args.seconds,workers=args.workers,trials=trials,
               bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())


if __name__ == "__main__": main()
