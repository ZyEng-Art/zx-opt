"""Host controller for an immutable general-angle study in the owned CPU container."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--container-id", required=True)
    p.add_argument("--sympy-wheel", type=Path, required=True)
    a = p.parse_args()
    if not a.task_root.is_absolute() or len(a.commit) != 40 or any(c not in "0123456789abcdef" for c in a.commit):
        raise ValueError("Absolute path and full commit required")
    short = a.commit[:12]; bundle = a.task_root/("zx-opt-"+short+".bundle")
    dest = a.task_root/("zx-opt-"+short); mount = "/workspace/"+dest.name
    receipt = a.task_root/("zx-opt-"+short+"-angle-controller.json")
    def inspect():
        return json.loads(subprocess.check_output(["docker", "inspect", a.container]))[0]
    def run(command):
        subprocess.run(command, check=True)
    def record(**values):
        values.update(time=datetime.datetime.now().astimezone().isoformat(), commit=a.commit,
                      container=a.container_id, gpu=False, source=str(dest))
        tmp = receipt.with_suffix(".tmp"); tmp.write_text(json.dumps(values, indent=2)+"\n"); tmp.replace(receipt)
    info = inspect()
    if info["Id"] != a.container_id or info["State"]["Running"] or dest.exists():
        raise RuntimeError("Require exact stopped owned container and new source directory")
    run(["git", "clone", str(bundle), str(dest)])
    assert subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip() == a.commit
    run(["docker", "start", a.container]); started = inspect()["State"]["StartedAt"]
    base = ["docker", "exec", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1", "-e", "PYTHONHASHSEED=0", "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "PYTHONPATH="+mount+"/.deps:/workspace/pyzx-298693bed2608e84fcb44a672282f873321c70b7",
            "-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory",
            "-e", "GIT_CONFIG_VALUE_0="+mount, "-w", mount, a.container, "/workspace/env310/bin/python"]
    status = "failed"; error = None
    try:
        record(status="checking_runtime", started_at=started)
        (dest/"runs").mkdir()
        if a.sympy_wheel.name != "sympy-1.13.1-py3-none-any.whl" or not a.sympy_wheel.is_file():
            raise ValueError("Require the pinned SymPy wheel")
        run(base+["-m", "pip", "install", "--no-deps", "--target", mount+"/.deps",
                  "/workspace/"+a.sympy_wheel.name])
        (dest/"runs/environment.json").write_bytes(subprocess.check_output(base+["tools/environment.py"]))
        run(base+["-c", "import sympy; assert sympy.__version__ == '1.13.1', sympy.__version__"])
        run(base+["-m", "unittest", "discover", "-s", "tests", "-v"])
        record(status="running", started_at=started, trials=48, seconds=600, workers=48)
        run(base+["tools/general_angle_study.py", "runs/angle-study", "--inputs",
                  "inputs/barenco_tof_3.qasm", "inputs/mod5_4.qasm", "inputs/qft_4_numeric.qasm",
                  "inputs/ising_4_numeric.qasm"])
        record(status="auditing", started_at=started)
        def audit(name):
            target = "runs/angle-study/"+name; audit_out = "runs/angle-audit/"+name
            commands = [["tools/audit_general.py", target, audit_out, "--workers", "6"],
                        ["tools/audit_sources.py", target, audit_out+"/source-audit.json"],
                        ["tools/probe_general_policy.py", target, audit_out+"/policy-probe.json"],
                        ["tools/summarize_general.py", target, "runs/angle-analysis/"+name]]
            log = dest/"runs"/(name+"-audit.log")
            with log.open("x") as stream:
                for command in commands:
                    subprocess.run(base+command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        names = [m+"-"+c for m in ("conditional", "general")
                 for c in ("barenco_tof_3", "mod5_4", "qft_4_numeric", "ising_4_numeric")]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(audit, names))
        status = "passed"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        latest = inspect(); restored = False
        if latest["Id"] == a.container_id and latest["State"]["Running"] and latest["State"]["StartedAt"] == started:
            run(["docker", "stop", "--timeout", "20", a.container]); restored = True
        record(status=status, error=error, restored_stopped=restored, started_at=started,
               trials=48, seconds=600, workers=48)
        archive = a.task_root/("zx-opt-"+short+"-angle-records.tar.gz")
        with tarfile.open(archive, "x:gz") as tf:
            tf.add(dest/"runs", arcname="runs"); tf.add(receipt, arcname="controller.json")
        sha = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                sha.update(block)
        manifest = dict(status=status, archive=archive.name, bytes=archive.stat().st_size,
                        sha256=sha.hexdigest(), commit=a.commit)
        (a.task_root/("zx-opt-"+short+"-angle-archive.json")).write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
