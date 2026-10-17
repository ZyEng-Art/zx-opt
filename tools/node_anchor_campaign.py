"""Own-container controller for an immutable anchor-first comparison checkout."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("--container", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--circuits", nargs="+", default=["barenco_tof_3", "mod5_4", "tof_5"])
    p.add_argument("--seeds", nargs="+", default=["11", "29", "47"])
    a = p.parse_args(); base = a.base.resolve()
    ownership = json.loads((base/"ownership.json").read_text())
    if ownership["container_name"] != a.container:
        raise RuntimeError("Container name differs from the task's recorded ownership")
    container_id = ownership["container_id"]

    def inspect():
        data = json.loads(subprocess.check_output(["docker", "inspect", container_id], text=True))[0]
        labels = data["Config"].get("Labels", {})
        if (data["Name"].lstrip("/") != a.container or labels.get("task") != "rlzx-anchor-20261018"
                or labels.get("owner") != "dengzengyong"
                or data["State"]["StartedAt"] != ownership["started_at"]):
            raise RuntimeError("Owned container incarnation changed")
        return data

    def status(phase, **extra):
        record = dict(phase=phase, commit=a.commit, container_id=container_id,
                      time=datetime.datetime.now().astimezone().isoformat(), **extra)
        temp = base/"controller.json.tmp"; temp.write_text(json.dumps(record, indent=2)+"\n")
        temp.replace(base/"controller.json")

    def run(arguments, log):
        inspect()
        cmd = ["docker", "exec", "-w", "/workspace/task/source",
               "-e", "PYTHONPATH=/workspace/pin:/workspace/offline",
               "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
               container_id, "/root/miniconda/envs/python310_torch29_cuda/bin/python", *arguments]
        with (base/log).open("x") as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        inspect()
        actual = subprocess.check_output(["git", "-C", str(base/"source"), "rev-parse", "HEAD"], text=True).strip()
        if actual != a.commit:
            raise RuntimeError("Checkout revision differs from the requested immutable source")
        status("unit_tests")
        run(["tools/check_pyzx_pin.py"], "pyzx-pin.log")
        run(["-m", "unittest", "discover", "-s", "tests", "-v"], "unit-tests.log")
        status("comparison", circuits=a.circuits, seeds=a.seeds, seconds=a.seconds, workers=a.workers)
        run(["tools/anchor_campaign.py", "runs/comparison", "--seconds", str(a.seconds),
             "--workers", str(a.workers), "--audit-workers", "2", "--circuits", *a.circuits,
             "--seeds", *a.seeds, "--modes", "global32", "anchor8", "anchor32",
             "--policies", "frozen", "online"], "comparison.log")
        finished = json.loads((base/"source/runs/comparison/finished.json").read_text())
        if finished["status"] != "passed":
            raise RuntimeError("Comparison did not pass its event and circuit audits")
        status("archiving", trials=finished["trials"])
        archive = base/("anchor-"+a.commit[:12]+"-records.tar.gz")
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            tar.add(base/"source/runs", arcname="runs")
            for name in ("unit-tests.log", "pyzx-pin.log", "comparison.log", "ownership.json"):
                tar.add(base/name, arcname=name)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        metadata = dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest.hexdigest())
        (base/"archive.json").write_text(json.dumps(metadata, indent=2)+"\n")
        status("passed", trials=finished["trials"], archive=metadata)
    except BaseException as exc:
        status("failed", error=repr(exc), artifacts_retained=True)
        raise
    finally:
        # Only this task's exact recorded container incarnation is stopped.
        data = inspect()
        if data["State"]["Running"]:
            subprocess.run(["docker", "stop", "-t", "30", container_id], check=True)


if __name__ == "__main__":
    main()
