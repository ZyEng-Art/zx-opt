"""Own-container controller for an immutable anchor-first comparison checkout."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def archive_records(base, commit, study_status):
    """Seal terminal records, including failed trials, without claiming success."""
    if study_status not in ("passed", "failed"):
        raise ValueError("Only terminal study records can be archived")
    archive = base/("anchor-"+commit[:12]+"-"+study_status+"-records.tar.gz")
    with archive.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as tar:
            if (base/"source/runs").exists():
                tar.add(base/"source/runs", arcname="runs")
            for name in ("unit-tests.log", "pyzx-pin.log", "comparison.log", "ownership.json",
                         "launch.json", "controller.json"):
                if (base/name).exists():
                    tar.add(base/name, arcname=name)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    metadata = dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest.hexdigest(),
                    study_status=study_status, source_commit=commit)
    temp = base/"archive.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2)+"\n")
    temp.replace(base/"archive.json")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("--container", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--circuits", nargs="+", default=["barenco_tof_3", "mod5_4", "tof_5"])
    p.add_argument("--seeds", nargs="+", default=["11", "29", "47"])
    p.add_argument("--modes", nargs="+", default=["global32", "anchor8", "anchor32"])
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

    failure = None
    try:
        inspect()
        actual = subprocess.check_output(["git", "-C", str(base/"source"), "rev-parse", "HEAD"], text=True).strip()
        if actual != a.commit:
            raise RuntimeError("Checkout revision differs from the requested immutable source")
        status("unit_tests")
        run(["tools/check_pyzx_pin.py"], "pyzx-pin.log")
        run(["-m", "unittest", "discover", "-s", "tests", "-v"], "unit-tests.log")
        status("comparison", circuits=a.circuits, seeds=a.seeds, modes=a.modes, seconds=a.seconds, workers=a.workers)
        run(["tools/anchor_campaign.py", "runs/comparison", "--seconds", str(a.seconds),
             "--workers", str(a.workers), "--audit-workers", "2", "--circuits", *a.circuits,
             "--seeds", *a.seeds, "--modes", *a.modes,
             "--policies", "frozen", "online"], "comparison.log")
        finished = json.loads((base/"source/runs/comparison/finished.json").read_text())
        if finished["status"] != "passed":
            raise RuntimeError("Comparison did not pass its event and circuit audits")
        status("archiving", trials=finished["trials"])
        metadata = archive_records(base, a.commit, "passed")
        status("passed", trials=finished["trials"], archive=metadata)
    except BaseException as exc:
        failure = repr(exc)
        status("failed", error=failure, artifacts_retained=True)
        raise
    finally:
        # Only this task's exact recorded container incarnation is stopped.
        data = inspect()
        if data["State"]["Running"]:
            subprocess.run(["docker", "stop", "-t", "30", container_id], check=True)
        if failure is not None:
            # Stop owned workers before reading their files. Collection success
            # must remain distinguishable from experiment success.
            try:
                metadata = archive_records(base, a.commit, "failed")
            except Exception as exc:
                status("failed", error=failure, artifacts_retained=True, archive_error=repr(exc))
            else:
                status("failed", error=failure, artifacts_retained=True, archive=metadata)


if __name__ == "__main__":
    main()
