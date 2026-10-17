"""Owned-container controller; preserves logs, stops its exact incarnation, archives."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--node", type=int, choices=[22, 23], required=True)
    a = p.parse_args(); resource = json.loads((a.base/"resource.json").read_text())
    name, cid = resource["container"], resource["id"]
    def inspect():
        return json.loads(subprocess.check_output(["docker", "inspect", cid]))[0]
    info = inspect()
    assert info["Id"] == cid and info["State"]["Running"]
    assert info["Config"]["Labels"]["task"] == "rlzx-nam-patience-20261017"
    assert not info["HostConfig"].get("DeviceRequests")
    started = info["State"]["StartedAt"]; dest = a.base/("zx-opt-"+a.commit[:12])
    assert not dest.exists()
    subprocess.run(["git", "clone", str(a.base/("zx-opt-"+a.commit[:12]+".bundle")), str(dest)], check=True)
    assert subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip() == a.commit
    mount = "/workspace/task/"+dest.name
    base = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "XPU_VISIBLE_DEVICES=",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
            "-e", "PYTHONHASHSEED=0", "-e", "PYTHONPATH=/workspace/task/pyzx-pin:/workspace/task/offline-deps",
            "-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory", "-e", "GIT_CONFIG_VALUE_0="+mount,
            "-w", mount, name, "/root/miniconda/envs/python310_torch29_cuda/bin/python"]
    def record(status, **extra):
        row = dict(status=status, node=a.node, source_commit=a.commit, container_id=cid,
                   started_at=started, gpu=False, workers=13, patience_seconds=10800,
                   updated_at=datetime.datetime.now().astimezone().isoformat(), **extra)
        path = a.base/("controller-"+a.commit[:12]+".json")
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(row, indent=2)+"\n"); tmp.replace(path)
    status = "failed"; error = None
    try:
        (dest/"runs").mkdir()
        # Old image Git builds may ignore safe.directory passed through env.
        # This configuration belongs only to our isolated container and path.
        subprocess.run(["docker", "exec", name, "git", "config", "--global", "--add", "safe.directory", mount], check=True)
        record("runtime_checks")
        (dest/"runs/pyzx-pin.json").write_bytes(subprocess.check_output(base+["tools/check_pyzx_pin.py"]))
        with (dest/"runs/unit-tests.txt").open("w") as log:
            subprocess.run(base+["-m", "unittest", "discover", "-s", "tests", "-v"], stdout=log, stderr=subprocess.STDOUT, check=True)
        (dest/"runs/environment.json").write_bytes(subprocess.check_output(base+["tools/environment.py"]))
        record("preflight_then_training")
        subprocess.run(base+["tools/nam_patience_campaign.py", "runs/nam-study", "--node", str(a.node)], check=True)
        status = "passed"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        current = inspect(); restored = False
        if current["Id"] == cid and current["State"]["Running"] and current["State"]["StartedAt"] == started:
            subprocess.run(["docker", "stop", "--timeout", "30", cid], check=True); restored = True
        record(status, error=error, restored_stopped=restored)
        archive = a.base/("nam-node"+str(a.node)+"-"+a.commit[:12]+"-records.tar.gz")
        with tarfile.open(archive, "x:gz") as tf:
            tf.add(dest/"runs", arcname="runs")
            tf.add(a.base/("controller-"+a.commit[:12]+".json"), arcname="controller.json")
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        receipt = dict(status=status, archive=archive.name, sha256=digest.hexdigest(), bytes=archive.stat().st_size,
                       source_commit=a.commit, node=a.node)
        (a.base/("archive-"+a.commit[:12]+".json")).write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
