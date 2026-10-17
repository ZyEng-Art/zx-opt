"""Collect live anchor comparison records and checksum their final archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--interval", type=int, default=300)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    finished = {}; history = []
    while len(finished) != 2:
        for node in (22, 23):
            if str(node) in finished:
                continue
            target = out/("node"+str(node)); target.mkdir(exist_ok=True)
            live = target/"live-runs"; live.mkdir(exist_ok=True)
            base = "/tmp/dengzengyong_rlzx_anchor_node"+str(node)+"_20261018"
            host = "dengzengyong@wxtky02-p800-backup-8nic-vd-node"+str(node)+".wxtky02"
            ssh = ["ssh", "-S", "/tmp/quarl-nam-node"+str(node)+"-live", "-o", "ProxyCommand=false",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
            try:
                with (target/"mirror.log").open("a") as log:
                    result = subprocess.run(["rsync", "-az", "--partial", "--timeout=120", "--no-owner", "--no-group",
                                             "--exclude=*.tmp", "-e", shlex.join(ssh), host+":"+base+"/source/runs/",
                                             str(live)+"/"], stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                if result.returncode:
                    raise RuntimeError("rsync failed: "+str(result.returncode))
                for name in ("controller.json", "ownership.json", "launch.json", "unit-tests.log", "pyzx-pin.log"):
                    result = subprocess.run(ssh+[host, "cat", base+"/"+name], capture_output=True, timeout=30)
                    if result.stdout:
                        (target/name).write_bytes(result.stdout)
                result = subprocess.run(ssh+[host, "cat", base+"/archive.json"], capture_output=True, text=True, timeout=30)
                meta = json.loads(result.stdout) if result.stdout.strip().startswith("{") else None
                if meta:
                    filename = Path(meta["path"]).name
                    if meta["path"] != base+"/"+filename:
                        raise ValueError("Archive must belong to this exact task directory")
                    part = target/(filename+".partial")
                    with part.open("wb") as stream:
                        subprocess.run(ssh+[host, "cat", meta["path"]], stdout=stream, check=True, timeout=3600)
                    digest = hashlib.sha256()
                    with part.open("rb") as stream:
                        for block in iter(lambda: stream.read(8 << 20), b""):
                            digest.update(block)
                    if digest.hexdigest() != meta["sha256"] or part.stat().st_size != meta["bytes"]:
                        raise ValueError("Final archive checksum/size mismatch")
                    part.replace(target/filename)
                    (target/"archive.json").write_text(json.dumps(meta, indent=2)+"\n")
                    finished[str(node)] = meta
                history.append(dict(node=node, status="mirrored", time=time.time()))
            except Exception as exc:
                history.append(dict(node=node, error=repr(exc), time=time.time()))
            state = dict(status="finished" if len(finished) == 2 else "running", finished=finished,
                         updated_unix=time.time(), history=history[-20:],
                         scope="Live rsync copies are monitoring snapshots; checksum-verified final archives are authoritative")
            temp = out/"status.tmp"; temp.write_text(json.dumps(state, indent=2)+"\n"); temp.replace(out/"status.json")
        if len(finished) != 2:
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
