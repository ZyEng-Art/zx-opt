"""Local detached collector: periodic rsync, final archive hash verification."""
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
    p.add_argument("--commit", required=True)
    p.add_argument("--interval", type=int, default=600)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    finished = {}; attempts = []
    def save():
        row = dict(status="all_nodes_finished" if len(finished) == 2 else "running",
                   source_commit=a.commit, finished=finished, updated_unix=time.time(), attempts=attempts[-20:])
        tmp = out/"status.tmp"; tmp.write_text(json.dumps(row, indent=2)+"\n"); tmp.replace(out/"status.json")
    save()
    while len(finished) < 2:
        for node in (22, 23):
            if str(node) in finished:
                continue
            userhost = "dengzengyong@wxtky02-p800-backup-8nic-vd-node"+str(node)+".wxtky02"
            ssh = ["ssh", "-S", "/tmp/quarl-nam-node"+str(node)+"-live", "-o", "ProxyCommand=false",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
            base = "/tmp/dengzengyong_rlzx_nam_node"+str(node)+"_20261017"
            target = out/("node"+str(node)); target.mkdir(exist_ok=True)
            try:
                # In-flight copies are monitoring artifacts; final archives are authoritative.
                run_dir = target/"live-runs"; run_dir.mkdir(exist_ok=True)
                command = ["rsync", "-az", "--partial", "--timeout=120", "--no-owner", "--no-group",
                           "--exclude=*.tmp", "-e", shlex.join(ssh),
                           userhost+":"+base+"/zx-opt-"+a.commit[:12]+"/runs/", str(run_dir)+"/"]
                with (target/"mirror.log").open("a") as log:
                    sync = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                attempts.append(dict(node=node, rsync_returncode=sync.returncode, time=time.time()))
                archive_meta = subprocess.run(ssh+[userhost, "cat", base+"/archive-"+a.commit[:12]+".json"],
                                              capture_output=True, text=True, timeout=30)
                try:
                    meta = json.loads(archive_meta.stdout)
                except ValueError:
                    save(); continue
                part = target/(meta["archive"]+".partial")
                with part.open("wb") as stream:
                    subprocess.run(ssh+[userhost, "cat", base+"/"+meta["archive"]], stdout=stream, check=True, timeout=3600)
                digest = hashlib.sha256()
                with part.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(block)
                if digest.hexdigest() != meta["sha256"] or part.stat().st_size != meta["bytes"]:
                    raise ValueError("Final archive incomplete or checksum mismatch")
                part.replace(target/meta["archive"])
                (target/"archive.json").write_text(json.dumps(meta, indent=2)+"\n")
                finished[str(node)] = meta
            except Exception as exc:
                attempts.append(dict(node=node, error=repr(exc), time=time.time()))
            save()
        if len(finished) < 2:
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
