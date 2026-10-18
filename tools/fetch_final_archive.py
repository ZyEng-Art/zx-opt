"""Fetch a terminal study archive directly, without re-mirroring all live objects."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--node", type=int, choices=[22, 23], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--resume", action="store_true", help="Resume this exact owned study's existing output directory")
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9_]{1,32}", a.tag):
        raise ValueError("Use a short owned-study tag")
    base = "/tmp/dengzengyong_rlzx_"+a.tag+"_node"+str(a.node)+"_20261018"
    host = "dengzengyong@wxtky02-p800-backup-8nic-vd-node"+str(a.node)+".wxtky02"
    socket = "/tmp/quarl-nam-node"+str(a.node)+"-live"
    ssh = ["ssh", "-S", socket, "-o", "ProxyCommand=false", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    code = ("import json,pathlib\np=pathlib.Path("+repr(base)+")\n"
            "data={name:json.loads((p/(name+'.json')).read_text()) for name in ('archive','controller','ownership')}\n"
            "print('FINAL_ARCHIVE_META '+json.dumps(data))\n")
    response = subprocess.run(ssh+[host, "python3 -"], input=code, text=True, capture_output=True, timeout=60)
    lines = [line for line in response.stdout.splitlines() if line.startswith("FINAL_ARCHIVE_META ")]
    if len(lines) != 1:
        raise RuntimeError("No terminal archive metadata; remote experiment is unchanged: "+response.stderr[-1000:])
    records = json.loads(lines[0].split(" ", 1)[1])
    meta, owner, controller = (records[name] for name in ("archive", "ownership", "controller"))
    if controller["phase"] not in ("passed", "failed"):
        raise ValueError("Study is not terminal")
    if (controller["commit"] != owner["source_commit"]
            or controller["container_id"] != owner["container_id"]
            or (meta.get("source_commit") and meta["source_commit"] != owner["source_commit"])
            or (meta.get("study_status") and meta["study_status"] != controller["phase"])):
        raise ValueError("Archive ownership/source/status mismatch")
    filename = Path(meta["path"]).name
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", filename) or meta["path"] != base+"/"+filename:
        raise ValueError("Archive must belong to the exact owned task directory")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=a.resume)
    for name, record in records.items():
        existing = out/(name+".json")
        if existing.exists():
            old = json.loads(existing.read_text())
            keys = {"ownership": ("source_commit", "container_id"),
                    "controller": ("commit", "container_id"),
                    "archive": ("path", "sha256", "bytes")}[name]
            if any(old.get(key) != record.get(key) for key in keys):
                raise ValueError("Existing output belongs to a different study/archive: "+name)
    for name, record in records.items():
        (out/(name+".json")).write_text(json.dumps(record, indent=2)+"\n")
    part = out/(filename+".partial")
    final = out/filename
    if part.is_symlink() or final.is_symlink():
        raise ValueError("Archive destination must be a regular local file")
    if not final.exists():
        # A timeout must preserve the already received bytes. In-place rsync
        # reuses that partial file as its delta basis on the next invocation;
        # only the independent size/SHA check below publishes a final archive.
        subprocess.run(["rsync", "--partial", "--inplace", "--timeout=120", "-e", shlex.join(ssh),
                        host+":"+meta["path"], str(part)], check=True, timeout=3600)
    downloaded = final if final.exists() else part
    digest = hashlib.sha256()
    with downloaded.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if downloaded.stat().st_size != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
        raise ValueError("Final archive transport size/checksum mismatch; partial file retained")
    if downloaded == part:
        part.replace(final)
    receipt = dict(status="final_archive_transport_verified", node=a.node, tag=a.tag,
                   source_commit=owner["source_commit"], recorded_study_status=controller["phase"],
                   archive_sha256=meta["sha256"], bytes=meta["bytes"],
                   scope="Transport receipt only. See the archived finished marker and complete replay audit for study validity; failed archives remain failed.")
    (out/"collection.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
