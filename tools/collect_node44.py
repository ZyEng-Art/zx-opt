"""Archive only this task's completed outputs; run on the remote host."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_root",type=Path);p.add_argument("commit");args=p.parse_args()
    short=args.commit[:12];source=args.task_root/("zx-opt-"+short)
    report=json.loads((source/"runs/pilot/report.json").read_text())
    summary=json.loads((source/"runs/pilot/summary.json").read_text())
    assert report["status"]=="passed"and report["runs"]==summary["expected"]==summary["completed"]
    controller=args.task_root/("zx-opt-"+short+"-controller.json")
    state=json.loads(controller.read_text())
    assert state["status"]=="passed"and state["restored_stopped"]and state["commit"]==args.commit
    archive=args.task_root/("zx-opt-"+short+"-records.tar.gz")
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,"w:gz")as tf:
        tf.add(source/"runs",arcname="runs")
        tf.add(controller,arcname="controller.json")
        for old in sorted(args.task_root.glob("zx-opt-*-controller.json")):
            if old!=controller:tf.add(old,arcname="historical-controllers/"+old.name)
        for prefix in ["91bdc8082f85","20ab7adcc2be"]:
            path=args.task_root/("zx-opt-"+prefix)/"runs"
            if path.exists():tf.add(path,arcname="setup-failures/"+prefix)
    h=hashlib.sha256()
    with archive.open("rb")as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    manifest=dict(status="passed",commit=args.commit,archive=archive.name,bytes=archive.stat().st_size,
                  sha256=h.hexdigest(),run_files=sum(p.is_file()for p in(source/"runs").rglob("*")),
                  controller=state)
    receipt=args.task_root/("zx-opt-"+short+"-archive.json")
    receipt.write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest))


if __name__=="__main__":main()
