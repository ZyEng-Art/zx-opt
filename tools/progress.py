"""Read atomic progress files without loading Torch or touching running jobs."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
summary=root/"summary.json"
if summary.exists():
    data=json.loads(summary.read_text())
    print(json.dumps(dict(completed=data["completed"],expected=data["expected"])))
for folder in sorted(p for p in root.iterdir()if p.is_dir()):
    path=folder/"report.json"
    if path.exists():
        d=json.loads(path.read_text())
        print(json.dumps(dict(run=folder.name,status=d["status"],best=d["best"]["cost"],steps=d["transitions"],updates=d["updates"])))
    elif (folder/"progress.json").exists():
        d=json.loads((folder/"progress.json").read_text())
        print(json.dumps(dict(run=folder.name,status="running",best=d["best"]["cost"],steps=d["transitions"],seconds=round(d["seconds"],1))))
    elif (folder/"failure.json").exists():
        print(json.dumps(dict(run=folder.name,status="failed",failure=json.loads((folder/"failure.json").read_text()))))
