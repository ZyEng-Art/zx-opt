"""Compare recorded experiment sources with the exact committed Git objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zxopt.experiment_input import recorded_input


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    summary=json.loads((args.campaign/"summary.json").read_text())
    campaign=json.loads((args.campaign/"campaign.json").read_text())
    assert summary["completed"]==summary["expected"]==len(summary["rows"])
    checked={};records=[]
    for row in summary["rows"]:
        protocol=json.loads((args.campaign/row["folder"]/"protocol.json").read_text())
        assert not protocol["tracked_dirty"] and protocol["sources"]==campaign["sources"]
        commit=protocol["git_commit"]
        if commit not in checked:
            git=["git","-c","safe.directory="+str(root),"--git-dir="+str(root/".git"),"--work-tree="+str(root)]
            def content(name):return subprocess.check_output(git+["show",commit+":"+name],cwd=root)
            sources={name:hashlib.sha256(content(name)).hexdigest()for name in protocol["sources"]}
            manifest_bytes=content("docs/baseline_sources.json")
            baseline=json.loads(manifest_bytes)
            for name,digest in baseline["files"].items():
                assert hashlib.sha256(content("legacy/"+name)).hexdigest()==digest
            checked[commit]=dict(sources=sources,baseline_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                                 frozen_legacy_files=len(baseline["files"]))
        audit=checked[commit]
        assert audit["sources"]==protocol["sources"]
        assert audit["baseline_sha256"]==protocol["baseline_manifest_sha256"]
        path=recorded_input(args.campaign/row["folder"],protocol)
        original=protocol.get("input_repository_path")
        if "input_snapshot" not in protocol:
            original="inputs/barenco_tof_3.qasm"
        if original:
            assert hashlib.sha256(content(original)).hexdigest()==protocol["input_sha256"]
        records.append(dict(arm=row["arm"],seed=row["seed"],commit=commit,status="passed",input_sha256=protocol["input_sha256"],input_git_verified=bool(original)))
    result=dict(status="passed",trials=len(records),commits=checked,records=records,
                checks="Every core source against Git objects; identical campaign snapshot; frozen legacy manifest and files; input digest; clean experiment checkout")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"passed","trials":len(records),"commits":list(checked)}))


if __name__=="__main__":main()
