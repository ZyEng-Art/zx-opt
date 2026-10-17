"""Select already reverified demonstration examples for a controlled diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--selection", choices=["gate-reducing", "last"], required=True)
    a = p.parse_args()
    import torch
    meta = json.loads((a.source/"demonstration.json").read_text())
    assert meta["status"] == "demonstration_path_reverified" and meta.get("full_ancestry", True)
    assert sha(a.source/"dataset.pt") == meta["dataset_sha256"]
    data = torch.load(a.source/"dataset.pt", map_location="cpu", weights_only=False)
    assert len(data["actions"]) == len(meta["path"]) == meta["examples"]
    previous = meta["root_cost"]["gates"]; selected = []
    for i, row in enumerate(meta["path"]):
        if row["cost"] is not None:
            if row["cost"]["gates"] < previous: selected.append(i)
            previous = row["cost"]["gates"]
    if a.selection == "last": selected = [len(data["actions"])-1]
    if not selected: raise ValueError("No selected examples")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(observations=[data["observations"][i] for i in selected],
                    actions=[data["actions"][i] for i in selected]), a.output/"dataset.pt")
    for name in ("input.qasm", "initial-model.pt"): shutil.copyfile(a.source/name, a.output/name)
    meta.update(examples=len(selected), path=[meta["path"][i] for i in selected],
                dataset_sha256=sha(a.output/"dataset.pt"), full_ancestry=False, selection_scope=a.selection,
                selected_indices=selected, parent_demonstration_sha256=sha(a.source/"demonstration.json"),
                scope="Subset of independently reverified successful-path observations. Gate-reducing selection compares already recorded verified costs along that ancestry. No new circuit scoring or target injection. This subset is not a complete root-to-target path; fit/utility must be assessed separately.")
    (a.output/"demonstration.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(dict(status="verified_example_subset", examples=len(selected), selection=a.selection)))


if __name__ == "__main__":
    main()
