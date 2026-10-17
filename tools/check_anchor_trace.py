"""Compare fixed-step traces/models after a computation-only optimization."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(left, right):
    events = lambda folder: [json.loads(s) for s in (folder/"events.jsonl").read_text().splitlines()]
    old, new = events(left), events(right)
    assert len(old) == len(new) and old
    keys = ("transition", "trajectory", "trajectory_step", "parent_graph_key", "restart_graph_key",
            "restart_source", "action", "available", "candidate_sha256", "selection", "context", "anchor",
            "status", "graph_key", "cost", "sha256", "reward", "terminal", "auxiliary", "buffer", "best",
            "unscored_depth_before", "unscored_depth", "admitted")
    for i, (a, b) in enumerate(zip(old, new), 1):
        for key in keys:
            assert a.get(key) == b.get(key), (i, key, a.get(key), b.get(key))
    x = torch.load(left/"final.pt", map_location="cpu", weights_only=True)
    y = torch.load(right/"final.pt", map_location="cpu", weights_only=True)
    for key in x["state_dict"]:
        assert torch.equal(x["state_dict"][key], y["state_dict"][key]), key
    r1, r2 = [json.loads((p/"report.json").read_text()) for p in (left, right)]
    assert r1["status"] == r2["status"] == "step_limit_reached"
    assert r1["best"]["cost"] == r2["best"]["cost"]
    return dict(status="passed", steps=len(old), model_tensors_bitwise_equal=True,
                accepted=r1["accepted"], updates=r1["updates"], before_seconds=r1["total_loop_seconds"],
                after_seconds=r2["total_loop_seconds"], speedup=r1["total_loop_seconds"]/r2["total_loop_seconds"],
                best=r1["best"]["cost"],
                scope="Fixed steps, identical sampled actions/rewards/buffer/model; timing on this execution only")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path); p.add_argument("after", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); result = check(a.before, a.after)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
