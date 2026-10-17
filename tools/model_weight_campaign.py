"""Audit fresh-pool searches using hash-bound initial/demonstration model weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.policy_weight_probes import command_for
from zxopt.model_snapshot import read_model_snapshot
from zxopt.search import atomic_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[853, 877, 911])
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--audit-workers", type=int, default=4)
    p.add_argument("--audit-native-replay", choices=["reference", "rooted"], default="rooted")
    p.add_argument("--audit-crosscheck-native-every", type=int, default=101)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--allow-development-models", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(); manifest = json.loads(a.manifest.read_text())
    if manifest["development_source"] and not a.allow_development_models:
        raise ValueError("Explicit development-model scope required")
    if a.seconds <= 0 or min(a.workers, a.audit_workers) < 1:
        raise ValueError("Positive model probe budgets/workers required")
    if a.audit_crosscheck_native_every < 0 or (a.audit_crosscheck_native_every and a.audit_native_replay != "rooted"):
        raise ValueError("Native crosschecks require rooted replay")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    if dirty and not a.allow_dirty: raise ValueError("Commit the model-probe source")
    config = manifest["config"]; circuit = manifest["circuit"]
    input_path = ROOT/"inputs/quarl-nam-raw"/(circuit+".qasm")
    assert sha(input_path) == manifest["input_sha256"]
    models = {}; baseline = manifest["baseline_stage"]
    for row in manifest["models"]:
        stage = row["stage"]; relative = Path(row["model"])
        if (not re.fullmatch(r"[a-z0-9_]{1,32}", stage) or stage in models
                or relative.is_absolute() or len(relative.parts) != 1):
            raise ValueError("Use unique short stages and flat relative model paths")
        path = a.manifest.resolve().parent/relative
        assert sha(path) == row["sha256"]
        read_model_snapshot(path, manifest["input_sha256"], config["selection"], config["anchor_prior"],
                            config["anchor_resource_mask"], config.get("neutral_stop_steps", 0))
        models[stage] = dict(stage=stage, model=str(path), sha256=row["sha256"])
    assert len(models) >= 2 and baseline in models
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    plan = [dict(circuit=circuit, stage=stage, policy="frozen", mode="imitation_"+stage,
                 arm="imitation_"+stage+"_frozen", seed=seed,
                 folder=circuit+"/imitation_"+stage+"_frozen_s"+str(seed)) for seed in a.seeds for stage in models]
    atomic_json(out/"campaign.json", dict(source_commit=revision, tracked_dirty=dirty, plan=plan,
        manifest_sha256=sha(a.manifest), runner_sha256=sha(Path(__file__)), model_manifest=manifest,
        seconds=a.seconds, workers=a.workers,
        development_models=manifest["development_source"], started_unix=time.time(),
        scope="Same-target supervised-model weight probes from original input, fresh archive and RNG; no teacher circuit/state transferred. These do not assess original PPO or cross-circuit generalization."))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="", XPU_VISIBLE_DEVICES="")

    def one(row):
        folder = out/row["folder"]; folder.parent.mkdir(exist_ok=True)
        spec = dict(config=config, circuit=circuit, seed=row["seed"], model=models[row["stage"]]["model"])
        try:
            with (out/(row["folder"].replace("/", "__")+".log")).open("x") as stream:
                subprocess.run(command_for(spec, folder, a.seconds, a.max_steps), cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            report = json.loads((folder/"report.json").read_text())
            protocol = json.loads((folder/"protocol.json").read_text())
            assert report["updates"] == 0 and report["root_cost"] == manifest["root_cost"]
            assert protocol["input_sha256"] == manifest["input_sha256"]
            assert protocol["model_resume"]["sha256"] == models[row["stage"]]["sha256"]
            assert not protocol["config"]["resume_state"] and not protocol["config"]["shared_buffer"]
            return dict(**row, status="passed", report=report)
        except Exception as exc:
            return dict(**row, status="failed", error=repr(exc), artifacts_retained=True)

    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in plan]):
            records.append(future.result()); atomic_json(out/"summary.json", dict(expected=len(plan), completed=len(records), rows=records))
            print(json.dumps({k:records[-1][k] for k in ("folder", "status")}), flush=True)
    if any(row["status"] != "passed" for row in records):
        raise ValueError("Failed model probes retained; no passed-study claim")
    rows = [dict(**r["report"], arm=r["arm"], seed=r["seed"], folder=r["folder"]) for r in records]
    atomic_json(out/"summary.json", dict(expected=len(plan), completed=len(records), rows=rows))
    command = [sys.executable, "tools/audit_general.py", str(out), str(out/"audit"), "--workers", str(a.audit_workers),
               "--native-replay", a.audit_native_replay, "--crosscheck-native-every", str(a.audit_crosscheck_native_every)]
    if a.allow_dirty: command += ["--allow-dirty"]
    with (out/"audit.log").open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    index = {(r["stage"],r["seed"]):r["report"] for r in records}; pairs = []
    for seed in a.seeds:
        before = index[baseline,seed]
        for stage in models:
            if stage == baseline: continue
            after = index[stage,seed]; delta = before["best"]["cost"]["gates"]-after["best"]["cost"]["gates"]
            pairs.append(dict(stage=stage, seed=seed, initial=before["best"]["cost"], trained=after["best"]["cost"],
                              saved_gates=delta, outcome="win" if delta>0 else "loss" if delta<0 else "tie"))
    atomic_json(out/"weight-comparison.json", dict(status="passed", pairs=pairs, development_models=manifest["development_source"],
        scope="Audited frozen initial versus demonstration-trained weights on the training circuit, from the original root and fresh search pools. Shared initial controls and selected demonstrations introduce dependence. No original-PPO or generalization claim."))
    atomic_json(out/"finished.json", dict(status="passed", trials=len(records), finished_unix=time.time()))
    print(json.dumps(dict(status="passed", trials=len(records), pairs=pairs)), flush=True)


if __name__ == "__main__":
    main()
