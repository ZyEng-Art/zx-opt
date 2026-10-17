"""Freeze initial/trained weights and compare searches on the same circuit.

No circuit or restart state is transferred. Every arm starts from the original
input and a fresh restart pool. Source models must have completed replay audits.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zxopt.model_snapshot import state_dict_digest
from zxopt.search import atomic_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_pair(source, circuit, mode, training_seed, output, allow_dirty=False):
    finished = json.loads((source/"finished.json").read_text())
    audit_path = source/"audit/verification.json"
    audit = json.loads(audit_path.read_text())
    campaign = json.loads((source/"campaign.json").read_text())
    if finished["status"] != "passed" or audit["status"] != "passed":
        raise ValueError("Training source has not passed its complete replay audit")
    if not (finished["trials"] == audit["trials"] == len(audit["records"]) == len(campaign["plan"])):
        raise ValueError("Training audit does not cover the entire planned study")
    if audit.get("development_allow_dirty") and not allow_dirty:
        raise ValueError("Formal probes require an audited clean training source")
    folder = source/circuit/(mode+"_online_s"+str(training_seed))
    protocol = json.loads((folder/"protocol.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    config = protocol["config"]
    if config["policy"] != "online" or not report["updates"] or report["verification"] != "passed":
        raise ValueError("Probe source is not a completed verified online training run")
    if (protocol["git_commit"] != campaign["source_commit"]
            or (protocol["tracked_dirty"] and not allow_dirty)):
        raise ValueError("Training source provenance mismatch")
    report_sha, events_sha = sha(folder/"report.json"), sha(folder/"events.jsonl")
    receipts = [row for row in audit["records"] if row["report_sha256"] == report_sha
                and row["events_sha256"] == events_sha and row["status"] == "passed"]
    if len(receipts) != 1:
        raise ValueError("The selected model's report/events lack a matching replay receipt")
    receipt = receipts[0]
    for key, name in (("protocol_sha256", "protocol.json"), ("initial_model_sha256", "initial.pt"),
                      ("final_model_sha256", "final.pt")):
        if receipt.get(key) and sha(folder/name) != receipt[key]:
            raise ValueError("Selected model/protocol changed after its replay audit: "+name)
    target = ROOT/"inputs/quarl-nam-raw"/(circuit+".qasm")
    if sha(target) != protocol["input_sha256"]:
        raise ValueError("Probe target differs from the training input")
    for name in ("zxopt/general_policy.py", "zxopt/anchor_policy.py", "zxopt/jpv_actions.py", "zxopt/rules.py"):
        if sha(ROOT/name) != protocol["sources"][name]:
            raise ValueError("Probe observation/action semantics changed: "+name)
    models = {stage: torch.load(folder/filename, map_location="cpu", weights_only=True)
              for stage, filename in (("initial", "initial.pt"), ("trained", "final.pt"))}
    initial, trained = models["initial"], models["trained"]
    if state_dict_digest(initial["state_dict"]) != protocol["initial_tensor_sha256"]:
        raise ValueError("Training initial weights differ from their recorded digest")
    if (trained["steps"] != report["transitions"] or trained["updates"] != report["updates"]
            or trained["input_sha256"] != protocol["input_sha256"]):
        raise ValueError("Final model does not belong to the selected completed report")
    for key, default in (("selection", "global"), ("anchor_prior", "uniform"), ("anchor_resource_mask", False),
                         ("neutral_stop_steps", 0)):
        if trained.get(key, default) != config.get(key, default):
            raise ValueError("Final training model policy support differs: "+key)
    if set(initial["state_dict"]) != set(trained["state_dict"]):
        raise ValueError("Training changed the model's parameter schema")
    if all(torch.equal(value, trained["state_dict"][key]) for key, value in initial["state_dict"].items()):
        raise ValueError("Final weights did not change during training")
    result = []
    for stage, saved in models.items():
        values = saved["state_dict"]
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Non-finite source model")
        name = mode+"_train"+str(training_seed)+"_"+stage
        path = output/(circuit+"__"+name+".pt")
        origin = dict(training_folder=str(folder), training_mode=mode, training_seed=training_seed,
                      stage=stage, source_commit=protocol["git_commit"],
                      source_model_sha256=sha(folder/("initial.pt" if stage == "initial" else "final.pt")),
                      training_protocol_sha256=sha(folder/"protocol.json"),
                      training_report_sha256=report_sha, training_events_sha256=events_sha,
                      training_audit_sha256=sha(audit_path), development_allow_dirty=bool(allow_dirty))
        package = dict(state_dict=values, input_sha256=protocol["input_sha256"],
                       selection=config.get("selection", "global"),
                       anchor_prior=config.get("anchor_prior", "uniform"),
                       anchor_resource_mask=config.get("anchor_resource_mask", False),
                       neutral_stop_steps=config.get("neutral_stop_steps", 0),
                       steps=0 if stage == "initial" else trained["steps"],
                       updates=0 if stage == "initial" else trained["updates"], probe_origin=origin,
                       scope="Frozen same-circuit weight probe; optimizer/archive/active circuit not transferred")
        with path.open("xb") as stream:
            torch.save(package, stream)
        result.append(dict(name=name, circuit=circuit, mode=mode, training_seed=training_seed,
                           stage=stage, model=str(path), model_sha256=sha(path),
                           tensor_sha256=state_dict_digest(values), source=origin, config=config))
    return result


def command_for(spec, folder, seconds, max_steps=None):
    config = spec["config"]
    command = [sys.executable, "-m", "zxopt.general_search", str(folder),
               "--input", str(ROOT/"inputs/quarl-nam-raw"/(spec["circuit"]+".qasm")),
               "--policy", "frozen", "--resume-model", spec["model"],
               "--seed", str(spec["seed"]), "--seconds", str(seconds), "--checkpoint-seconds", "300"]
    for key in ("selection", "anchor_candidate_cap", "anchor_prior", "learning_rate", "max_growth",
                "max_edge_growth", "candidate_cap", "total_candidate_cap", "matching_work", "horizon",
                "max_unscored", "archive", "latent_archive", "rollout", "extraction_timeout", "rewrite_timeout",
                "verification"):
        command += ["--"+key.replace("_", "-"), str(config[key])]
    command += ["--archive-qasm-cap", str(config.get("archive_qasm_cap", 0))]
    command += ["--neutral-stop-steps", str(config.get("neutral_stop_steps", 0))]
    command += ["--discount-factor", str(config.get("discount_factor", .99)),
                "--gae-lambda", str(config.get("gae_lambda", .95))]
    for key in ("general_zx", "jpv", "allow_arbitrary_angles", "anchor_resource_mask", "anchor_noop_stop", "gate_improvement_stop"):
        if config.get(key):
            command += ["--"+key.replace("_", "-")]
    if not config["feasibility_mask"]:
        command.append("--no-feasibility-mask")
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--circuits", nargs="+", default=["barenco_tof_3"])
    p.add_argument("--modes", nargs="+", choices=["global32", "anchor8_resource", "anchor8_resource_lr1e4"],
                   default=["global32", "anchor8_resource", "anchor8_resource_lr1e4"])
    p.add_argument("--training-seeds", nargs="+", type=int, default=[137, 163, 197])
    p.add_argument("--seeds", nargs="+", type=int, default=[401, 431, 461])
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--audit-workers", type=int, default=4)
    p.add_argument("--audit-native-replay", choices=["reference", "rooted"], default="reference")
    p.add_argument("--audit-crosscheck-native-every", type=int, default=0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--allow-dirty", action="store_true", help="Development smoke only, retained in all provenance")
    a = p.parse_args()
    if a.audit_crosscheck_native_every < 0 or (a.audit_crosscheck_native_every and a.audit_native_replay != "rooted"):
        raise ValueError("Native crosschecks require rooted replay and a nonnegative interval")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    if dirty and not a.allow_dirty:
        raise RuntimeError("Commit probe source before running a formal comparison")
    if a.seconds <= 0 or a.workers < 1 or a.audit_workers < 1:
        raise ValueError("Positive probe budgets/workers required")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    models_path = out/"probe-models"; models_path.mkdir()
    models = [model for circuit in a.circuits for mode in a.modes for seed in a.training_seeds
              for model in prepare_pair(a.source.resolve(), circuit, mode, seed, models_path, a.allow_dirty)]
    plan = [dict(**model, seed=seed, arm=model["name"]+"_frozen",
                 folder=model["circuit"]+"/"+model["name"]+"_frozen_s"+str(seed))
            for seed in a.seeds for model in models]
    atomic_json(out/"campaign.json", dict(source_commit=revision, tracked_dirty=dirty,
                training_source=str(a.source.resolve()), models=models, plan=plan,
                seconds=a.seconds, workers=a.workers, started_unix=time.time(),
                audit_workers=a.audit_workers, audit_native_replay=a.audit_native_replay,
                audit_crosscheck_native_every=a.audit_crosscheck_native_every,
                scope="Same-circuit initial versus trained frozen weights. Fresh search RNG/pool from original input; no circuit transfer or cross-circuit generalization claim. Reused initial controls across learning-rate variants are correlated."))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="", XPU_VISIBLE_DEVICES="")

    def one(spec):
        folder = out/spec["folder"]; folder.parent.mkdir(exist_ok=True)
        record = {key: spec[key] for key in ("circuit", "mode", "training_seed", "stage", "seed", "arm", "folder")}
        try:
            with (out/(spec["folder"].replace("/", "__")+".log")).open("x") as stream:
                subprocess.run(command_for(spec, folder, a.seconds, a.max_steps), cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            report = json.loads((folder/"report.json").read_text())
            assert report["updates"] == 0
            protocol = json.loads((folder/"protocol.json").read_text())
            assert protocol["initial_tensor_sha256"] == spec["tensor_sha256"]
            assert protocol["model_resume"]["sha256"] == spec["model_sha256"]
            return dict(**record, status="passed", report=report)
        except Exception as exc:
            return dict(**record, status="failed", error=repr(exc), artifacts_retained=True)

    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one, spec) for spec in plan]):
            records.append(future.result())
            atomic_json(out/"summary.json", dict(expected=len(plan), completed=len(records), rows=records))
            row = records[-1]
            print(json.dumps(dict(folder=row["folder"], status=row["status"],
                                  best=row.get("report", {}).get("best"))), flush=True)
    if any(row["status"] != "passed" for row in records):
        raise RuntimeError("Failed weight probes retained; no successful comparison")
    rows = [dict(**row["report"], arm=row["arm"], seed=row["seed"], folder=row["folder"]) for row in records]
    atomic_json(out/"summary.json", dict(expected=len(plan), completed=len(records), rows=rows))
    command = [sys.executable, str(ROOT/"tools/audit_general.py"), str(out), str(out/"audit"),
               "--workers", str(a.audit_workers), "--native-replay", a.audit_native_replay,
               "--crosscheck-native-every", str(a.audit_crosscheck_native_every)]
    if a.allow_dirty:
        command.append("--allow-dirty")
    with (out/"audit.log").open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    index = {(row["circuit"], row["mode"], row["training_seed"], row["seed"], row["stage"]): row
             for row in records}
    pairs = []
    for circuit in a.circuits:
        for mode in a.modes:
            for training_seed in a.training_seeds:
                for seed in a.seeds:
                    before, after = (index[(circuit, mode, training_seed, seed, stage)]["report"]
                                     for stage in ("initial", "trained"))
                    delta = before["best"]["cost"]["gates"]-after["best"]["cost"]["gates"]
                    pairs.append(dict(circuit=circuit, mode=mode, training_seed=training_seed, search_seed=seed,
                                      initial=before["best"]["cost"], trained=after["best"]["cost"],
                                      saved_gates=delta, outcome="win" if delta > 0 else "loss" if delta < 0 else "tie"))
    outcomes = {mode: dict(Counter(row["outcome"] for row in pairs if row["mode"] == mode)) for mode in a.modes}
    atomic_json(out/"weight-comparison.json", dict(status="passed", pairs=pairs, by_mode=outcomes,
                scope="Gate-count comparison of audited frozen initial/trained weights on the training circuit; training seeds and reused controls introduce dependence. No convergence or generalization claim."))
    atomic_json(out/"finished.json", dict(status="passed", trials=len(records), finished_unix=time.time()))


if __name__ == "__main__":
    main()
