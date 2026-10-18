"""Audit every accepted general-ZX rewrite, saved circuit, reward and model pair."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immediate_terminal(event, config, root_gates, restart_gates=None):
    """The recorded step's terminal flag, before any later dead-end discovery."""
    done = event["trajectory_step"] >= config["horizon"]
    if event["status"] == "anchor_noop":
        done = done or config.get("anchor_noop_stop", False)
    elif event["status"] == "accepted_unscored":
        done = done or event["unscored_depth"] >= config["max_unscored"]
    elif event["status"] == "accepted":
        done = done or event["cost"]["gates"] > 1.5*root_gates
        if config.get("gate_improvement_stop") and restart_gates is not None:
            done = done or event["cost"]["gates"] < restart_gates
    limit = config.get("neutral_stop_steps", 0)
    return done or bool(limit and event["neutral_streak"] >= limit)


def replay_native_rooted(kernel, raw, view, action, anchor=None):
    """Recheck the native rule witness, then replay the frozen rewrite prefix.

    All host neighborhoods remain visible to the original predicates. The
    post-rewrite global menus and legacy observations are not audit outputs.
    Operator, saved-child, HH-receipt and QASM checks remain in audit_one.
    """
    if action.mode != "legacy" or action.family != action.payload.kind:
        raise ValueError("Expected a consistent native action record")
    from runtime import graph_key
    if view.graph is None or graph_key(view.graph) != action.parent_view_key:
        raise ValueError("Native action belongs to another scoring view")
    if anchor is not None:
        if anchor["space"] != "view" or anchor["kind"] != "node":
            raise ValueError("Native action has no scoring-view node anchor")
        root = anchor["position"]
    else:
        if not action.payload.vertices:
            raise ValueError("Native action is missing its root witness")
        root = action.payload.vertices[0]
    before = graph_key(view.graph)
    available = kernel.native.prepare_at(view.graph, root)
    desired = action.payload.vertices
    matches = [(identifier, candidate) for _, identifier, candidate in available
               if candidate.kind == action.family and
               (set(candidate.vertices) == set(desired) if action.family == "GF"
                else candidate.vertices == desired and identifier == action.identifier)]
    if len(matches) != 1:
        raise ValueError("Native rule witness is absent or ambiguous at its recorded root")
    identifier, candidate = matches[0]
    if action.family == "GF":
        if not isinstance(action.identifier, int) or action.identifier >= -1:
            raise ValueError("Invalid gadget-fusion action encoding")
        env = kernel.native.env
        index = -identifier-2
        original = env.gadget_fusion_ids[index]
        env.gadget_info_dict[desired] = env.gadget_info_dict[original]
        env.gadget_fusion_ids[index] = desired
    kernel.remember_matches(view)
    child, receipts = kernel.preview(raw, view, replace(action, identifier=identifier))
    if graph_key(view.graph) != before:
        raise AssertionError("Rooted replay mutated the parent scoring view")
    return child, receipts


def audit_one(spec):
    campaign, row, allow_dirty, native_replay, crosscheck_native_every = spec
    folder = Path(campaign)/row["folder"]
    import numpy as np
    import torch
    from qiskit import qasm2
    from zxopt.complete import Diagram
    from zxopt.legacy import environment
    from zxopt.general_kernel import Views, GeneralKernel, HybridAction, rule_allowed
    from zxopt.general_archive import SearchState, GeneralArchive, following_state, score
    from zxopt.jpv_actions import replay_h_pairs
    from zxopt.verify import qiskit_matrix_zx_order, matrix_error
    from zxopt.experiment_input import recorded_input
    from zxopt.scalable_verify import reference_for, check_qasm
    torch.set_num_threads(1)
    protocol = json.loads((folder/"protocol.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    assert allow_dirty or not protocol["tracked_dirty"]
    input_path = recorded_input(folder, protocol)
    original = qasm2.loads(input_path.read_text()); n = original.num_qubits
    assert n == protocol.get("num_qubits", 5)
    reference = reference_for(input_path.read_text(), protocol["config"].get("verification", "matrix"))
    views = Views(n, reference); kernel = GeneralKernel(environment(input_path), jpv=False)

    @lru_cache(maxsize=256)
    def graph(key):
        p = folder/"objects"/(key+".json.gz")
        d = Diagram.from_record(json.loads(gzip.decompress(p.read_bytes())))
        assert d.digest() == key
        return d

    @lru_cache(maxsize=4096)
    def circuit(key):
        text = (folder/"candidates"/(key+".qasm")).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == key
        c = qasm2.loads(text)
        assert c.num_qubits == n and set(c.count_ops()) <= {"h", "rz", "x", "cx"}
        error = check_qasm(reference, text)
        assert error < 2e-8
        return dict(gates=len(c), cx=int(c.count_ops().get("cx", 0)), depth=int(c.depth())), error

    root = SearchState(graph(protocol["root_graph"]), protocol["root_cost"], protocol["root_cost"])
    config = protocol["config"]
    root_signature = hashlib.sha256((folder/"baseline.qasm").read_bytes()).hexdigest()
    archive = GeneralArchive(root, config["archive"], config["latent_archive"],
                             config.get("archive_qasm_cap", 0), root_signature)
    best = dict(root.cost); state = start = None; trajectory = 0
    statuses = Counter(); families = Counter(); repairs = Counter(); score_by_view = {}
    reward_sum = 0.; recoveries = 0; qasm_error = 0.; max_graph_error = 0.; auxiliary = 0
    events = [json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
    history_entries = [json.loads(line) for line in (folder/"best-history.jsonl").read_text().splitlines()]
    pending_history = {entry["transition"]: entry for entry in history_entries}
    assert len(pending_history) == len(history_entries)
    restart_path = folder/"restarts.jsonl"
    restart_rows = ([json.loads(l) for l in restart_path.read_text().splitlines()] if restart_path.exists() else [])
    restart_cursor = 0
    prior_event = None
    native_calls = native_crosschecks = 0
    neutral_streak = neutral_terminations = success_terminations = 0
    def restart(record):
        nonlocal best
        key, source = record["graph_key"], record["source"]
        if source == "root":
            value = archive.root; assert key == value.key
        elif source.startswith("shared:"):
            import sqlite3
            import pyzx as zx
            from zxopt.complete.adapter import from_pyzx
            from zxopt.general_score import VerifiedCompletion
            shared_key = source.split(":", 1)[1]
            database = Path(config["shared_buffer"])
            with sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True) as db:
                qasm = db.execute('SELECT qasm FROM objects WHERE key=?', (shared_key,)).fetchone()[0]
            assert hashlib.sha256(qasm.encode()).hexdigest() == shared_key
            assert matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(qasm))) < 1e-9
            d = from_pyzx(zx.Circuit.from_qasm(qasm).to_graph())
            assert d.digest() == key and d.record() == graph(key).record()
            result = VerifiedCompletion(reference).evaluate(views.get(d), config["extraction_timeout"]).result
            assert result is not None
            value = SearchState(d, result["cost"], result["cost"])
            archive.add(value, value.last_counts, result["sha256"])
            if score(value.cost) < score(best):
                best = dict(value.cost)
        elif source == "latent":
            value = archive.latent[key]; archive.latent_restarts += 1
        else:
            value = archive.scored[key]
            if source == "best":
                assert score(value.cost) == min(score(s.cost) for s in archive.scored.values())
        if "buffer" in record:
            assert archive.summary() == record["buffer"]
        return value
    for i, event in enumerate(events, 1):
        assert event["transition"] == i
        assert rule_allowed(config.get("rule_profile", "full"),
                            event["action"]["mode"], event["action"]["family"])
        statuses[event["status"]] += 1
        if prior_event is not None and prior_event.get("terminal"):
            assert event["trajectory"] > prior_event["trajectory"]
        if event["trajectory"] != trajectory:
            assert event["trajectory_step"] == 1
            trajectory = event["trajectory"]
            neutral_streak = 0
            if restart_rows:
                while restart_cursor < len(restart_rows) and restart_rows[restart_cursor]["trajectory"] <= trajectory:
                    start = restart(restart_rows[restart_cursor]); restart_cursor += 1
                assert start.key == event["restart_graph_key"]
            else:
                start = restart(dict(graph_key=event["restart_graph_key"], source=event["restart_source"]))
            state = start
        assert state.key == event["parent_graph_key"]
        assert event["last_scored_cost"] == state.last_counts
        assert event["parent_cost"] == state.cost
        assert event["unscored_depth_before"] == state.unscored_depth
        if config.get("selection") == "anchor":
            from zxopt.anchor_policy import catalog
            current_view = views.get(state.diagram)
            anchors = catalog(state.diagram, current_view.graph, config["jpv"],
                              protocol["max_vertices"] if config.get("anchor_resource_mask") else None,
                              len(root.diagram.edges)+config["max_edge_growth"] if config.get("anchor_resource_mask") else None)
            assert event["anchor"] == anchors[event["selection"]["anchor_slot"]].record()
            assert event["selection"]["anchors"] == len(anchors)
            selection = event["selection"]
            if selection["log_probability"] is not None:
                assert abs(selection["log_probability"]-selection["anchor_log_probability"]
                           -selection["conditional_log_probability"]) < 2e-5
        if event["status"] == "excluded_after_deadline":
            assert event["seconds"] > event.get("deadline_seconds", config["seconds"])
            continue
        if config.get("neutral_stop_steps", 0):
            assert abs(event["context"][-1]-neutral_streak/config["neutral_stop_steps"]) < 1e-12
            neutral_streak = (neutral_streak+1 if event["status"] in ("accepted", "anchor_noop")
                              and event["cost"] is not None and event["reward"] == 0 else 0)
            neutral_stop = neutral_streak >= config["neutral_stop_steps"]
            assert event["neutral_streak"] == neutral_streak and event["neutral_stop"] == neutral_stop
            neutral_terminations += int(neutral_stop)
        restart_gates = None if start.cost is None else start.cost["gates"]
        if config.get("gate_improvement_stop"):
            success = bool(event["status"] == "accepted" and restart_gates is not None
                           and event["cost"]["gates"] < restart_gates)
            assert event["gate_improvement_stop"] == success
            success_terminations += int(success)
        assert event["terminal"] == immediate_terminal(event, config, root.cost["gates"], restart_gates)
        prior_event = event
        assert 0 <= event["seconds"] <= event.get("deadline_seconds", config["seconds"])
        if event["status"] == "rejected":
            assert event["reward"] == 0
            continue
        if event["status"] == "anchor_noop":
            action = HybridAction.from_record(event["action"])
            assert action.mode == "noop" and action.family == "anchor_noop"
            child, receipts = kernel.step(state.diagram, views.get(state.diagram), action)
            assert child.record() == state.diagram.record() and not receipts
            assert event["graph_key"] == state.key and event["cost"] == state.cost
            assert event["reward"] == 0 and not event["admitted"]
            assert event["unscored_depth"] == state.unscored_depth
            assert archive.summary() == event["buffer"] and event["best"] == best
            continue
        assert event["status"] in ("accepted", "accepted_unscored")
        action = HybridAction.from_record(event["action"])
        parent_view = views.get(state.diagram)
        # Replaying a macro also reproduces its normalized representation key.
        if action.mode == "legacy" and native_replay == "rooted":
            native_calls += 1
            child, receipts = replay_native_rooted(kernel, state.diagram, parent_view, action, event.get("anchor"))
            if crosscheck_native_every and (native_calls-1) % crosscheck_native_every == 0:
                reference_child, reference_receipts = kernel.step(state.diagram, parent_view, action)
                assert child.record() == reference_child.record() and receipts == reference_receipts
                native_crosschecks += 1
        else:
            child, receipts = kernel.step(state.diagram, parent_view, action)
        assert child.digest() == event["graph_key"]
        assert receipts == event["auxiliary"]
        if action.mode == "jpv":
            intermediate = action.payload.apply(state.diagram)
            assert replay_h_pairs(intermediate, event["auxiliary"]).digest() == child.digest()
        stored = graph(event["graph_key"])
        assert stored.record() == child.record()
        assert len(child.nodes) <= protocol["max_vertices"]
        assert len(child.edges) <= len(root.diagram.edges)+config["max_edge_growth"]
        checked = views.get(child)
        max_graph_error = max(max_graph_error, checked.error)
        families[action.family] += 1; auxiliary += len(receipts)
        result = None
        if event["status"] == "accepted":
            counts, error = circuit(event["sha256"])
            qasm_error = max(qasm_error, error)
            assert counts == event["cost"]
            result = dict(cost=counts)
            if checked.key is not None:
                prior = score_by_view.setdefault(checked.key, counts)
                assert prior == counts
            if state.unscored_depth:
                recoveries += 1
            repairs[event.get("completion_branch")] += 1
            improves = score(counts) < score(best)
            history = pending_history.pop(event["transition"], None)
            assert bool(history) == improves
            if history:
                assert history["cost"] == counts and history["sha256"] == event["sha256"]
                assert history["graph"] == event["graph_key"] and history["trajectory"] == event["trajectory"]
                assert history["seconds"] == event["seconds"]
                assert history["strict_gate_improvement"] == (counts["gates"] < best["gates"])
            if improves:
                best = counts
        else:
            assert event["cost"] is None and event["score_failure"]
        state, reward = following_state(state, child, result)
        assert abs(reward-event["reward"]) < 1e-12
        assert state.unscored_depth == event["unscored_depth"] <= config["max_unscored"]
        reward_sum += reward
        admitted = (archive.add(state, start.last_counts, event.get("sha256"))
                    if state.cost is not None or state.unscored_depth < config["max_unscored"] else False)
        assert admitted == event["admitted"]
        assert archive.summary() == event["buffer"]
        assert event["best"] == best
    for p in (folder/"candidates").glob("*.qasm"):
        _, error = circuit(p.stem); qasm_error = max(qasm_error, error)
    assert not ({p.stem for p in (folder/"candidates").glob("*.qasm")}
                & {p.stem for p in (folder/"invalid-extractions").glob("*.qasm")})
    invalid_saved = 0
    for p in (folder/"invalid-extractions").glob("*.qasm"):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == p.stem
        wrong = qasm2.loads(p.read_text())
        assert matrix_error(reference, qiskit_matrix_zx_order(wrong)) > 2e-8
        invalid_saved += 1
    assert invalid_saved == report["completion"]["unique_invalid_qasm"]
    assert len(events) == report["transitions"]
    assert statuses["accepted"]+statuses["accepted_unscored"] == report["accepted"]
    assert statuses["accepted_unscored"] == report["unscored"]
    assert recoveries == report["recovered"]
    assert best == report["best"]["cost"]
    assert not pending_history
    while restart_cursor < len(restart_rows):
        restart(restart_rows[restart_cursor]); restart_cursor += 1
    assert archive.summary() == report["buffer"]
    if config.get("anchor_noop_stop"):
        assert report["anchor_stats"].get("noop_terminations", 0) == statuses["anchor_noop"]
    if config.get("neutral_stop_steps", 0):
        assert report["anchor_stats"].get("neutral_terminations", 0) == neutral_terminations
    if config.get("gate_improvement_stop"):
        assert report["anchor_stats"].get("gate_improvement_terminations", 0) == success_terminations
    best_circuit = qasm2.loads((folder/"best.qasm").read_text())
    assert check_qasm(reference, (folder/"best.qasm").read_text()) < 2e-8
    if (folder/"initial.pt").exists():
        from zxopt.model_snapshot import state_dict_digest
        initial = torch.load(folder/"initial.pt", map_location="cpu", weights_only=True)["state_dict"]
        final_package = torch.load(folder/"final.pt", map_location="cpu", weights_only=True)
        assert final_package.get("rule_profile", "full") == config.get("rule_profile", "full")
        final = final_package["state_dict"]
        assert state_dict_digest(initial) == protocol["initial_tensor_sha256"]
        if protocol.get("model_resume"):
            origin = protocol["model_resume"]
            assert origin["snapshot"] == "resume-model.pt"
            snapshot = folder/origin["snapshot"]
            assert sha(snapshot) == origin["sha256"]
            saved = torch.load(snapshot, map_location="cpu", weights_only=True)
            assert saved["input_sha256"] == protocol["input_sha256"] == origin["input_sha256"]
            assert saved.get("neutral_stop_steps", 0) == config.get("neutral_stop_steps", 0)
            assert state_dict_digest(saved["state_dict"]) == origin["tensor_sha256"]
            assert origin["tensor_sha256"] == protocol["initial_tensor_sha256"]
            assert set(initial) == set(saved["state_dict"])
            assert all(torch.equal(value, saved["state_dict"][k]) for k, value in initial.items())
        for v in final.values():
            assert bool(torch.isfinite(v).all())
        delta = float(sum((final[k]-v).double().square().sum() for k, v in initial.items()).sqrt())
        actor_delta = float(sum((final[k]-v).double().square().sum() for k, v in initial.items()
                                if k.startswith(("actor_encoder.", "actor_head."))).sqrt())
        if row["arm"].endswith("frozen"):
            assert delta == 0.
        else:
            assert report["updates"] > 0 and delta > 0. and actor_delta > 0.
        model = dict(parameter_l2_delta=delta, actor_l2_delta=actor_delta, finite=True)
    else:
        model = None
    snapshots = 0
    snapshot_receipts = {}
    for entry in history_entries:
        required = bool(config.get("save_improvement_models") and model is not None
                        and entry["strict_gate_improvement"])
        receipt = entry.get("policy_snapshot")
        assert bool(receipt) == required
        if not required:
            continue
        expected = "improvement-models/g"+str(entry["cost"]["gates"])+"-t"+str(entry["transition"])+".pt"
        assert receipt["path"] == expected and receipt["steps"] == entry["transition"]
        path = folder/expected
        assert sha(path) == receipt["sha256"]
        snapshot_receipts[expected] = receipt["sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["steps"] == entry["transition"] and saved["updates"] == receipt["updates"]
        assert 0 <= saved["updates"] <= report["updates"]
        assert saved["input_sha256"] == protocol["input_sha256"]
        assert saved["source_commit"] == protocol["git_commit"]
        assert saved["selection"] == config["selection"]
        assert saved["anchor_noop_stop"] == config.get("anchor_noop_stop", False)
        assert saved.get("neutral_stop_steps", 0) == config.get("neutral_stop_steps", 0)
        for key, default in (("discount_factor", .99), ("gae_lambda", .95), ("gate_improvement_stop", False)):
            assert saved.get(key, default) == config.get(key, default)
        assert saved["anchor_prior"] == config["anchor_prior"]
        assert saved["anchor_resource_mask"] == config["anchor_resource_mask"]
        assert set(saved["state_dict"]) == set(initial)
        assert all(saved["state_dict"][k].shape == value.shape and bool(torch.isfinite(saved["state_dict"][k]).all())
                   for k, value in initial.items())
        if config["policy"] == "frozen":
            assert all(torch.equal(value, saved["state_dict"][k]) for k, value in initial.items())
        snapshots += 1
    assert snapshots == len(list((folder/"improvement-models").glob("*.pt")))
    return dict(arm=row["arm"], seed=row["seed"], status="passed", transitions=len(events),
                accepted_rewrites=report["accepted"], replayed_auxiliary=auxiliary,
                legal_unscored=report["unscored"], recovered=recoveries,
                saved_qasm=sum(1 for _ in (folder/"candidates").glob("*.qasm")),
                quarantined_invalid_qasm=invalid_saved,
                max_graph_operator_error=max_graph_error, max_qasm_operator_error=qasm_error,
                accepted_families=dict(families), completion_branches=dict(repairs),
                verification_method=protocol.get("verification_method", "full operator"),
                model=model, improvement_model_snapshots=snapshots,
                native_replay=native_replay, rooted_native_replays=native_calls,
                full_native_crosschecks=native_crosschecks,
                protocol_sha256=sha(folder/"protocol.json"), best_history_sha256=sha(folder/"best-history.jsonl"),
                initial_model_sha256=sha(folder/"initial.pt") if (folder/"initial.pt").exists() else None,
                final_model_sha256=sha(folder/"final.pt") if (folder/"final.pt").exists() else None,
                improvement_model_receipts=snapshot_receipts,
                events_sha256=sha(folder/"events.jsonl"), report_sha256=sha(folder/"report.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--allow-dirty", action="store_true", help="Development-only audit; report preserves dirty-checkout status")
    p.add_argument("--native-replay", choices=["reference", "rooted"], default="reference")
    p.add_argument("--crosscheck-native-every", type=int, default=0,
                   help="Rooted mode: also execute the full reference native path periodically;1 checks every native rewrite")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    if a.crosscheck_native_every < 0 or (a.crosscheck_native_every and a.native_replay != "rooted"):
        raise ValueError("Native crosschecks require rooted replay and a nonnegative interval")
    summary = json.loads((a.campaign/"summary.json").read_text())
    assert summary["completed"] == summary["expected"] == len(summary["rows"])
    assert not any(r["status"] == "failed" for r in summary["rows"])
    audits = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(audit_one, [(str(a.campaign.resolve()), r, a.allow_dirty,
                                        a.native_replay, a.crosscheck_native_every) for r in summary["rows"]]):
            audits.append(row)
            (a.output/"progress.json").write_text(json.dumps(dict(completed=len(audits), expected=len(summary["rows"])))+"\n")
            print(json.dumps(dict(arm=row["arm"], seed=row["seed"], status="passed",
                                  rewrites=row["accepted_rewrites"])), flush=True)
    pairs = []
    for base in ("macros", "jpv"):
        for seed in sorted({r["seed"] for r in summary["rows"]}):
            frozen, online = (a.campaign/(base+"_"+policy+"_s"+str(seed)) for policy in ("frozen", "online"))
            if not frozen.exists() or not online.exists():
                continue
            pf, po = (json.loads((d/"protocol.json").read_text()) for d in (frozen, online))
            assert pf["initial_tensor_sha256"] == po["initial_tensor_sha256"]
            ef, eo = ([json.loads(l) for l in (d/"events.jsonl").read_text().splitlines()] for d in (frozen, online))
            count = po["config"]["rollout"]
            keys = ("parent_graph_key", "restart_graph_key", "restart_source", "action", "available",
                    "candidate_sha256", "selection", "context", "status", "cost", "reward", "auxiliary", "terminal")
            assert len(ef) >= count and len(eo) >= count
            for f, o in zip(ef[:count], eo[:count]):
                assert {k: f.get(k) for k in keys} == {k: o.get(k) for k in keys}, (base, seed, f["transition"])
            pairs.append(dict(base=base, seed=seed, same_initialization=True, identical_prefix=count, status="passed"))
    result = dict(status="passed", development_allow_dirty=a.allow_dirty, trials=len(audits), records=audits, model_pairs=pairs,
                  native_replay=a.native_replay, crosscheck_native_every=a.crosscheck_native_every,
                  saved_qasm=sum(r["saved_qasm"] for r in audits), accepted_rewrites=sum(r["accepted_rewrites"] for r in audits),
                  caveat="Quarantined invalid extractions are excluded from candidates, rewards and best gate counts")
    (a.output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status="passed", trials=len(audits), saved_qasm=result["saved_qasm"])))


if __name__ == "__main__":
    main()
