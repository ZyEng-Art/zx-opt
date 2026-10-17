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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_one(spec):
    campaign, row, allow_dirty = spec
    folder = Path(campaign)/row["folder"]
    import numpy as np
    import torch
    from qiskit import qasm2
    from zxopt.complete import Diagram
    from zxopt.legacy import environment
    from zxopt.general_kernel import Views, GeneralKernel, HybridAction
    from zxopt.general_archive import SearchState, GeneralArchive, following_state, score
    from zxopt.jpv_actions import replay_h_pairs
    from zxopt.verify import qiskit_matrix_zx_order, matrix_error
    from zxopt.experiment_input import recorded_input
    torch.set_num_threads(1)
    protocol = json.loads((folder/"protocol.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    assert allow_dirty or not protocol["tracked_dirty"]
    input_path = recorded_input(folder, protocol)
    original = qasm2.loads(input_path.read_text()); n = original.num_qubits
    assert n == protocol.get("num_qubits", 5)
    reference = qiskit_matrix_zx_order(original)
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
        error = matrix_error(reference, qiskit_matrix_zx_order(c))
        assert error < 2e-8
        return dict(gates=len(c), cx=int(c.count_ops().get("cx", 0)), depth=int(c.depth())), error

    root = SearchState(graph(protocol["root_graph"]), protocol["root_cost"], protocol["root_cost"])
    config = protocol["config"]
    archive = GeneralArchive(root, config["archive"], config["latent_archive"])
    best = dict(root.cost); state = start = None; trajectory = 0
    statuses = Counter(); families = Counter(); repairs = Counter(); score_by_view = {}
    reward_sum = 0.; recoveries = 0; qasm_error = 0.; max_graph_error = 0.; auxiliary = 0
    events = [json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
    restart_path = folder/"restarts.jsonl"
    restart_rows = ([json.loads(l) for l in restart_path.read_text().splitlines()] if restart_path.exists() else [])
    restart_cursor = 0
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
            archive.add(value, value.last_counts)
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
        statuses[event["status"]] += 1
        if event["trajectory"] != trajectory:
            trajectory = event["trajectory"]
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
        if event["status"] == "excluded_after_deadline":
            assert event["seconds"] > config["seconds"]
            continue
        assert 0 <= event["seconds"] <= config["seconds"]
        if event["status"] == "rejected":
            assert event["reward"] == 0
            continue
        assert event["status"] in ("accepted", "accepted_unscored")
        action = HybridAction.from_record(event["action"])
        parent_view = views.get(state.diagram)
        # Replaying a macro also reproduces its normalized representation key.
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
            if score(counts) < score(best):
                best = counts
        else:
            assert event["cost"] is None and event["score_failure"]
        state, reward = following_state(state, child, result)
        assert abs(reward-event["reward"]) < 1e-12
        assert state.unscored_depth == event["unscored_depth"] <= config["max_unscored"]
        reward_sum += reward
        admitted = (archive.add(state, start.last_counts)
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
    while restart_cursor < len(restart_rows):
        restart(restart_rows[restart_cursor]); restart_cursor += 1
    assert archive.summary() == report["buffer"]
    best_circuit = qasm2.loads((folder/"best.qasm").read_text())
    assert matrix_error(reference, qiskit_matrix_zx_order(best_circuit)) < 2e-8
    if (folder/"initial.pt").exists():
        initial = torch.load(folder/"initial.pt", map_location="cpu", weights_only=True)["state_dict"]
        final = torch.load(folder/"final.pt", map_location="cpu", weights_only=True)["state_dict"]
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
    return dict(arm=row["arm"], seed=row["seed"], status="passed", transitions=len(events),
                accepted_rewrites=report["accepted"], replayed_auxiliary=auxiliary,
                legal_unscored=report["unscored"], recovered=recoveries,
                saved_qasm=sum(1 for _ in (folder/"candidates").glob("*.qasm")),
                quarantined_invalid_qasm=invalid_saved,
                max_graph_operator_error=max_graph_error, max_qasm_operator_error=qasm_error,
                accepted_families=dict(families), completion_branches=dict(repairs),
                model=model, events_sha256=sha(folder/"events.jsonl"), report_sha256=sha(folder/"report.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--allow-dirty", action="store_true", help="Development-only audit; report preserves dirty-checkout status")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((a.campaign/"summary.json").read_text())
    assert summary["completed"] == summary["expected"] == len(summary["rows"])
    assert not any(r["status"] == "failed" for r in summary["rows"])
    audits = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(audit_one, [(str(a.campaign.resolve()), r, a.allow_dirty) for r in summary["rows"]]):
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
                  saved_qasm=sum(r["saved_qasm"] for r in audits), accepted_rewrites=sum(r["accepted_rewrites"] for r in audits),
                  caveat="Quarantined invalid extractions are excluded from candidates, rewards and best gate counts")
    (a.output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status="passed", trials=len(audits), saved_qasm=result["saved_qasm"])))


if __name__ == "__main__":
    main()
