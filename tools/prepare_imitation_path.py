"""Recover a successful search ancestry with exact recorded local action menus.

This prepares same-circuit demonstrations, not evidence for the original PPO.
It does not inject a circuit, a target action or an archive into later search.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(trial, root_key, target):
    """First-creation predecessor edges connect each state back to the root."""
    parents = {}; visits = Counter(); chosen = None
    with (trial/"events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            anchor_key = json.dumps(event["anchor"], sort_keys=True)
            key = (event["parent_graph_key"], anchor_key)
            before = visits[key]; visits[key] += 1
            if event["status"] not in ("accepted", "accepted_unscored"):
                continue
            child = event["graph_key"]
            if child != root_key and child not in parents:
                parents[child] = (event, before)
            if event["status"] == "accepted" and event["cost"]["gates"] <= target:
                chosen = event
                break
    if chosen is None:
        raise ValueError("No recorded accepted target circuit")
    path = []; seen = set(); key = chosen["graph_key"]
    while key != root_key:
        if key in seen or key not in parents:
            raise ValueError("Incomplete/cyclic ancestry; refusing to truncate a demonstration")
        seen.add(key); event, visit = parents[key]; path.append((event, visit))
        key = event["parent_graph_key"]
    return list(reversed(path)), chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--target-gates", type=int, default=36)
    p.add_argument("--allow-unaudited-source", action="store_true",
                   help="Explicit development dataset; selected states/actions/circuits are reverified here")
    a = p.parse_args(); trial = a.trial.resolve()
    import torch
    from qiskit import qasm2
    from zxopt import anchor_policy
    from zxopt.complete import Diagram
    from zxopt.general_kernel import GeneralKernel, Views
    from zxopt.general_feasibility import Feasibility, signature
    from zxopt.general_search import candidate_digest
    from zxopt.legacy import environment
    from zxopt.model_snapshot import state_dict_digest
    from zxopt.scalable_verify import reference_for, check_qasm
    torch.set_num_threads(1)
    protocol = json.loads((trial/"protocol.json").read_text())
    report = json.loads((trial/"report.json").read_text()); config = protocol["config"]
    assert report["verification"] == "passed" and report["best"]["cost"]["gates"] <= a.target_gates
    assert not protocol["tracked_dirty"] and config["selection"] == "anchor"
    assert not config["resume_state"] and not config["shared_buffer"]
    assert not config.get("neutral_stop_steps", 0), "This initial experiment uses the existing default context"
    for name in ("zxopt/anchor_policy.py", "zxopt/general_policy.py", "zxopt/general_kernel.py",
                 "zxopt/general_feasibility.py", "zxopt/jpv_actions.py", "zxopt/rules.py"):
        assert sha(ROOT/name) == protocol["sources"][name], ("Changed demonstration semantics", name)
    campaign = trial.parent.parent
    audit_path = campaign/"audit/verification.json"
    audited = (campaign/"finished.json").exists() and audit_path.exists()
    if audited:
        from tools.summarize_anchor import require_audit_receipt
        completed = json.loads((campaign/"finished.json").read_text())
        verification = json.loads(audit_path.read_text())
        plan = json.loads((campaign/"campaign.json").read_text())["plan"]
        assert completed["status"] == verification["status"] == "passed"
        assert completed["trials"] == verification["trials"] == len(plan)
        spec = next(r for r in plan if r["folder"] == str(trial.relative_to(campaign)))
        require_audit_receipt(trial, spec, verification)
    if not audited and not a.allow_unaudited_source:
        raise ValueError("Whole source audit pending; request an explicit development dataset or wait")
    path, target = recover(trial, protocol["root_graph"], a.target_gates)
    if not path:
        raise ValueError("The target must improve the root")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    text = (trial/"input.qasm").read_text(); assert sha(trial/"input.qasm") == protocol["input_sha256"]
    (out/"input.qasm").write_text(text)
    n = qasm2.loads(text).num_qubits
    reference = reference_for(text, config.get("verification", "matrix")); views = Views(n, reference)
    env = environment(trial/"input.qasm")

    def graph(key):
        raw = Diagram.from_record(json.loads(gzip.decompress((trial/"objects"/(key+".json.gz")).read_bytes())))
        assert raw.digest() == key
        return raw

    root = graph(protocol["root_graph"]); root_view = views.get(root)
    wire_limit = len(root.edges)+config["max_edge_growth"]
    observations = []; actions = []; manifest = []; max_error = max_qasm_error = 0.
    for event, previous_visits in path:
        raw = graph(event["parent_graph_key"]); view = views.get(raw)
        kernel = GeneralKernel(env, config["seed"], config["jpv"], config["candidate_cap"],
                               config["matching_work"], config["general_zx"], config["total_candidate_cap"])
        kernel.candidate_cap = min(config["candidate_cap"], config["anchor_candidate_cap"])
        kernel.total_candidate_cap = min(config["total_candidate_cap"] or config["anchor_candidate_cap"],
                                         config["anchor_candidate_cap"])
        if kernel.jpv is not None:
            kernel.jpv.cap = kernel.candidate_cap
        assert kernel.preview_prefix_sha256 == protocol["preview_native_prefix_sha256"]
        record = event["anchor"]; position = record["position"]
        if isinstance(position, list): position = tuple(position)
        anchor = anchor_policy.Anchor(record["space"], record["kind"], position)
        # Both proposal and menu-cap RNGs are independent per state/anchor key.
        # Replaying earlier visits to this key reconstructs its exact counters.
        for _ in range(previous_visits+1):
            available = kernel.prepare(raw, view, root_view.graph.num_vertices()+config["max_growth"], anchor.query())
        previews = {}
        if config["feasibility_mask"]:
            available, previews = Feasibility(protocol["max_vertices"], wire_limit).prepare(raw, view, available, kernel)
        assert len(available) == event["available"] and candidate_digest(available) == event["candidate_sha256"], (
            "Recorded local menu mismatch; no target injection", event["transition"], previous_visits)
        slot = event["selection"]["slot"]; chosen = available[slot]
        assert chosen.record() == event["action"]
        anchor_list = anchor_policy.catalog(raw, view.graph, config["jpv"],
            protocol["max_vertices"] if config["anchor_resource_mask"] else None,
            wire_limit if config["anchor_resource_mask"] else None)
        anchor_slot = event["selection"]["anchor_slot"]
        assert len(anchor_list) == event["selection"]["anchors"]
        assert anchor_list[anchor_slot].record() == record
        key = signature(raw, chosen)
        child, receipts = previews[key] if key in previews else kernel.step(raw, view, chosen)
        assert child.digest() == event["graph_key"] and receipts == event["auxiliary"]
        assert child.record() == graph(event["graph_key"]).record()
        max_error = max(max_error, view.error, views.get(child).error)
        if event["status"] == "accepted":
            qasm_path = trial/"candidates"/(event["sha256"]+".qasm")
            assert sha(qasm_path) == event["sha256"]
            qc = qasm2.loads(qasm_path.read_text())
            assert dict(gates=len(qc), cx=int(qc.count_ops().get("cx", 0)), depth=int(qc.depth())) == event["cost"]
            max_qasm_error = max(max_qasm_error, check_qasm(reference, qasm_path.read_text()))
        assert max_error < 2e-8 and max_qasm_error < 2e-8
        ao = anchor_policy.anchor_observation(raw, view.graph, anchor_list, event["context"], config["anchor_prior"])
        ao.selected = torch.tensor([anchor_slot])
        observations.append((ao, *anchor_policy.conditional_observation(raw, view.graph, available, event["context"], anchor)))
        actions.append(slot)
        manifest.append(dict(transition=event["transition"], trajectory=event["trajectory"],
            trajectory_step=event["trajectory_step"], parent=event["parent_graph_key"], child=event["graph_key"],
            action=event["action"], anchor=record, visits_before=previous_visits,
            candidate_sha256=event["candidate_sha256"], anchor_slot=anchor_slot, slot=slot, cost=event["cost"]))
        print(json.dumps(dict(transition=event["transition"], prepared=len(manifest), previous_visits=previous_visits)), flush=True)
    initial = torch.load(trial/"initial.pt", map_location="cpu", weights_only=True)["state_dict"]
    assert state_dict_digest(initial) == protocol["initial_tensor_sha256"]
    torch.save(dict(observations=observations, actions=actions), out/"dataset.pt")
    package = dict(state_dict=initial, input_sha256=protocol["input_sha256"], selection="anchor",
                   anchor_prior=config["anchor_prior"], anchor_resource_mask=config["anchor_resource_mask"],
                   neutral_stop_steps=0, steps=0, updates=0,
                   probe_origin=dict(stage="source_initial", source_commit=protocol["git_commit"],
                                     source_initial_sha256=sha(trial/"initial.pt")))
    torch.save(package, out/"initial-model.pt")
    metadata = dict(status="demonstration_path_reverified", source_whole_audit_passed=audited,
        development_unaudited_source=not audited, source_trial=str(trial), source_commit=protocol["git_commit"],
        source_protocol_sha256=sha(trial/"protocol.json"), source_events_sha256=sha(trial/"events.jsonl"),
        source_report_sha256=sha(trial/"report.json"), source_initial_sha256=sha(trial/"initial.pt"),
        input_sha256=protocol["input_sha256"], root_graph=protocol["root_graph"], root_cost=protocol["root_cost"],
        target_transition=target["transition"], target_graph=target["graph_key"], target_cost=target["cost"],
        examples=len(path), path=manifest, max_graph_operator_error=max_error, max_qasm_operator_error=max_qasm_error,
        dataset_sha256=sha(out/"dataset.pt"), initial_model_sha256=sha(out/"initial-model.pt"), config=config,
        preparer_sha256=sha(Path(__file__)),
        scope="Same-circuit successful search ancestry, potentially across archive restarts; recorded sampling contexts retained. Every local menu SHA, witness, saved child, HH receipt and fixed-wire operator/circuit cost reverified. No target was injected into a menu. Demonstration learning is separate from assessing the original PPO; later probes must use only model weights and the original input with fresh pools/RNG.")
    (out/"demonstration.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ("status", "examples", "source_whole_audit_passed", "target_cost", "max_graph_operator_error")}))


if __name__ == "__main__":
    main()
