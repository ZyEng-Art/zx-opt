"""General ZX PPO/search: verified legal diagrams survive extraction failure."""
import argparse
from collections import Counter, OrderedDict
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from qiskit import qasm2

from .legacy import ROOT, LEGACY, environment
from .search import atomic_json, operation_limit
from .complete.adapter import from_pyzx
from .general_kernel import GeneralKernel, Views, OperatorMismatch
from .general_archive import SearchState, GeneralArchive, following_state, score
from .general_policy import GeneralAgent, observation, batch
from .general_score import VerifiedCompletion
from .general_feasibility import Feasibility, FeasibilityMismatch, signature as feasibility_signature
from .jpv_actions import hh_identity
from .verify import qiskit_matrix_zx_order, matrix_error
from monitored_ppo import ppo_update
from .scalable_verify import reference_for, check_qasm, ZXReference
from .patience import GatePatience
from .rule_schedule import RuleSchedule
from .model_snapshot import read_model_snapshot, state_dict_digest
from . import anchor_policy


def tensor_digest(net):
    return state_dict_digest(net.state_dict())


def candidate_digest(available):
    return hashlib.sha256(json.dumps([a.record() for a in available], sort_keys=True).encode()).hexdigest()


def save_graph(folder, diagram):
    key = diagram.digest(); path = folder/(key+".json.gz")
    if not path.exists():
        path.write_bytes(gzip.compress(json.dumps(diagram.record(), separators=(",", ":")).encode(), compresslevel=1, mtime=0))
    return key


def context_for(state, start, best, step, args, root_gates, root_vertices, neutral_streak=0):
    return [(state.last_counts["gates"]-root_gates)/root_gates,
            (best["cost"]["gates"]-root_gates)/root_gates, step/args.horizon,
            state.unscored_depth/args.max_unscored, float(state.cost is not None),
            start.last_counts["gates"]/root_gates,
            (len(state.diagram.nodes)-root_vertices)/max(1, args.max_growth),
            (neutral_streak/args.neutral_stop_steps if args.neutral_stop_steps
             else float(state.unscored_depth > 0))]


def run(args):
    if args.shared_buffer and (args.general_zx or args.allow_arbitrary_angles):
        raise ValueError("Shared restart v1 is Clifford+T only; general-angle ZX remains available without --shared-buffer")
    if args.general_zx:
        args.jpv = True
        args.allow_arbitrary_angles = True
    if ((args.seconds <= 0 and args.no_improvement_seconds <= 0) or args.seconds < 0
            or args.no_improvement_seconds < 0 or args.checkpoint_seconds < 0 or args.total_candidate_cap < 0
            or args.horizon < 1 or args.max_unscored < 1 or args.max_growth < 0
            or args.candidate_cap < 1 or args.matching_work < 1 or args.rewrite_timeout <= 0
            or args.anchor_candidate_cap < 1 or args.archive_qasm_cap < 0 or args.rollout < 1
            or args.neutral_stop_steps < 0
            or args.native_steps < 1 or args.expanded_steps < 1
            or args.learning_rate <= 0 or not np.isfinite(args.learning_rate)):
        raise ValueError("Invalid search bounds")
    if args.selection != "anchor" and args.anchor_prior != "uniform":
        raise ValueError("Anchor prior requires --selection anchor")
    if args.rule_profile != "full" and (args.selection != "global" or not args.general_zx):
        raise ValueError("Rule ablations require global selection and --general-zx")
    if (args.rule_schedule != "static" or args.candidate_allocation != "balanced") and (
            args.selection != "global" or not args.general_zx or args.rule_profile != "full"):
        raise ValueError("Rule scheduling/allocation require global general-ZX with the full library")
    if args.selection != "anchor" and args.anchor_resource_mask:
        raise ValueError("Anchor resource mask requires --selection anchor")
    if args.selection != "anchor" and args.anchor_noop_stop:
        raise ValueError("Anchor noop termination requires --selection anchor")
    if not (0 <= args.discount_factor <= 1 and 0 <= args.gae_lambda <= 1):
        raise ValueError("Discount and GAE lambda must lie in [0,1]")
    if args.selection != "anchor" and (args.discount_factor != .99 or args.gae_lambda != .95):
        raise ValueError("Custom return credit is currently supported for anchor PPO")
    if args.verify_native_flow and not args.native_flow_module:
        raise ValueError("Native flow verification requires a native module")
    native_flow = None
    if args.native_flow_module:
        from .native_flow import enable_native_flow
        native_flow = enable_native_flow(args.native_flow_module)
        native_flow["verify_all_batches"] = args.verify_native_flow
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if args.save_improvement_models:
        (out/"improvement-models").mkdir()
    (out/"objects").mkdir(); (out/"candidates").mkdir()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    input_bytes = args.input.read_bytes()
    (out/"input.qasm").write_bytes(input_bytes)
    env = environment(out/"input.qasm"); n = len(env.graph.inputs())
    input_circuit = qasm2.loads(input_bytes.decode())
    reference = reference_for(input_bytes.decode(), args.verification, out/"verification/receipts.jsonl")
    views = Views(n, reference); scorer = VerifiedCompletion(reference, out/"invalid-extractions")
    if args.verify_native_flow:
        scorer.primary.scorer.verify_flow = scorer.repair.scorer.verify_flow = True
    kernel = GeneralKernel(env, args.seed, args.jpv, args.candidate_cap, args.matching_work,
                           args.general_zx, args.total_candidate_cap, args.rule_profile, args.candidate_allocation)
    rule_schedule = RuleSchedule(args.rule_schedule, args.native_steps, args.expanded_steps)
    anchored = args.selection == "anchor"
    if anchored:
        kernel.candidate_cap = min(args.candidate_cap, args.anchor_candidate_cap)
        kernel.total_candidate_cap = min(args.total_candidate_cap or args.anchor_candidate_cap, args.anchor_candidate_cap)
        if kernel.jpv is not None:
            kernel.jpv.cap = kernel.candidate_cap
    Agent = anchor_policy.AnchorAgent if anchored else GeneralAgent
    pack = anchor_policy.batch if anchored else batch
    update_ppo = ppo_update
    if anchored:
        from .anchor_ppo import ppo_update as update_ppo
    diagram = from_pyzx(env.graph, allow_arbitrary=args.allow_arbitrary_angles)
    root_view = views.get(diagram)
    if root_view.graph is None:
        raise RuntimeError("Initial normalization failed")
    root_evaluation = scorer.evaluate(root_view, args.extraction_timeout)
    root_result = root_evaluation.result
    if root_result is None:
        raise RuntimeError("Initial completion failed: "+str(root_evaluation.reason))
    root_error = check_qasm(reference, root_result["qasm"])
    if root_error > 2e-8:
        raise OperatorMismatch("Initial extracted circuit changed the operator")
    root = SearchState(diagram, root_result["cost"], root_result["cost"])
    pool = GeneralArchive(root, args.archive, args.latent_archive,
                          args.archive_qasm_cap, root_result["sha256"])
    raw_limit = len(diagram.nodes)+args.max_growth
    wire_limit = len(diagram.edges)+args.max_edge_growth
    feasibility = Feasibility(raw_limit, wire_limit, mode=args.feasibility_mode) if args.feasibility_mask else None
    view_limit = root_view.graph.num_vertices()+args.max_growth
    net = Agent(general_zx=args.general_zx) if args.policy != "uniform" else None
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, eps=1e-5) if args.policy == "online" else None
    model_resume = None
    if args.resume_model:
        if net is None:
            raise ValueError("Model resume requires frozen/online policy")
        saved, payload, model_resume = read_model_snapshot(
            args.resume_model, hashlib.sha256(input_bytes).hexdigest(),
            args.selection, args.anchor_prior, args.anchor_resource_mask, args.neutral_stop_steps, args.rule_profile,
            args.rule_schedule, args.native_steps, args.expanded_steps, args.candidate_allocation)
        net.load_state_dict(saved["state_dict"])
        (out/"resume-model.pt").write_bytes(payload)
        if optimizer is not None:
            if saved.get("optimizer_state_dict") is None:
                raise ValueError("Resume checkpoint has no optimizer")
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
    initial_sha = tensor_digest(net) if net is not None else None
    if net is not None:
        torch.save(dict(state_dict=net.state_dict(), seed=args.seed, steps=0), out/"initial.pt")
    source_paths = list((ROOT/"zxopt").rglob("*.py")) + list((ROOT/"zxopt").rglob("*.cpp"))
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    git = ["git", "-c", "safe.directory="+str(ROOT), "--git-dir="+str(ROOT/".git"), "--work-tree="+str(ROOT)]
    revision = subprocess.check_output(git+["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(git+["status", "--porcelain", "--untracked-files=no"], text=True).strip()
    baseline = json.loads((ROOT/"docs/baseline_sources.json").read_text())
    for path, digest in baseline["files"].items():
        if hashlib.sha256((LEGACY/path).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen baseline modified: "+path)
    try:
        input_repository_path = str(args.input.resolve().relative_to(ROOT))
    except ValueError:
        input_repository_path = None
    protocol = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    git_commit=revision, tracked_dirty=bool(dirty), sources=sources,
                    initial_tensor_sha256=initial_sha, started_at=datetime.datetime.now().astimezone().isoformat(),
                    input_sha256=hashlib.sha256(input_bytes).hexdigest(), input_snapshot="input.qasm",
                    circuit=args.input.stem, num_qubits=n, input_gates=len(input_circuit),
                    input_repository_path=input_repository_path,
                    baseline_manifest_sha256=hashlib.sha256((ROOT/"docs/baseline_sources.json").read_bytes()).hexdigest(),
                    root_cost=root_result["cost"], root_graph=diagram.digest(), root_vertices=len(diagram.nodes),
                    root_view_vertices=root_view.graph.num_vertices(), max_vertices=raw_limit,
                    scope="independent on-target training/search; total NAM gates then CX; no historical witness or model transfer",
                    state="general ZX multigraph; raw state never overwritten by its scoring view; macro actions explicitly transition via that view",
                    scoring="verified frozen fixed-wire completion/NAM cleanup; on failure/mismatch repair a separate copy with clifford_simp then full_reduce; first valid branch, exact view-key cache",
                    unscored="verified legal child remains active and enters separate latent FIFO; actual cost=null; reward 0 until next success, then delta from last scored anchor",
                    dedup="exact raw state archive/cost cache and within-proposal signatures; no permanent hard action suppression",
                    proposals="11 JPV families when enabled, finite family-balanced proposals and deterministic matching branch budgets; not complete search",
                    observation="raw colored multigraph + separate normalized graph, boundary correspondences, rule/direction/phase/shape, eight explicit context values",
                    sampling="NumPy search/restart RNG separate from deterministic state/visit proposal RNG and torch PPO shuffle; frozen/online same init",
                    reward="last successfully scored NAM delta / 10; no invented latent cost",
                    verification="each unique raw/view state full fixed-wire operator; every unique QASM independent Qiskit operator; unit-modulus phase only",
                    timing="setup/final model excluded; matching, state export/verification/extraction, graph objects, restart maintenance and PPO inside wall budget",
                    limits="candidate/node/matching/horizon/latent-depth bounds; extraction failures are legal transitions, verification/resource failures rejected",
                    hh="I_X backward / H forward / I_Z forward proof; every contextual cancellation has a replay receipt")
    protocol["invalid_extraction"] = "quarantined and excluded from all gate costs/rewards; legal raw graph can continue. Raw/repair operator mismatches remain fatal. Repairs apply identically in every arm."
    protocol["active_rules"] = dict(profile=args.rule_profile,
        macros=args.rule_profile != "native", jpv=args.jpv and args.rule_profile in ("full", "no-a"),
        nonlinear_a=args.general_zx and args.rule_profile == "full",
        scope="Rules disabled before proposal construction; bounded menus do not imply complete search")
    if native_flow is not None:
        protocol["native_flow"] = native_flow
    if model_resume is not None:
        protocol["model_resume"] = model_resume
        protocol["scope"] = "same-target search from recorded model weights; total NAM gates then CX; this is model continuation/probing, not independent fresh-model training"
    if args.archive_qasm_cap:
        protocol["archive_grouping"] = dict(
            key="SHA256 of the already verified completion QASM; no extra extraction/lookahead",
            representatives_per_qasm=args.archive_qasm_cap,
            preference="fewer raw nodes, then wires, then stable raw digest; root retained",
            scope="Opt-in search heuristic. Distinct raw representatives may enable different future rewrites; no completeness preservation claim. Latent archive unchanged.")
    protocol["verification_method"] = ("fixed-wire ZX identity reduction; scalar modulus tolerance1e-9; failure is inconclusive; zero errors in numeric fields are certificate placeholders, not measured operator errors"
                                       if isinstance(reference, ZXReference) else "independent full fixed-wire Qiskit operator, tolerance2e-8")
    protocol["stop_rule"] = dict(no_improvement_seconds=args.no_improvement_seconds,
                                resets_on="strict reduction in total gates only; CX/depth ties do not reset",
                                overall_cap_seconds=args.seconds or None,
                                checkpoint_seconds=args.checkpoint_seconds)
    protocol["return_credit"] = dict(discount_factor=args.discount_factor, gae_lambda=args.gae_lambda,
        gate_improvement_stop=args.gate_improvement_stop,
        scope="Gate-delta rewards unchanged. Optional success cutoff requires a strictly lower verified gate count than the known restart cost; unknown restart costs never trigger it. Gamma1/lambda1 gives undiscounted episode returns, with normal value bootstrap at collection-window cuts.")
    if args.neutral_stop_steps:
        protocol["neutral_stop"] = dict(steps=args.neutral_stop_steps,
            counter="Consecutive accepted scored rewrites or scored unchanged-state noops with zero total-gate reward; unscored/rejected steps and restarts reset",
            context="The eighth context entry is neutral_streak/limit instead of the redundant latent-depth flag; checkpoints and model capsules retain the limit",
            scope="Optional trajectory allocation heuristic; reward, legal actions and archive rules unchanged. It can discard useful longer plateaus; no completeness or convergence claim.")
    protocol["feasibility_mask"] = "When enabled, rewrite-only previews exclude node/wire-budget violations before categorical sampling; no extraction/cost lookahead. Exact-state/action LRU; all preview work inside budget."
    protocol["resource_bounds"] = dict(mode=args.feasibility_mode,
        scope="Bounds only certify candidates inside both resource budgets; uncertain cases use exact previews. Chosen rewrites retain all proof/operator checks.")
    protocol["rule_schedule"] = dict(mode=args.rule_schedule, native_steps=args.native_steps,
        expanded_steps=args.expanded_steps, candidate_allocation=args.candidate_allocation,
        scope="Past strict total-gate improvements reset a sampled-transition counter. Staged mode alternates native-only and expanded windows; an empty native menu falls back to expanded rules. No completeness claim.")
    protocol["preview_native_prefix_sha256"] = kernel.preview_prefix_sha256
    protocol["angle_support"] = dict(
        allow_arbitrary=args.allow_arbitrary_angles,
        representation="exact rational coefficients of pi from the pinned QASM parser",
        jpv=("12 general-phase JPV schemata including A; exact affine inference and checked A constraints"
             if args.general_zx else "Clifford+T templates only; non-pi/4 internal matches skipped, arbitrary context retained"),
        verification="existing numeric full-operator checks at 2e-8; C8 exact_tensor remains pi/4-only",
        completeness=("JPV Theorem 1.3 rule schemata; executable instances require real expressions and checked constraints; finite proposals are not complete search"
                      if args.general_zx else "Clifford+T rule kernel only; no arbitrary-angle completeness claim"))
    if args.general_zx:
        protocol["proposals"] = "12 general-phase families; finite matching/candidates and per-template phase-operation budget 128"
        protocol["symbolic_state"] = "Exact expression records retained; numeric evaluation only for scoring. Native/macro transitions disabled while raw phases contain expressions."
    protocol["selection"] = (dict(mode="anchor", order="graph encoding -> sample anchor -> rooted matching -> local resource previews -> sample instance",
                                  anchors="raw nodes/edges/closed wires/global scalar; scoring-view nodes/H edges",
                                  policy="learned anchor actor and conditional instance actor; joint log probability in PPO; not Quarl critic-based node sampling",
                                  empty=("explicit terminal zero-reward unchanged-state transition; next step restarts from archive; no hidden resampling"
                                         if args.anchor_noop_stop else "explicit zero-reward unchanged-state transition; no hidden resampling"),
                                  entropy="anchor entropy plus entropy at sampled anchor; sampled conditional regularizer",
                                  prior=args.anchor_prior,
                                  resource_mask=args.anchor_resource_mask,
                                  forward_reuse="Same-step deterministic sampler logits reused for stored probabilities; no cross-state cache",
                                  candidate_cap=kernel.total_candidate_cap)
                             if anchored else dict(mode="global"))
    atomic_json(out/"protocol.json", protocol)
    atomic_json(out/"hh-derivation.json", hh_identity()[2])
    save_graph(out/"objects", diagram)
    (out/"baseline.qasm").write_text(root_result["qasm"]); (out/"best.qasm").write_text(root_result["qasm"])
    best = dict(cost=root_result["cost"], sha256=root_result["sha256"], seconds=0., graph=diagram.digest())
    atomic_json(out/"best.json", best)
    events = (out/"events.jsonl").open("w", buffering=1); updates_file = (out/"updates.jsonl").open("w", buffering=1)
    restart_file = (out/"restarts.jsonl").open("w", buffering=1)
    deadend_file = (out/"deadends.jsonl").open("w", buffering=1)
    best_history = (out/"best-history.jsonl").open("w", buffering=1)
    shared = None
    if args.shared_buffer:
        from .shared_search import ZXSharedBridge
        shared = ZXSharedBridge(args.shared_buffer, input_bytes.decode(), args.seed, out)
        shared.publish(root_result, diagram.digest())
    started = time.perf_counter()
    budget_deadline = started+args.seconds if args.seconds else float("inf")
    patience = GatePatience(args.no_improvement_seconds, best["cost"]["gates"])
    deadline = min(budget_deadline, started+patience.remaining())
    transition = trajectory = accepted = unscored = recovered = updates = 0
    invalid = Counter(); families = Counter(); accepted_families = Counter(); latent_families = Counter()
    timers = Counter(); rollout = []; anchor_stats = Counter()
    state = None; status = "budget_reached"; start = None
    step = 0; restart_source = None; last_checkpoint = started; neutral_streak = 0
    first_best = {str(root_result["cost"]["gates"]-k): None for k in range(1, 6)}

    if args.resume_state:
        if args.resume_model:
            raise ValueError("Choose complete state or model-only resume")
        saved = torch.load(args.resume_state, map_location="cpu", weights_only=False)
        if saved["input_sha256"] != protocol["input_sha256"] or saved["sources"] != sources:
            raise ValueError("Resume input/source differs")
        semantic = ["seed", "policy", "jpv", "general_zx", "allow_arbitrary_angles", "horizon", "max_unscored", "max_growth", "max_edge_growth",
                    "candidate_cap", "matching_work", "archive", "latent_archive", "rollout", "feasibility_mask",
                    "verification", "no_improvement_seconds", "total_candidate_cap", "rewrite_timeout",
                    "selection", "anchor_candidate_cap"]
        semantic += ["anchor_prior", "learning_rate", "anchor_resource_mask", "archive_qasm_cap",
                     "native_flow_module", "verify_native_flow", "anchor_noop_stop", "neutral_stop_steps",
                     "discount_factor", "gae_lambda", "gate_improvement_stop", "rule_profile",
                     "feasibility_mode", "rule_schedule", "native_steps", "expanded_steps", "candidate_allocation"]
        if any(saved["config"][k] != protocol["config"][k] for k in semantic):
            raise ValueError("Resume search options differ")
        pool, state, start = saved["pool"], saved["state"], saved["start"]
        step, restart_source = saved["step"], saved["restart_source"]
        neutral_streak = saved["neutral_streak"]
        rollout = saved["rollout"]; best = saved["best"]; first_best = saved["first_best"]
        patience = GatePatience(args.no_improvement_seconds, best["cost"]["gates"], record=saved.get("patience"))
        deadline = min(budget_deadline, time.perf_counter()+patience.remaining())
        transition, trajectory, accepted, unscored, recovered, updates = saved["counts"]
        if net is not None:
            net.load_state_dict(saved["net"])
        if optimizer is not None:
            optimizer.load_state_dict(saved["optimizer"])
        if kernel.jpv is not None:
            kernel.jpv.rounds, kernel.jpv.stats = saved["jpv"]
        kernel.cap_rounds = saved.get("proposal_cap_rounds", {})
        rule_schedule = RuleSchedule(args.rule_schedule, args.native_steps, args.expanded_steps,
                                     saved.get("rule_schedule"))
        anchor_stats = Counter(saved.get("anchor_stats", {}))
        rng.bit_generator.state = saved["rng"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if shared is not None:
            shared.rng.setstate(saved["exchange_rng"])
        # Objects and QASMs remain source-bound to the previous immutable segment.
        protocol["resumed_from"] = str(args.resume_state.resolve())
        protocol["resume_segment_start_counts"] = saved["counts"]
        protocol["initial_tensor_sha256"] = tensor_digest(net) if net is not None else None
        protocol["time_scope"] = "seconds are segment-local; counters include prior checkpoint; best may originate in prior segment"
        if net is not None:
            torch.save(dict(state_dict=net.state_dict(), seed=args.seed, steps=transition), out/"initial.pt")
        atomic_json(out/"protocol.json", protocol)
        (out/"best.qasm").write_text(saved["best_qasm"])
        atomic_json(out/"best.json", best)

    def checkpoint_now():
        t = time.perf_counter()
        checkpoint = _checkpoint_payload(protocol, pool, state, start, step, restart_source,
                                         rollout, best, (out/"best.qasm").read_text(), first_best,
                                         (transition, trajectory, accepted, unscored, recovered, updates),
                                         net, optimizer, kernel, rng, shared, patience)
        checkpoint["rule_schedule"] = rule_schedule.record()
        checkpoint["anchor_stats"] = dict(anchor_stats)
        checkpoint["neutral_streak"] = neutral_streak
        torch.save(checkpoint, out/"search-state.pt.tmp")
        (out/"search-state.pt.tmp").replace(out/"search-state.pt")
        atomic_json(out/"checkpoint.json", dict(path="search-state.pt", transition=transition,
                    updates=updates, patience=patience.record(), saved_unix=time.time(),
                    input_sha256=protocol["input_sha256"]))
        timers["checkpoint"] += time.perf_counter()-t

    if args.checkpoint_seconds:
        checkpoint_now()
    try:
        while time.perf_counter() < deadline and (args.max_steps is None or transition < args.max_steps):
            if state is None:
                imported = None
                if shared is not None:
                    t = time.perf_counter()
                    imported = shared.choose(views, scorer, args.extraction_timeout)
                    timers["shared_exchange"] += time.perf_counter()-t
                if imported is not None and (len(imported[0].diagram.nodes) > raw_limit or len(imported[0].diagram.edges) > wire_limit):
                    shared.pool.event("conversion_resource_rejected", imported[1]["key"], run=str(out))
                    imported = None
                if imported is None:
                    start, restart_source = pool.choose(rng)
                else:
                    start, record = imported
                    restart_source = "shared:"+record["key"]
                    pool.add(start, start.last_counts, record["completion"]["sha256"])
                    save_graph(out/"objects", start.diagram)
                    if score(start.cost) < score(best["cost"]):
                        result = record["completion"]
                        patience.observe(start.cost["gates"])
                        deadline = min(budget_deadline, time.perf_counter()+patience.remaining())
                        best = dict(cost=start.cost, sha256=result["sha256"],
                                    seconds=time.perf_counter()-started, graph=start.key,
                                    origin="shared_import", shared_key=record["key"])
                        (out/"best.qasm").write_text(result["qasm"])
                        atomic_json(out/"best.json", best)
                state = start
                step = 0; trajectory += 1; neutral_streak = 0
                restart_file.write(json.dumps(dict(trajectory=trajectory, graph_key=start.key,
                                   source=restart_source, buffer=pool.summary()))+"\n")
            t = time.perf_counter()
            view = views.get(state.diagram)
            anchor = anchor_obs = None; anchor_slot = None; anchor_scores = None
            if anchored:
                timers["views"] += time.perf_counter()-t
                t = time.perf_counter()
                context = context_for(state, start, best, step, args, root_result["cost"]["gates"], len(diagram.nodes), neutral_streak)
                anchors = anchor_policy.catalog(state.diagram, view.graph, kernel.jpv is not None,
                                                raw_limit if args.anchor_resource_mask else None,
                                                wire_limit if args.anchor_resource_mask else None,
                                                anchor_stats if args.anchor_resource_mask else None)
                if not anchors:
                    timers["anchor_inference"] += time.perf_counter()-t
                    deadend_file.write(json.dumps(dict(trajectory=trajectory, graph_key=state.key,
                                       seconds=time.perf_counter()-started, reason="empty resource-bounded anchor catalog"))+"\n")
                    if rollout:
                        previous = list(rollout[-1]); previous[-1] = True; rollout[-1] = tuple(previous)
                    state = None
                    if len(pool.scored)+len(pool.latent) == 1:
                        status = "no_resource_anchors"; break
                    continue
                anchor_obs = anchor_policy.anchor_observation(state.diagram, view.graph, anchors, context, args.anchor_prior)
                if net is None:
                    if "prior" in anchor_obs:
                        anchor_slot = anchor_policy.draw(anchor_obs.prior[anchor_obs.y != -1], rng)
                    else:
                        anchor_slot = int(rng.integers(len(anchors)))
                else:
                    with torch.no_grad():
                        from torch_geometric.data import Batch
                        a = Batch.from_data_list([anchor_obs])
                        anchor_scores = net.actor(a).flatten()[a.y != -1]
                        anchor_slot = anchor_policy.draw(anchor_scores, rng)
                anchor = anchors[anchor_slot]
                anchor_obs.selected = torch.tensor([anchor_slot])
                anchor_stats["samples"] += 1
                anchor_stats[anchor.space+":"+anchor.kind] += 1
                timers["anchor_inference"] += time.perf_counter()-t
                t = time.perf_counter()
            try:
                with operation_limit(max(.001, deadline-time.perf_counter())):
                    expanded = rule_schedule.expanded(); native_empty_fallback = False
                    available = kernel.prepare(state.diagram, view, view_limit,
                                               None if anchor is None else anchor.query(), expanded=expanded)
                    if not available and not expanded:
                        native_empty_fallback = True
                        available = kernel.prepare(state.diagram, view, view_limit, expanded=True)
            except TimeoutError:
                timers["matching"] += time.perf_counter()-t
                break
            timers["matching"] += time.perf_counter()-t
            if anchored:
                anchor_stats["matched_candidates"] += len(available)
            previews = {}
            if feasibility is not None:
                t = time.perf_counter()
                try:
                    with operation_limit(max(.001, deadline-time.perf_counter())):
                        available, previews = feasibility.prepare(state.diagram, view, available, kernel)
                except TimeoutError:
                    timers["feasibility"] += time.perf_counter()-t
                    break
                timers["feasibility"] += time.perf_counter()-t
            if not available and not expanded and not native_empty_fallback:
                native_empty_fallback = True
                t = time.perf_counter()
                try:
                    with operation_limit(max(.001, deadline-time.perf_counter())):
                        available = kernel.prepare(state.diagram, view, view_limit, expanded=True)
                except TimeoutError:
                    timers["matching"] += time.perf_counter()-t
                    break
                timers["matching"] += time.perf_counter()-t
                if feasibility is not None:
                    t = time.perf_counter()
                    try:
                        with operation_limit(max(.001, deadline-time.perf_counter())):
                            available, previews = feasibility.prepare(state.diagram, view, available, kernel)
                    except TimeoutError:
                        timers["feasibility"] += time.perf_counter()-t
                        break
                    timers["feasibility"] += time.perf_counter()-t
            if anchored:
                anchor_stats["eligible_candidates"] += len(available)
                if not available:
                    anchor_stats["empty"] += 1
                    available = [anchor_policy.empty_action()]
            if not available:
                deadend_file.write(json.dumps(dict(trajectory=trajectory, graph_key=state.key,
                                                  seconds=time.perf_counter()-started))+"\n")
                if rollout:
                    previous = list(rollout[-1]); previous[-1] = True; rollout[-1] = tuple(previous)
                state = None
                continue
            context = context_for(state, start, best, step, args, root_result["cost"]["gates"], len(diagram.nodes), neutral_streak)
            t = time.perf_counter()
            obs = ((anchor_obs, *anchor_policy.conditional_observation(state.diagram, view.graph, available, context, anchor))
                   if anchored else observation(state.diagram, view.graph, available, context))
            if net is None:
                slot = int(rng.integers(len(available))); old_log = value = None
            else:
                with torch.no_grad():
                    packed = pack([obs]); policy_graph = packed[1] if anchored else packed[0]
                    local_scores = net.actor(policy_graph).flatten()[policy_graph.y != -1]
                    logits = local_scores.double().numpy()
                    probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
                    slot = min(int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="right")), len(available)-1)
                    if anchored:
                        logs, values, anchor_log, local_log = net.selected_action_value(
                            anchor_scores, local_scores, anchor_slot, slot, packed[2])
                    else:
                        _, logs, _, values, anchor_log, local_log = net.get_action_and_value(packed, action=torch.tensor([slot]))
                    old_log = float(logs[0]); value = float(values[0])
            timers["inference"] += time.perf_counter()-t
            chosen = available[slot]; transition += 1; step += 1; families[chosen.family] += 1
            strict_rule_improvement = False
            row = dict(transition=transition, trajectory=trajectory, trajectory_step=step,
                       parent_graph_key=state.key, restart_graph_key=start.key, restart_source=restart_source,
                       action=chosen.record(), available=len(available), candidate_sha256=candidate_digest(available),
                       selection=dict(slot=slot, log_probability=old_log, value=value), context=context,
                       parent_cost=state.cost, last_scored_cost=state.last_counts, unscored_depth_before=state.unscored_depth,
                       deadline_seconds=deadline-started)
            if args.rule_schedule != "static":
                row["rule_schedule"] = dict(**rule_schedule.record(), expanded=expanded,
                                             native_empty_fallback=native_empty_fallback)
            if anchored:
                row["anchor"] = anchor.record()
                row["selection"].update(anchor_slot=anchor_slot, anchors=len(anchors),
                                        anchor_log_probability=None if net is None else float(anchor_log[0]),
                                        conditional_log_probability=None if net is None else float(local_log[0]))
            child = child_view = None; reason = None; rejected = None; result = None; auxiliary = []
            t = time.perf_counter()
            try:
                with operation_limit(min(args.rewrite_timeout, max(.001, deadline-time.perf_counter()))):
                    preview_key = feasibility_signature(state.diagram, chosen) if feasibility is not None else None
                    if preview_key in previews:
                        child, auxiliary = previews[preview_key]
                    else:
                        child, auxiliary = kernel.step(state.diagram, view, chosen)
                    if len(child.nodes) > raw_limit:
                        if feasibility is not None:
                            raise FeasibilityMismatch("A masked action exceeded its node budget")
                        raise ValueError("raw vertex growth bound exceeded")
                    if len(child.edges) > wire_limit:
                        if feasibility is not None:
                            raise FeasibilityMismatch("A masked action exceeded its wire budget")
                        raise ValueError("raw wire growth bound exceeded")
                    child_view = views.get(child)
            except (OperatorMismatch, FeasibilityMismatch):
                raise
            except Exception as exc:
                rejected = type(exc).__name__+": "+str(exc); invalid[rejected] += 1
            timers["rewrite_verify"] += time.perf_counter()-t
            if rejected is None and chosen.mode != "noop":
                t = time.perf_counter()
                try:
                    evaluation = scorer.evaluate(child_view, min(args.extraction_timeout, max(.001, deadline-time.perf_counter())))
                    result, reason, hit = evaluation.result, evaluation.reason, evaluation.hit
                    row.update(completion_branch=evaluation.branch, completion_failures=evaluation.failures)
                except OperatorMismatch:
                    raise
                except Exception as exc:
                    reason = type(exc).__name__+": "+str(exc); hit = False
                timers["score"] += time.perf_counter()-t
                row.update(graph_key=save_graph(out/"objects", child), cost=None if result is None else result["cost"],
                           auxiliary=auxiliary, score_failure=reason, cache_hit=hit,
                           view_key=child_view.key, raw_operator_error=child_view.error,
                           vertices=len(child.nodes), edges=len(child.edges))
                if result is not None:
                    path = out/"candidates"/(result["sha256"]+".qasm")
                    if not path.exists():
                        path.write_text(result["qasm"])
                    row["sha256"] = result["sha256"]
            elapsed = time.perf_counter()-started; row["seconds"] = elapsed
            if time.perf_counter() > deadline:
                row["status"] = "excluded_after_deadline"; events.write(json.dumps(row)+"\n"); break
            done = step >= args.horizon; reward = 0.
            if rejected is None and chosen.mode == "noop":
                if args.anchor_noop_stop:
                    done = True
                    anchor_stats["noop_terminations"] += 1
                row.update(status="anchor_noop", admitted=False, unscored_depth=state.unscored_depth,
                           graph_key=state.key, cost=state.cost, view_key=view.key,
                           raw_operator_error=view.error, vertices=len(child.nodes), edges=len(child.edges))
            elif rejected is None:
                old_unscored = state.unscored_depth
                state, reward = following_state(state, child, result)
                accepted += 1; accepted_families[chosen.family] += 1
                if result is None:
                    unscored += 1; latent_families[chosen.family] += 1
                    done = done or state.unscored_depth >= args.max_unscored
                else:
                    if shared is not None:
                        t = time.perf_counter()
                        shared.publish(result, child.digest())
                        timers["shared_exchange"] += time.perf_counter()-t
                    if old_unscored:
                        recovered += 1
                    done = done or result["cost"]["gates"] > 1.5*root_result["cost"]["gates"]
                    for target in first_best:
                        if first_best[target] is None and result["cost"]["gates"] <= int(target):
                            first_best[target] = elapsed
                    if score(result["cost"]) < score(best["cost"]):
                        gate_improvement = patience.observe(result["cost"]["gates"])
                        strict_rule_improvement = gate_improvement
                        deadline = min(budget_deadline, time.perf_counter()+patience.remaining())
                        best = dict(cost=result["cost"], sha256=result["sha256"], seconds=elapsed, graph=state.key)
                        (out/"best.qasm.tmp").write_text(result["qasm"]); (out/"best.qasm.tmp").replace(out/"best.qasm")
                        atomic_json(out/"best.json", best)
                        history = dict(**best, transition=transition, trajectory=trajectory,
                                       strict_gate_improvement=gate_improvement, patience=patience.record())
                        if gate_improvement and args.save_improvement_models and net is not None:
                            t = time.perf_counter()
                            snapshot = out/"improvement-models"/("g"+str(result["cost"]["gates"])+"-t"+str(transition)+".pt")
                            with snapshot.open("xb") as stream:
                                torch.save(dict(state_dict=net.state_dict(), steps=transition, updates=updates,
                                    optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
                                    input_sha256=protocol["input_sha256"], source_commit=revision,
                                    selection=args.selection, anchor_prior=args.anchor_prior, rule_profile=args.rule_profile,
                                    rule_schedule=args.rule_schedule, native_steps=args.native_steps,
                                    expanded_steps=args.expanded_steps, candidate_allocation=args.candidate_allocation,
                                    anchor_resource_mask=args.anchor_resource_mask,
                                    anchor_noop_stop=args.anchor_noop_stop,
                                    neutral_stop_steps=args.neutral_stop_steps,
                                    discount_factor=args.discount_factor, gae_lambda=args.gae_lambda,
                                    gate_improvement_stop=args.gate_improvement_stop,
                                    scope="Policy that generated this strict gate improvement, before this transition's PPO update; no active circuit/archive"), stream)
                            history["policy_snapshot"] = dict(path=str(snapshot.relative_to(out)),
                                sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), steps=transition, updates=updates)
                            timers["improvement_model"] += time.perf_counter()-t
                        best_history.write(json.dumps(history)+"\n")
                admitted = (pool.add(state, start.last_counts, None if result is None else result["sha256"])
                            if state.cost is not None or state.unscored_depth < args.max_unscored else False)
                row.update(status="accepted" if result is not None else "accepted_unscored", admitted=admitted,
                           unscored_depth=state.unscored_depth)
            else:
                row.update(status="rejected", reason=rejected)
            if args.gate_improvement_stop:
                success = bool(row["status"] == "accepted" and start.cost is not None
                               and row["cost"]["gates"] < start.cost["gates"])
                row["gate_improvement_stop"] = success
                if success:
                    done = True
                    anchor_stats["gate_improvement_terminations"] += 1
            if args.neutral_stop_steps:
                neutral_streak = (neutral_streak+1 if row["status"] in ("accepted", "anchor_noop")
                                  and row["cost"] is not None and reward == 0 else 0)
                neutral_stop = neutral_streak >= args.neutral_stop_steps
                row.update(neutral_streak=neutral_streak, neutral_stop=neutral_stop)
                if neutral_stop:
                    done = True
                    anchor_stats["neutral_terminations"] += 1
            row.update(reward=reward, terminal=done, best=best["cost"], buffer=pool.summary())
            events.write(json.dumps(row)+"\n")
            rule_schedule.observe(strict_rule_improvement)
            if optimizer is not None:
                rollout.append((obs, slot, old_log, value, reward, done))
            if done:
                state = None
            if optimizer is not None and len(rollout) >= args.rollout and time.perf_counter() < deadline:
                t = time.perf_counter(); bootstrap = 0.
                if state is not None:
                    next_view = views.get(state.diagram)
                    ctx = context_for(state, start, best, step, args, root_result["cost"]["gates"], len(diagram.nodes), neutral_streak)
                    next_obs = observation(state.diagram, next_view.graph, [], ctx)
                    with torch.no_grad():
                        bootstrap = float(net.get_value(batch([next_obs])[1])[0])
                options = dict(epochs=4, minibatch=32)
                if anchored:
                    options.update(discount_factor=args.discount_factor, gae_lambda=args.gae_lambda)
                update = update_ppo(net, optimizer, rollout, bootstrap, options, deadline)
                choices = [int((r[0][1 if anchored else 0].y != -1).sum()) for r in rollout]
                fraction = sum(k > 1 for k in choices)/len(choices)
                local_entropy = update["metrics"]["normalized_local_entropy"] if anchored else update["metrics"]["normalized_entropy"]
                update["metrics"]["conditional_normalized_entropy"] = (local_entropy/fraction if fraction else None)
                update["metrics"]["forced_action_fraction"] = 1-fraction
                update.update(transition=transition, seconds=time.perf_counter()-started)
                updates_file.write(json.dumps(update)+"\n"); updates += 1; rollout = []
                timers["ppo"] += time.perf_counter()-t
            if args.checkpoint_seconds and time.perf_counter()-last_checkpoint >= args.checkpoint_seconds:
                checkpoint_now()
                last_checkpoint = time.perf_counter()
            if transition % 64 == 0:
                atomic_json(out/"progress.json", dict(transitions=transition, accepted=accepted, unscored=unscored,
                                                     recovered=recovered, updates=updates, best=best,
                                                     seconds=time.perf_counter()-started, buffer=pool.summary(), timers=dict(timers),
                                                     anchor_stats=dict(anchor_stats),
                                                     patience=patience.record(), patience_remaining_seconds=patience.remaining()))
        if patience.remaining() <= 0:
            status = "no_improvement_timeout"
        if args.max_steps is not None and transition >= args.max_steps:
            status = "step_limit_reached"
    except BaseException as exc:
        atomic_json(out/"failure.json", dict(type=type(exc).__name__, message=str(exc),
                    transition=transition, last_proposal=row if "row" in locals() else None))
        raise
    finally:
        events.flush(); os.fsync(events.fileno()); events.close()
        updates_file.flush(); os.fsync(updates_file.fileno()); updates_file.close()
        restart_file.flush(); os.fsync(restart_file.fileno()); restart_file.close()
        deadend_file.flush(); os.fsync(deadend_file.fileno()); deadend_file.close()
        best_history.flush(); os.fsync(best_history.fileno()); best_history.close()
    loop_seconds = time.perf_counter()-started
    if net is not None:
        torch.save(dict(state_dict=net.state_dict(), steps=transition, updates=updates,
                        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
                        input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                        selection=args.selection, rule_profile=args.rule_profile,
                        rule_schedule=args.rule_schedule, native_steps=args.native_steps,
                        expanded_steps=args.expanded_steps, candidate_allocation=args.candidate_allocation,
                        anchor_prior=args.anchor_prior, learning_rate=args.learning_rate,
                        anchor_resource_mask=args.anchor_resource_mask,
                        anchor_noop_stop=args.anchor_noop_stop,
                        neutral_stop_steps=args.neutral_stop_steps,
                        discount_factor=args.discount_factor, gae_lambda=args.gae_lambda,
                        gate_improvement_stop=args.gate_improvement_stop,
                        scope="model/Adam only, not a complete search checkpoint"), out/"final.pt")
    checkpoint_now()
    if shared is not None:
        shared.pool.snapshot(out/"shared-buffer.sqlite")
        shared.pool.close()
    for name, digest in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise AssertionError("Source changed during run: "+name)
    report = dict(status=status, transitions=transition, accepted=accepted, unscored=unscored, recovered=recovered,
                  updates=updates, best=best, root_cost=root_result["cost"], proposal_counts=dict(families),
                  accepted_counts=dict(accepted_families), unscored_counts=dict(latent_families),
                  rejections=dict(invalid), buffer=pool.summary(), checked_qasm=len(scorer.valid_qasm),
                  view_cache=dict(hits=views.hits, misses=views.misses),
                  jpv_proposals=None if kernel.jpv is None else kernel.jpv.stats,
                  anchor_stats=dict(anchor_stats),
                  completion=scorer.summary(),
                  feasibility=None if feasibility is None else feasibility.summary(),
                  max_graph_operator_error=views.max_error, max_qasm_operator_error=scorer.max_valid_error,
                  first_passage_seconds=first_best, budget_seconds=args.seconds,
                  total_loop_seconds=loop_seconds, timers=dict(timers), verification="passed",
                  verification_method=protocol["verification_method"], patience=patience.record(),
                  certificate_stats=dict(reference.stats) if isinstance(reference, ZXReference) else None,
                  finished_unix=time.time())
    if native_flow is not None:
        report["native_flow"] = dict(**native_flow,
            primary_statistics=dict(scorer.primary.scorer.flow_statistics),
            repair_statistics=dict(scorer.repair.scorer.flow_statistics))
    atomic_json(out/"report.json", report)
    print(json.dumps(report), flush=True)
    return report


def _checkpoint_payload(protocol, pool, state, start, step, restart_source, rollout,
                        best, best_qasm, first_best, counts, net, optimizer, kernel,
                        rng, shared, patience):
    return dict(schema=1, input_sha256=protocol["input_sha256"], sources=protocol["sources"], config=protocol["config"],
                      pool=pool, state=state, start=start, step=step,
                      restart_source=restart_source, rollout=rollout, best=best,
                      best_qasm=best_qasm, first_best=first_best, counts=counts,
                      patience=patience.record(),
                      net=None if net is None else net.state_dict(),
                      optimizer=None if optimizer is None else optimizer.state_dict(),
                      jpv=None if kernel.jpv is None else (kernel.jpv.rounds, kernel.jpv.stats),
                      proposal_cap_rounds=kernel.cap_rounds,
                      rng=rng.bit_generator.state, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                      torch_rng=torch.get_rng_state(), exchange_rng=None if shared is None else shared.rng.getstate(),
                      scope="Trusted local checkpoint. Native pools, active trajectory, pending on-policy rollout, model/Adam and RNG retained. Recomputable matching/view/score caches excluded; wall-budget bitwise replay not guaranteed.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--resume-state", type=Path, help="Trusted complete search checkpoint; additional segment budget")
    p.add_argument("--resume-model", type=Path, help="Model/Adam continuation only; new trajectories, not exact state resume")
    p.add_argument("--shared-buffer", type=Path, help="Opt-in verified cross-IR SQLite restart pool")
    p.add_argument("--input", type=Path, default=ROOT/"inputs/barenco_tof_3.qasm")
    p.add_argument("--jpv", action="store_true")
    p.add_argument("--general-zx", action="store_true",
                   help="Enable general-phase templates and checked nonlinear A; implies --jpv --allow-arbitrary-angles")
    p.add_argument("--allow-arbitrary-angles", action="store_true",
                   help="Preserve non-pi/4 phases as context; skip inapplicable JPV matches")
    p.add_argument("--policy", choices=["uniform", "frozen", "online"], default="uniform")
    p.add_argument("--rule-profile", choices=["full", "no-a", "macros", "native"], default="full",
                   help="Opt-in global-selection rule ablation; model schema remains general ZX")
    p.add_argument("--feasibility-mode", choices=["reference", "bounds"], default="bounds",
                   help="Bounds certify resource-safe candidates without constructing every child; uncertain cases use exact previews")
    p.add_argument("--rule-schedule", choices=["static", "staged"], default="static")
    p.add_argument("--native-steps", type=int, default=256)
    p.add_argument("--expanded-steps", type=int, default=64)
    p.add_argument("--candidate-allocation", choices=["balanced", "native-heavy"], default="balanced")
    p.add_argument("--selection", choices=["global", "anchor"], default="global",
                   help="Sample a topology-only anchor before any rule matching; global preserves the baseline")
    p.add_argument("--anchor-candidate-cap", type=int, default=8,
                   help="Maximum local instances/previews after sampling one anchor")
    p.add_argument("--anchor-prior", choices=["uniform", "nodes"], default="uniform",
                   help="Optional topology-only node-focused category prior, included in the PPO density")
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--discount-factor", type=float, default=.99)
    p.add_argument("--gae-lambda", type=float, default=.95)
    p.add_argument("--gate-improvement-stop", action="store_true",
                   help="Opt-in terminal after a strict verified gate reduction below the known restart cost")
    p.add_argument("--anchor-resource-mask", action="store_true",
                   help="Before sampling, remove only wire/scalar anchors whose insertion is provably over budget")
    p.add_argument("--anchor-noop-stop", action="store_true",
                   help="Opt-in terminal empty-anchor noop; record zero reward then restart next step")
    p.add_argument("--neutral-stop-steps", type=int, default=0,
                   help="Opt-in trajectory stop after N consecutive scored zero-reward rewrites/noops;0 disables")
    p.add_argument("--save-improvement-models", action="store_true",
                   help="Save policy/Adam on each strict total-gate improvement, before the next PPO update")
    p.add_argument("--native-flow-module", type=Path,
                   help="Opt-in built CPU CFlow scoring module with matching build.json receipt")
    p.add_argument("--verify-native-flow", action="store_true",
                   help="Development check: compare every native scoring batch and flow match update to pinned PyZX")
    p.add_argument("--seconds", type=float, default=1200.)
    p.add_argument("--no-improvement-seconds", type=float, default=0., help="Strict total-gate plateau patience; --seconds 0 disables the overall cap")
    p.add_argument("--checkpoint-seconds", type=float, default=0., help="Atomic full search-state checkpoint interval")
    p.add_argument("--verification", choices=["matrix", "adaptive"], default="matrix", help="Adaptive retains full matrices up to7 wires, sufficient fixed-wire ZX identity reduction above7")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-growth", type=int, default=64)
    p.add_argument("--max-edge-growth", type=int, default=256)
    p.add_argument("--candidate-cap", type=int, default=32)
    p.add_argument("--total-candidate-cap", type=int, default=0, help="Family-balanced deterministic menu before exact previews;0 preserves the full native menu")
    p.add_argument("--matching-work", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=70)
    p.add_argument("--max-unscored", type=int, default=8)
    p.add_argument("--archive", type=int, default=64)
    p.add_argument("--latent-archive", type=int, default=128)
    p.add_argument("--archive-qasm-cap", type=int, default=0,
                   help="Opt-in per-verified-QASM raw restart representatives; prefer compact graphs;0 preserves the baseline")
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--extraction-timeout", type=float, default=2.)
    p.add_argument("--rewrite-timeout", type=float, default=5.)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--no-feasibility-mask", dest="feasibility_mask", action="store_false")
    p.set_defaults(feasibility_mask=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
