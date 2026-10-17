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


def tensor_digest(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def candidate_digest(available):
    return hashlib.sha256(json.dumps([a.record() for a in available], sort_keys=True).encode()).hexdigest()


def save_graph(folder, diagram):
    key = diagram.digest(); path = folder/(key+".json.gz")
    if not path.exists():
        path.write_bytes(gzip.compress(json.dumps(diagram.record(), separators=(",", ":")).encode(), compresslevel=1, mtime=0))
    return key


def context_for(state, start, best, step, args, root_gates, root_vertices):
    return [(state.last_counts["gates"]-root_gates)/root_gates,
            (best["cost"]["gates"]-root_gates)/root_gates, step/args.horizon,
            state.unscored_depth/args.max_unscored, float(state.cost is not None),
            start.last_counts["gates"]/root_gates,
            (len(state.diagram.nodes)-root_vertices)/max(1, args.max_growth),
            float(state.unscored_depth > 0)]


def run(args):
    if (args.seconds <= 0 or args.horizon < 1 or args.max_unscored < 1 or args.max_growth < 0
            or args.candidate_cap < 1 or args.matching_work < 1):
        raise ValueError("Invalid search bounds")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/"objects").mkdir(); (out/"candidates").mkdir()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = environment(args.input); n = len(env.graph.inputs())
    reference = qiskit_matrix_zx_order(qasm2.loads(args.input.read_text()))
    views = Views(n, reference); scorer = VerifiedCompletion(reference, out/"invalid-extractions")
    kernel = GeneralKernel(env, args.seed, args.jpv, args.candidate_cap, args.matching_work)
    diagram = from_pyzx(env.graph)
    root_view = views.get(diagram)
    if root_view.graph is None:
        raise RuntimeError("Initial normalization failed")
    root_evaluation = scorer.evaluate(root_view, args.extraction_timeout)
    root_result = root_evaluation.result
    if root_result is None:
        raise RuntimeError("Initial completion failed: "+str(root_evaluation.reason))
    root_error = matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(root_result["qasm"])))
    if root_error > 2e-8:
        raise OperatorMismatch("Initial extracted circuit changed the operator")
    root = SearchState(diagram, root_result["cost"], root_result["cost"])
    pool = GeneralArchive(root, args.archive, args.latent_archive)
    raw_limit = len(diagram.nodes)+args.max_growth
    wire_limit = len(diagram.edges)+args.max_edge_growth
    feasibility = Feasibility(raw_limit, wire_limit) if args.feasibility_mask else None
    view_limit = root_view.graph.num_vertices()+args.max_growth
    net = GeneralAgent() if args.policy != "uniform" else None
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003, eps=1e-5) if args.policy == "online" else None
    initial_sha = tensor_digest(net) if net is not None else None
    if net is not None:
        torch.save(dict(state_dict=net.state_dict(), seed=args.seed, steps=0), out/"initial.pt")
    source_paths = list((ROOT/"zxopt").rglob("*.py"))
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    git = ["git", "--git-dir="+str(ROOT/".git"), "--work-tree="+str(ROOT)]
    revision = subprocess.check_output(git+["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(git+["status", "--porcelain", "--untracked-files=no"], text=True).strip()
    baseline = json.loads((ROOT/"docs/baseline_sources.json").read_text())
    for path, digest in baseline["files"].items():
        if hashlib.sha256((LEGACY/path).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen baseline modified: "+path)
    protocol = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    git_commit=revision, tracked_dirty=bool(dirty), sources=sources,
                    initial_tensor_sha256=initial_sha, started_at=datetime.datetime.now().astimezone().isoformat(),
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    baseline_manifest_sha256=hashlib.sha256((ROOT/"docs/baseline_sources.json").read_bytes()).hexdigest(),
                    root_cost=root_result["cost"], root_graph=diagram.digest(), root_vertices=len(diagram.nodes),
                    root_view_vertices=root_view.graph.num_vertices(), max_vertices=raw_limit,
                    scope="barenco_tof_3 only; total NAM gates then CX; no historical witness or held-out circuits",
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
    protocol["feasibility_mask"] = "When enabled, rewrite-only previews exclude node/wire-budget violations before categorical sampling; no extraction/cost lookahead. Exact-state/action LRU; all preview work inside budget."
    protocol["preview_native_prefix_sha256"] = kernel.preview_prefix_sha256
    atomic_json(out/"protocol.json", protocol)
    atomic_json(out/"hh-derivation.json", hh_identity()[2])
    save_graph(out/"objects", diagram)
    (out/"baseline.qasm").write_text(root_result["qasm"]); (out/"best.qasm").write_text(root_result["qasm"])
    best = dict(cost=root_result["cost"], sha256=root_result["sha256"], seconds=0., graph=diagram.digest())
    atomic_json(out/"best.json", best)
    events = (out/"events.jsonl").open("w", buffering=1); updates_file = (out/"updates.jsonl").open("w", buffering=1)
    restart_file = (out/"restarts.jsonl").open("w", buffering=1)
    deadend_file = (out/"deadends.jsonl").open("w", buffering=1)
    started = time.perf_counter(); deadline = started+args.seconds
    transition = trajectory = accepted = unscored = recovered = updates = 0
    invalid = Counter(); families = Counter(); accepted_families = Counter(); latent_families = Counter()
    timers = Counter(); rollout = []
    state = None; status = "budget_reached"; start = None
    first_best = {str(k): None for k in (39, 38, 37, 36, 35)}

    try:
        while time.perf_counter() < deadline and (args.max_steps is None or transition < args.max_steps):
            if state is None:
                start, restart_source = pool.choose(rng); state = start
                step = 0; trajectory += 1
                restart_file.write(json.dumps(dict(trajectory=trajectory, graph_key=start.key,
                                   source=restart_source, buffer=pool.summary()))+"\n")
            t = time.perf_counter()
            view = views.get(state.diagram)
            available = kernel.prepare(state.diagram, view, view_limit)
            timers["matching"] += time.perf_counter()-t
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
            if not available:
                deadend_file.write(json.dumps(dict(trajectory=trajectory, graph_key=state.key,
                                                  seconds=time.perf_counter()-started))+"\n")
                if rollout:
                    previous = list(rollout[-1]); previous[-1] = True; rollout[-1] = tuple(previous)
                state = None
                continue
            context = context_for(state, start, best, step, args, root_result["cost"]["gates"], len(diagram.nodes))
            t = time.perf_counter(); obs = observation(state.diagram, view.graph, available, context)
            if net is None:
                slot = int(rng.integers(len(available))); old_log = value = None
            else:
                with torch.no_grad():
                    packed = batch([obs]); logits = net.actor(packed[0]).flatten()[packed[0].y != -1].double().numpy()
                    probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
                    slot = min(int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="right")), len(available)-1)
                    _, logs, _, values, _, _ = net.get_action_and_value(packed, action=torch.tensor([slot]))
                    old_log = float(logs[0]); value = float(values[0])
            timers["inference"] += time.perf_counter()-t
            chosen = available[slot]; transition += 1; step += 1; families[chosen.family] += 1
            row = dict(transition=transition, trajectory=trajectory, trajectory_step=step,
                       parent_graph_key=state.key, restart_graph_key=start.key, restart_source=restart_source,
                       action=chosen.record(), available=len(available), candidate_sha256=candidate_digest(available),
                       selection=dict(slot=slot, log_probability=old_log, value=value), context=context,
                       parent_cost=state.cost, last_scored_cost=state.last_counts, unscored_depth_before=state.unscored_depth)
            child = child_view = None; reason = None; rejected = None; result = None; auxiliary = []
            t = time.perf_counter()
            try:
                with operation_limit(min(5., max(.001, deadline-time.perf_counter()))):
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
            if rejected is None:
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
            if elapsed > args.seconds:
                row["status"] = "excluded_after_deadline"; events.write(json.dumps(row)+"\n"); break
            done = step >= args.horizon; reward = 0.
            if rejected is None:
                old_unscored = state.unscored_depth
                state, reward = following_state(state, child, result)
                accepted += 1; accepted_families[chosen.family] += 1
                if result is None:
                    unscored += 1; latent_families[chosen.family] += 1
                    done = done or state.unscored_depth >= args.max_unscored
                else:
                    if old_unscored:
                        recovered += 1
                    done = done or result["cost"]["gates"] > 1.5*root_result["cost"]["gates"]
                    for target in first_best:
                        if first_best[target] is None and result["cost"]["gates"] <= int(target):
                            first_best[target] = elapsed
                    if score(result["cost"]) < score(best["cost"]):
                        best = dict(cost=result["cost"], sha256=result["sha256"], seconds=elapsed, graph=state.key)
                        (out/"best.qasm.tmp").write_text(result["qasm"]); (out/"best.qasm.tmp").replace(out/"best.qasm")
                        atomic_json(out/"best.json", best)
                admitted = (pool.add(state, start.last_counts)
                            if state.cost is not None or state.unscored_depth < args.max_unscored else False)
                row.update(status="accepted" if result is not None else "accepted_unscored", admitted=admitted,
                           unscored_depth=state.unscored_depth)
            else:
                row.update(status="rejected", reason=rejected)
            row.update(reward=reward, terminal=done, best=best["cost"], buffer=pool.summary())
            events.write(json.dumps(row)+"\n")
            if optimizer is not None:
                rollout.append((obs, slot, old_log, value, reward, done))
            if done:
                state = None
            if optimizer is not None and len(rollout) >= args.rollout and time.perf_counter() < deadline:
                t = time.perf_counter(); bootstrap = 0.
                if state is not None:
                    next_view = views.get(state.diagram)
                    ctx = context_for(state, start, best, step, args, root_result["cost"]["gates"], len(diagram.nodes))
                    next_obs = observation(state.diagram, next_view.graph, [], ctx)
                    with torch.no_grad():
                        bootstrap = float(net.get_value(batch([next_obs])[1])[0])
                update = ppo_update(net, optimizer, rollout, bootstrap, dict(epochs=4, minibatch=32), deadline)
                choices = [int((r[0][0].y != -1).sum()) for r in rollout]
                fraction = sum(k > 1 for k in choices)/len(choices)
                update["metrics"]["conditional_normalized_entropy"] = (update["metrics"]["normalized_entropy"]/fraction if fraction else None)
                update["metrics"]["forced_action_fraction"] = 1-fraction
                update.update(transition=transition, seconds=time.perf_counter()-started)
                updates_file.write(json.dumps(update)+"\n"); updates += 1; rollout = []
                timers["ppo"] += time.perf_counter()-t
            if transition % 64 == 0:
                atomic_json(out/"progress.json", dict(transitions=transition, accepted=accepted, unscored=unscored,
                                                     recovered=recovered, updates=updates, best=best,
                                                     seconds=time.perf_counter()-started, buffer=pool.summary(), timers=dict(timers)))
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
    loop_seconds = time.perf_counter()-started
    if net is not None:
        torch.save(dict(state_dict=net.state_dict(), steps=transition, updates=updates,
                        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
                        scope="model/Adam only, not a complete search checkpoint"), out/"final.pt")
    for name, digest in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise AssertionError("Source changed during run: "+name)
    report = dict(status=status, transitions=transition, accepted=accepted, unscored=unscored, recovered=recovered,
                  updates=updates, best=best, root_cost=root_result["cost"], proposal_counts=dict(families),
                  accepted_counts=dict(accepted_families), unscored_counts=dict(latent_families),
                  rejections=dict(invalid), buffer=pool.summary(), checked_qasm=len(scorer.valid_qasm),
                  view_cache=dict(hits=views.hits, misses=views.misses),
                  jpv_proposals=None if kernel.jpv is None else kernel.jpv.stats,
                  completion=scorer.summary(),
                  feasibility=None if feasibility is None else feasibility.summary(),
                  max_graph_operator_error=views.max_error, max_qasm_operator_error=scorer.max_valid_error,
                  first_passage_seconds=first_best, budget_seconds=args.seconds,
                  total_loop_seconds=loop_seconds, timers=dict(timers), verification="passed")
    atomic_json(out/"report.json", report)
    print(json.dumps(report), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--input", type=Path, default=ROOT/"inputs/barenco_tof_3.qasm")
    p.add_argument("--jpv", action="store_true")
    p.add_argument("--policy", choices=["uniform", "frozen", "online"], default="uniform")
    p.add_argument("--seconds", type=float, default=1200.)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-growth", type=int, default=64)
    p.add_argument("--max-edge-growth", type=int, default=256)
    p.add_argument("--candidate-cap", type=int, default=32)
    p.add_argument("--matching-work", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=70)
    p.add_argument("--max-unscored", type=int, default=8)
    p.add_argument("--archive", type=int, default=64)
    p.add_argument("--latent-archive", type=int, default=128)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--extraction-timeout", type=float, default=2.)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--no-feasibility-mask", dest="feasibility_mask", action="store_false")
    p.set_defaults(feasibility_mask=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
