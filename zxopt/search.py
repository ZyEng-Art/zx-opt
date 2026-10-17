"""Single-case search with recorded proposals, exact costs and optional online PPO."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from qiskit import qasm2

from .legacy import ROOT, LEGACY, environment, oracle
from .kernel import Kernel
from .rules import candidates
from .dedup import StructuralIndex, action_key
from .verify import graph_matrix, matrix_error, qiskit_matrix_zx_order
from .policy import ExpandedAgent, observation, batch
from buffer import CircuitBuffer, Entry, score
from runtime import graph_key
from monitored_ppo import ppo_update


class EquivalenceError(RuntimeError):
    pass


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def operation_limit(seconds):
    def alarm(signum, frame):
        raise TimeoutError("candidate operation exceeded deadline")
    handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def run(args):
    if args.seconds <= 0 or args.horizon < 1 or args.candidate_cap < 0 or args.max_growth < 0:
        raise ValueError("Invalid search budget or bounds")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "candidates").mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = environment(args.input)
    kernel = Kernel(env)
    scorer = oracle()
    reference = qiskit_matrix_zx_order(qasm2.loads(args.input.read_text()))
    if matrix_error(reference, graph_matrix(env.graph)) > 2e-8:
        raise AssertionError("Initial ZX graph differs from supplied circuit")
    root_graph = env.graph.clone()
    root_key, root_result, _ = scorer.evaluate(root_graph)
    if matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(root_result["qasm"]))) > 2e-8:
        raise AssertionError("Initial completion differs from supplied circuit")
    index = StructuralIndex()
    root_state, _, _ = index.intern(root_graph)
    root_pool_key = str(root_state.identifier) if args.dedup else root_key
    pool = CircuitBuffer(Entry(root_pool_key, root_result["cost"], root_graph), 64)
    max_vertices = root_graph.num_vertices() + args.max_growth
    net = ExpandedAgent() if args.policy != "uniform" else None
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003, eps=1e-5) if args.policy == "online" else None
    if net is not None:
        torch.save(dict(state_dict=net.state_dict(), steps=0, seed=args.seed), out / "initial.pt")
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "zxopt").glob("*.py"))}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    config = dict(vars(args)); config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    baseline = json.loads((ROOT / "docs/baseline_sources.json").read_text())
    for path, digest in baseline["files"].items():
        if hashlib.sha256((LEGACY / path).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen baseline source changed: " + path)
    protocol = dict(config=config, git_commit=revision, tracked_dirty=bool(dirty), sources=sources,
                    baseline_manifest_sha256=hashlib.sha256((ROOT/"docs/baseline_sources.json").read_bytes()).hexdigest(),
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    started_at=datetime.datetime.now().astimezone().isoformat(),
                    root_cost=root_result["cost"], root_vertices=root_graph.num_vertices(),
                    max_vertices=max_vertices, input_scope="barenco_tof_3 only, no held-out circuits",
                    cost="unchanged v49 fixed-wire completion + NAM cleanup; gates then CX",
                    verification="every scored graph: independent Z-factor contraction; every unique QASM: Qiskit full operator; only unit-modulus global phase",
                    dedup="WL bucket + exact labelled isomorphism, scalar and ordered boundaries retained; cost cache remains serialized-state",
                    timing="setup and final checkpoint/report excluded; matching, proposals, masking, policy, verification, completion, archive, PPO and streaming log inside search wall time; late candidates excluded",
                    invalid_proposal="zero-reward self-loop, attempted action recorded; never enters archive",
                    learning="v49 GAT/context PPO plus new-rule embeddings; current NAM delta / 10 or trajectory record delta / 10; 128 rollout, 32 minibatch, 4 epochs",
                    limits="finite pi/4 phase menu, bounded partitions/candidates/vertices/horizon; not a complete calculus")
    atomic_json(out / "protocol.json", protocol)
    (out / "baseline.qasm").write_text(root_result["qasm"])
    (out / "best.qasm").write_text(root_result["qasm"])
    best = dict(cost=root_result["cost"], sha256=root_result["sha256"], seconds=0.)
    checked_qasm = {root_result["sha256"]}
    errors = {}; counts = {}; accepted_counts = {}; exhausted = set()
    rollout = []; updates = []; transitions = accepted = 0
    active = False; trajectory = 0; step = 0; context_record = 0
    max_graph_error = max_qasm_error = 0.
    started = time.perf_counter(); deadline = started + args.seconds
    events = (out / "events.jsonl").open("w", buffering=1)
    update_log = (out / "updates.jsonl").open("w", buffering=1)
    status = "budget_reached"

    def build(g, counts, start_counts, record, step):
        state, mapping, fresh = index.intern(g)
        obs, old_actions = kernel.prepare(g)
        extra = candidates(g, max_vertices, args.candidate_cap) if args.expanded else []
        scale = max(1, root_result["cost"]["gates"])
        ctx = [(counts["gates"]-start_counts["gates"])/scale,
               (record-start_counts["gates"])/scale, step/args.horizon,
               start_counts["gates"]/scale]
        obs, actions = observation(obs, old_actions, extra, ctx,
                                   state.attempted if args.dedup else None, mapping)
        return state, mapping, obs, actions

    try:
        while time.perf_counter() < deadline and (args.max_steps is None or transitions < args.max_steps):
            if not active:
                available = [e for k, e in pool.entries.items() if k not in exhausted]
                if not available:
                    status = "retained_frontier_exhausted"; break
                start = pool.choose("diverse", rng)
                for _ in range(16):
                    if start.key not in exhausted: break
                    start = pool.choose("diverse", rng)
                if start.key in exhausted: start = available[int(rng.integers(len(available)))]
                graph = start.payload.clone(); previous = start.counts
                record = previous["gates"]; step = 0; trajectory += 1; active = True
            state, mapping, obs, available = build(graph, previous, start.counts, record, step)
            if not available:
                exhausted.add(str(state.identifier) if args.dedup else graph_key(graph))
                active = False
                if rollout:
                    last = list(rollout[-1]); last[-1] = True; rollout[-1] = tuple(last)
                continue
            if net is None:
                slot = int(rng.integers(len(available))); old_log = value = None
            else:
                with torch.no_grad():
                    chosen, lp, _, val, _, _ = net.get_action_and_value(batch([obs]), device="cpu")
                slot = int(chosen[0]); old_log = float(lp[0]); value = float(val[0])
            action, identifier = available[slot]
            signature = action_key(action, mapping)
            if args.dedup:
                assert signature not in state.attempted
                state.attempted.add(signature)
            transitions += 1; step += 1
            counts[action.kind] = counts.get(action.kind, 0) + 1
            row = dict(transition=transitions, trajectory=trajectory, trajectory_step=step,
                       parent_state=state.identifier, parent_graph_key=graph_key(graph),
                       restart_key=start.key, action=action.record(), legacy_identifier=identifier,
                       available=len(available), previous_cost=previous, previous_record=record,
                       selection=dict(slot=slot, log_probability=old_log, value=value),
                       context=obs[0].context.tolist())
            reward = 0.; following = None; candidate = None; done = step >= args.horizon
            try:
                with operation_limit(min(5., deadline-time.perf_counter())):
                    following = kernel.step(graph, action, identifier)
                    if following.num_vertices() > max_vertices:
                        raise ValueError("vertex growth bound exceeded")
                    error = matrix_error(reference, graph_matrix(following))
                    max_graph_error = max(max_graph_error, error)
                    if error > 2e-8: raise EquivalenceError("ZX rewrite operator mismatch: " + str(error))
                    key, candidate, hit = scorer.evaluate(following)
                    if candidate["sha256"] not in checked_qasm:
                        error = matrix_error(reference, qiskit_matrix_zx_order(qasm2.loads(candidate["qasm"])))
                        max_qasm_error = max(max_qasm_error, error)
                        if error > 2e-8: raise EquivalenceError("extracted operator mismatch: " + str(error))
                        checked_qasm.add(candidate["sha256"])
                    row.update(graph_key=key, cache_hit=hit, cost=candidate["cost"], sha256=candidate["sha256"])
            except EquivalenceError:
                raise  # An unsound rewrite/extraction must fail the run, never be hidden as an ordinary rejection.
            except (Exception, MemoryError) as exc:
                name = type(exc).__name__ + ": " + str(exc)
                errors[name] = errors.get(name, 0) + 1
                row.update(status="rejected", reason=name)
                candidate = None
            elapsed = time.perf_counter() - started
            row["seconds"] = elapsed
            if elapsed > args.seconds:
                row["status"] = "excluded_after_deadline"
                events.write(json.dumps(row) + "\n")
                break
            if candidate is not None:
                accepted += 1; accepted_counts[action.kind] = accepted_counts.get(action.kind, 0) + 1
                child, _, fresh = index.intern(following)
                child_key = str(child.identifier) if args.dedup else row["graph_key"]
                record_after = min(record, candidate["cost"]["gates"])
                reward = ((previous["gates"]-candidate["cost"]["gates"]) if args.reward == "current"
                          else (record-record_after)) / 10.
                done = done or candidate["cost"]["gates"] > 1.5 * root_result["cost"]["gates"]
                admitted = pool.add(Entry(child_key, candidate["cost"], following.clone(), step), start.counts)
                row.update(status="accepted", state=child.identifier, structurally_new=fresh,
                           admitted=admitted, vertices=following.num_vertices(), edges=following.num_edges())
                path = out / "candidates" / (candidate["sha256"] + ".qasm")
                if not path.exists(): path.write_text(candidate["qasm"])
                if score(candidate["cost"]) < score(best["cost"]):
                    best = dict(cost=candidate["cost"], sha256=candidate["sha256"], seconds=elapsed)
                    tmp = out / "best.qasm.tmp"; tmp.write_text(candidate["qasm"]); tmp.replace(out / "best.qasm")
                    atomic_json(out / "best.json", best)
                previous = candidate["cost"]; graph = following; record = record_after
            row.update(reward=reward, terminal=done, best=best["cost"], buffer=pool.summary())
            events.write(json.dumps(row) + "\n")
            active = not done
            if optimizer is not None:
                rollout.append((obs, slot, old_log, value, reward, done))
                if len(rollout) >= 128 and time.perf_counter() < deadline:
                    bootstrap = 0.
                    if active:
                        _, _, next_obs, next_actions = build(graph, previous, start.counts, record, step)
                        if next_actions:
                            with torch.no_grad(): bootstrap = float(net.get_value(batch([next_obs])[1])[0])
                        else:
                            last = list(rollout[-1]); last[-1] = True; rollout[-1] = tuple(last)
                            active = False
                    update = ppo_update(net, optimizer, rollout, bootstrap,
                                        dict(epochs=4, minibatch=32), deadline)
                    update.update(transition=transitions, seconds=time.perf_counter()-started)
                    updates.append(update); update_log.write(json.dumps(update) + "\n"); rollout = []
            if transitions % 128 == 0:
                atomic_json(out / "progress.json", dict(transitions=transitions, accepted=accepted,
                            best=best, seconds=time.perf_counter()-started, index=index.summary()))
        if args.max_steps is not None and transitions >= args.max_steps:
            status = "step_limit_reached"
    except BaseException as exc:
        atomic_json(out / "failure.json", dict(type=type(exc).__name__, message=str(exc), transition=transitions,
                    last_proposal=row if "row" in locals() else None))
        raise
    finally:
        events.flush(); os.fsync(events.fileno()); events.close()
        update_log.flush(); os.fsync(update_log.fileno()); update_log.close()
    if net is not None:
        torch.save(dict(state_dict=net.state_dict(), steps=transitions, updates=len(updates),
                        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
                        scope="model/Adam only, not full search resume"), out / "final.pt")
    report = dict(status=status, transitions=transitions, accepted=accepted, updates=len(updates),
                  best=best, root_cost=root_result["cost"], proposal_counts=counts,
                  accepted_counts=accepted_counts, rejections=errors, index=index.summary(),
                  checked_qasm=len(checked_qasm), max_graph_operator_error=max_graph_error,
                  max_qasm_operator_error=max_qasm_error, verification="passed",
                  budget_seconds=args.seconds, total_loop_seconds=time.perf_counter()-started)
    atomic_json(out / "report.json", report)
    print(json.dumps(report), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--input", type=Path, default=ROOT/"inputs/barenco_tof_3.qasm")
    p.add_argument("--seconds", type=float, default=120.)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--policy", choices=["uniform", "frozen", "online"], default="uniform")
    p.add_argument("--expanded", action="store_true")
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--reward", choices=["current", "record"], default="current")
    p.add_argument("--max-growth", type=int, default=16)
    p.add_argument("--candidate-cap", type=int, default=32)
    p.add_argument("--horizon", type=int, default=35)
    p.add_argument("--max-steps", type=int)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
