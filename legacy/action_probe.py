"""Synthetic one-step ZX diagnostic, not an optimizer or paper reproduction.

Run with PyZX 0.10.0 and NumPy. No held-out project circuits are used.
Matrix checks share PyZX's circuit semantics: these are consistency checks,
not independent certification. No candidate is integrated or accepted.
"""
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyzx as zx


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def evaluate(graph):
    circuit, extraction = timed(lambda: zx.extract_circuit(graph.copy(), up_to_perm=False).to_basic_gates())
    clean, cleanup = timed(lambda: zx.basic_optimization(circuit, do_swaps=False).to_basic_gates())
    return clean, dict(extraction_seconds=extraction, cleanup_seconds=cleanup)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    rows, states = [], []
    for qubits in (3, 5):
        for depth in (30, 60):
            for seed in (15001, 15002, 15003):
                case = f'q{qubits}_d{depth}_s{seed}'
                graph = zx.generate.cliffordT(qubits, depth, p_t=.25, p_s=.25, p_hsh=.25, seed=seed)
                source = zx.Circuit.from_graph(graph).to_basic_gates()
                text = source.to_qasm()
                (out / (case + '.qasm')).write_text(text)
                zx.to_graph_like(graph)
                base, base_time = evaluate(graph)
                reference = source.to_matrix()
                start = time.perf_counter()
                actions = [('lc', (v,)) for v in graph.vertices() if zx.lcomp_rule.check_lcomp(graph, v)]
                actions += [('pivot', graph.edge_st(e)) for e in graph.edges()
                            if zx.pivot_rule.check_pivot(graph, *graph.edge_st(e))]
                matching = time.perf_counter() - start
                random.Random(seed).shuffle(actions)
                state = dict(case=case, qubits=qubits, generator_depth=depth, seed=seed,
                             source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             source_gates=len(source.gates), base_gates=len(base.gates),
                             available_actions=len(actions), evaluated_actions=min(24, len(actions)),
                             matching_seconds=matching, **base_time)
                states.append(state)
                for kind, vertices in actions[:24]:
                    row = dict(case=case, kind=kind, vertices=vertices)
                    rows.append(row)
                    try:
                        # copy() relabels IDs; clone() preserves the matched vertices.
                        trial, copy_time = timed(graph.clone)
                        rule = zx.lcomp_rule.lcomp if kind == 'lc' else zx.pivot_rule.pivot
                        applied, rewrite_time = timed(lambda: rule(trial, *vertices))
                        if not applied:
                            raise RuntimeError('Matched rule was not applied')
                        candidate, costs = evaluate(trial)
                        matrix, matrix_time = timed(candidate.to_matrix)
                        overlap = np.vdot(reference, matrix)
                        phase = overlap / abs(overlap) if abs(overlap) else 1
                        error = float(np.linalg.norm(matrix - phase * reference) / np.linalg.norm(reference))
                        row.update(status='consistent' if error < 1e-9 else 'matrix_mismatch',
                                   relative_matrix_error=error, matrix_seconds=matrix_time,
                                   copy_seconds=copy_time, rewrite_seconds=rewrite_time,
                                   nodes_removed=graph.num_vertices()-trial.num_vertices(),
                                   edges_removed=graph.num_edges()-trial.num_edges(),
                                   gates_saved=len(base.gates)-len(candidate.gates), **costs)
                    except Exception as exc:
                        row.update(status='error', error_type=type(exc).__name__, error=str(exc))
                print(case, len(actions), flush=True)
    valid = [r for r in rows if r['status'] == 'consistent']
    totals = {key: sum(r[key] for r in valid) for key in
              ('copy_seconds', 'rewrite_seconds', 'extraction_seconds', 'cleanup_seconds', 'matrix_seconds')}
    summary = dict(states=len(states), candidates=len(rows), consistent=len(valid),
                   errors=[r for r in rows if r['status'] != 'consistent'],
                   improved=sum(r['gates_saved'] > 0 for r in valid),
                   unchanged=sum(r['gates_saved'] == 0 for r in valid),
                   worse=sum(r['gates_saved'] < 0 for r in valid),
                   fewer_edges_but_worse=sum(r['edges_removed'] > 0 and r['gates_saved'] < 0 for r in valid),
                   fewer_nodes_but_worse=sum(r['nodes_removed'] > 0 and r['gates_saved'] < 0 for r in valid),
                   stage_seconds=totals)
    report = dict(scope=__doc__, pyzx_version=zx.__version__, numpy_version=np.__version__,
                  python=sys.version, machine=platform.platform(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  options=dict(max_actions_per_state=24, up_to_perm=False, do_swaps=False,
                               preprocessing='graph-like only; no teleport_reduce',
                               objective='PyZX basic gate count; NOT NAM gate count',
                               actions='standard local complementation and Pauli pivot only'),
                  states=states, rows=rows, summary=summary)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v if k != 'errors' else len(v) for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
