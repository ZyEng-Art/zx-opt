"""Synthetic random short-path diagnostic. Not RL or a full optimizer benchmark."""
import json
import hashlib
import random
import sys
import time
from pathlib import Path
import numpy as np
import pyzx as zx
from action_probe import evaluate


def actions(graph):
    result = [('lc', (v,)) for v in graph.vertices() if zx.lcomp_rule.check_lcomp(graph, v)]
    result += [('pivot', graph.edge_st(e)) for e in graph.edges()
               if zx.pivot_rule.check_pivot(graph, *graph.edge_st(e))]
    return result


def main():
    inputs, output = map(Path, sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for source_path in sorted(inputs.glob('*.qasm')):
        original = zx.Circuit.from_qasm(source_path.read_text())
        graph = original.to_graph()
        zx.to_graph_like(graph)
        base, _ = evaluate(graph)
        reference = original.to_matrix()
        for horizon in (1, 2, 4):
            for seed in range(16):
                started = time.perf_counter()
                rng = random.Random(f'{source_path.stem}-{seed}')
                trial = graph.clone()
                row = dict(case=source_path.stem, horizon=horizon, seed=seed,
                           base_gates=len(base.gates), steps=[], status='completed')
                records.append(row)
                for step in range(horizon):
                    available = actions(trial)
                    if not available:
                        break
                    kind, vertices = rng.choice(available)
                    rule = zx.lcomp_rule.lcomp if kind == 'lc' else zx.pivot_rule.pivot
                    if not rule(trial, *vertices):
                        raise AssertionError('ID-stable matched action failed')
                    item = dict(step=step, kind=kind, vertices=vertices)
                    row['steps'].append(item)
                    try:
                        candidate, costs = evaluate(trial)
                        item.update(gates=len(candidate.gates), **costs)
                    except Exception as exc:
                        item.update(error_type=type(exc).__name__, error=str(exc))
                        row['status'] = 'extraction_error'
                        break
                row['seconds'] = time.perf_counter()-started
                if row['status'] == 'completed' and row['steps']:
                    matrix = candidate.to_matrix()
                    overlap = np.vdot(reference, matrix)
                    phase = overlap/abs(overlap) if abs(overlap) else 1
                    row['matrix_error'] = float(np.linalg.norm(matrix-phase*reference)/np.linalg.norm(reference))
                    if row['matrix_error'] >= 1e-9:
                        row['status'] = 'matrix_mismatch'
                    row['final_gates'] = len(candidate.gates)
                    row['best_gates'] = min([row['base_gates']]+[s['gates'] for s in row['steps']])
                    row['temporary_worse_then_improved'] = any(
                        a['gates'] > row['base_gates'] and b['gates'] < row['base_gates']
                        for i, a in enumerate(row['steps']) for b in row['steps'][i+1:])
                # Incremental ledger survives interruption.
                with (output/'rows.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
        print(source_path.stem, flush=True)
    summary = {}
    for horizon in (1, 2, 4):
        group = [r for r in records if r['horizon'] == horizon]
        valid = [r for r in group if r['status'] == 'completed' and 'final_gates' in r]
        summary[str(horizon)] = dict(paths=len(group), completed=len(valid),
            failures=sum(r['status'] != 'completed' for r in group),
            final_improved=sum(r['final_gates'] < r['base_gates'] for r in valid),
            ever_improved=sum(r['best_gates'] < r['base_gates'] for r in valid),
            temporary_worse_then_improved=sum(r['temporary_worse_then_improved'] for r in valid),
            evaluation_calls=sum(len(r['steps']) for r in group),
            seconds=sum(r['seconds'] for r in group))
    (output/'summary.json').write_text(json.dumps(dict(summary=summary,
        pyzx=zx.__version__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='12 synthetic circuits; random paired prefixes; 16 paths per horizon; exact evaluation every step; no cflow/teleport; basic gate cost, not NAM; shared-library matrix consistency only; not independent certification or an RL experiment'), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
