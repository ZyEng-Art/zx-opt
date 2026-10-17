"""Grouped synthetic action-ranking experiment; no optimizer integration or PPO.

Predict immediate extracted/basic gate savings. Standard PyZX, not paper fork.
Held-out synthetic seeds, no project benchmark or expert circuits. Every outcome
is retained. Matrix consistency shares PyZX semantics, not independent proof.
"""
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
import numpy as np
import pyzx as zx
from action_probe import evaluate
from short_path_probe import actions

FEATURES = ['lc', 'pivot', 'qubits', 'nodes', 'edges', 'base_gates',
            'degree_min', 'degree_max', 'degree_sum', 'common_neighbors',
            'union_neighbors', 'neighbor_degree_mean', 'neighbor_degree_max',
            'neighbor_nonclifford', 'neighbor_boundary', 'phase_sin', 'phase_cos',
            'nodes_removed', 'edges_removed', 'neighbor_internal_edges']


def features(graph, trial, kind, vertices, qubits, base_gates):
    neighbors = [set(graph.neighbors(v)) for v in vertices]
    union = set.union(*neighbors)
    common = set.intersection(*neighbors) if len(vertices) > 1 else set()
    degrees = [graph.vertex_degree(v) for v in vertices]
    neighbor_degrees = [graph.vertex_degree(v) for v in union]
    phases = [float(graph.phase(v)) for v in vertices]
    return [int(kind == 'lc'), int(kind == 'pivot'), qubits,
            graph.num_vertices(), graph.num_edges(), base_gates,
            min(degrees), max(degrees), sum(degrees), len(common), len(union),
            float(np.mean(neighbor_degrees)) if union else 0, max(neighbor_degrees, default=0),
            sum(graph.phase(v).denominator > 2 for v in union),
            sum(graph.type(v) == zx.VertexType.BOUNDARY for v in union),
            sum(math.sin(p*math.pi) for p in phases), sum(math.cos(p*math.pi) for p in phases),
            graph.num_vertices()-trial.num_vertices(), graph.num_edges()-trial.num_edges(),
            sum(graph.connected(a, b) for a in union for b in union if a < b)]


def fit_model(rows, kind, alpha, seed):
    x = np.asarray([r['features'] for r in rows], dtype=float)
    y = np.asarray([r['gates_saved'] for r in rows], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (x-mean)/scale
    weight = bias = None
    if kind == 'random_features':
        rng = np.random.default_rng(seed)
        weight = rng.normal(size=(z.shape[1], 128))/np.sqrt(z.shape[1])
        bias = rng.normal(size=128)
        z = np.concatenate([z, np.tanh(z@weight+bias)], axis=1)
    z = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(z.shape[1])*alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(z.T@z+penalty, z.T@y)
    return dict(kind=kind, alpha=alpha, seed=seed, mean=mean.tolist(), scale=scale.tolist(),
                weight=None if weight is None else weight.tolist(),
                bias=None if bias is None else bias.tolist(), coefficients=coefficients.tolist())


def predict(model, rows):
    x = np.asarray([r['features'] for r in rows], dtype=float)
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    if model['weight'] is not None:
        z = np.concatenate([z, np.tanh(z@np.asarray(model['weight'])+np.asarray(model['bias']))], axis=1)
    return np.column_stack([np.ones(len(z)), z])@np.asarray(model['coefficients'])


def assess(groups, score_fn):
    scores, per_state = {}, []
    for case, rows in groups.items():
        order = np.argsort(-np.asarray(score_fn(rows)), kind='stable')
        gains = np.asarray([max(0, r['gates_saved']) for r in rows])
        item = dict(case=case, oracle_gain=int(max(gains)), candidate_count=len(rows))
        for budget in (1, 2, 4, 8):
            item[str(budget)] = int(max(gains[order[:budget]]))
        per_state.append(item)
    for budget in (1, 2, 4, 8):
        gains = [s[str(budget)] for s in per_state]
        scores[str(budget)] = dict(gates_saved=sum(gains), mean_saved=float(np.mean(gains)),
                                  states_improved=sum(g > 0 for g in gains))
    return dict(summary=scores, per_state=per_state)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = dict(scope=__doc__, feature_names=FEATURES, states_per_shape=60,
        split_indices=dict(train=[0, 40], validation=[40, 50], test=[50, 60]),
        seed_origin=26015000, shapes=[[3, 30], [3, 60], [5, 30], [5, 60]],
        candidate_cap=24, selection='fixed seeded random subset before evaluation',
        model_selection='highest validation gate savings at k=4; test used once after selection',
        model_grid=dict(kinds=['linear','random_features'], alphas=[1,10,100], seeds=[11,29,47]),
        input_metric='PyZX basic gates, not NAM; lc and Pauli pivot only; graph-like preprocessing',
        up_to_perm=False, do_swaps=False, hostname=platform.node(),
        python=sys.version, pyzx=zx.__version__, numpy=np.__version__,
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path(__file__).with_name('action_probe.py'),Path(__file__).with_name('short_path_probe.py')]})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    states, rows = [], []
    for qubits, depth in protocol['shapes']:
        for index in range(60):
            split = 'train' if index < 40 else 'validation' if index < 50 else 'test'
            seed = protocol['seed_origin']+index
            case = f'q{qubits}_d{depth}_s{seed}'
            graph = zx.generate.cliffordT(qubits, depth, p_t=.25, p_s=.25, p_hsh=.25, seed=seed)
            source = zx.Circuit.from_graph(graph).to_basic_gates()
            qasm = source.to_qasm()
            (out/(case+'.qasm')).write_text(qasm)
            zx.to_graph_like(graph)
            base, _ = evaluate(graph)
            reference = source.to_matrix()
            matching_start = time.perf_counter()
            choices = actions(graph)
            matching_seconds = time.perf_counter()-matching_start
            random.Random(seed).shuffle(choices)
            state = dict(case=case, split=split, qubits=qubits, generator_depth=depth, seed=seed,
                source_sha256=hashlib.sha256(qasm.encode()).hexdigest(),
                available_actions=len(choices), selected_actions=min(24,len(choices)),
                base_gates=len(base.gates), matching_seconds=matching_seconds)
            states.append(state)
            for kind, vertices in choices[:24]:
                row = dict(case=case, split=split, kind=kind, vertices=vertices)
                rows.append(row)
                try:
                    feature_start=time.perf_counter()
                    trial = graph.clone()
                    rule = zx.lcomp_rule.lcomp if kind == 'lc' else zx.pivot_rule.pivot
                    if not rule(trial,*vertices):
                        raise AssertionError('ID-stable matched action failed')
                    row['features'] = features(graph,trial,kind,vertices,qubits,len(base.gates))
                    row['feature_seconds'] = time.perf_counter()-feature_start
                    candidate, costs = evaluate(trial)
                    matrix = candidate.to_matrix()
                    overlap = np.vdot(reference,matrix)
                    phase = overlap/abs(overlap) if abs(overlap) else 1
                    error = float(np.linalg.norm(matrix-phase*reference)/np.linalg.norm(reference))
                    row.update(status='consistent' if error < 1e-9 else 'matrix_mismatch',
                               matrix_error=error,gates_saved=len(base.gates)-len(candidate.gates),**costs)
                except Exception as exc:
                    row.update(status='error',error_type=type(exc).__name__,error=str(exc))
                with (out/'rows.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
            if index % 10 == 9:
                (out/'progress.json').write_text(json.dumps(dict(states=len(states),rows=len(rows),seconds=time.time()-started))+'\n')
                print(case,len(rows),flush=True)
    (out/'states.json').write_text(json.dumps(states,indent=2)+'\n')
    failed = [r for r in rows if r['status']!='consistent']
    # Keep failures in the ledger; do not silently remove candidates from ranking.
    if failed:
        (out/'failed.json').write_text(json.dumps(failed,indent=2)+'\n')
        raise RuntimeError('Candidate errors/mismatches: ranking analysis halted')
    groups = {split:{s['case']:[r for r in rows if r['case']==s['case']] for s in states
                     if s['split']==split and s['selected_actions']>0} for split in ['train','validation','test']}
    train = [r for r in rows if r['split']=='train']
    models, trials = [], []
    fit_start=time.perf_counter()
    for kind in protocol['model_grid']['kinds']:
        for alpha in protocol['model_grid']['alphas']:
            for seed in ([11] if kind=='linear' else protocol['model_grid']['seeds']):
                model=fit_model(train,kind,alpha,seed)
                validation=assess(groups['validation'],lambda rs:predict(model,rs))
                models.append(model)
                trials.append(dict(kind=kind,alpha=alpha,seed=seed,validation=validation['summary']))
    selected=max(range(len(models)),key=lambda i:trials[i]['validation']['4']['gates_saved'])
    model=models[selected]
    (out/'model.json').write_text(json.dumps(model)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(trials=trials,selected=selected),indent=2)+'\n')
    model_fit_seconds=time.perf_counter()-fit_start
    rule_means={kind:float(np.mean([r['gates_saved'] for r in train if r['kind']==kind])) for kind in ['lc','pivot']}
    prediction_start=time.perf_counter()
    learned=assess(groups['test'],lambda rs:predict(model,rs))
    prediction_seconds=time.perf_counter()-prediction_start
    controls=dict(learned=learned,
        edges_removed=assess(groups['test'],lambda rs:[r['features'][18] for r in rs]),
        rule_mean=assess(groups['test'],lambda rs:[rule_means[r['kind']] for r in rs]),
        untrained_random_features=assess(groups['test'],lambda rs:
            np.tanh(((np.asarray([r['features'] for r in rs])-np.asarray(model['mean']))/np.asarray(model['scale']))@
                    np.random.default_rng(981).normal(size=len(FEATURES)))))
    random_runs=[assess(groups['test'],lambda rs,seed=seed:np.random.default_rng(
                 int(hashlib.sha256((rs[0]['case']+str(seed)).encode()).hexdigest()[:8],16)).random(len(rs)))
                 for seed in range(100)]
    random_per_case={case:{str(k):float(np.mean([next(s for s in rr['per_state'] if s['case']==case)[str(k)]
                      for rr in random_runs])) for k in (1,2,4,8)} for case in groups['test']}
    paired={}
    for k in (1,2,4,8):
        differences=np.asarray([s[str(k)]-random_per_case[s['case']][str(k)] for s in learned['per_state']])
        # Resample seed groups, preserving four related shape instances together.
        seeds=sorted({s['seed'] for s in states if s['split']=='test'})
        per_seed=np.asarray([np.mean([s[str(k)]-random_per_case[s['case']][str(k)] for s in learned['per_state']
                                    if s['case'].endswith('_s'+str(seed))]) for seed in seeds])
        rng=np.random.default_rng(712)
        boot=np.mean(rng.choice(per_seed,size=(2000,len(per_seed)),replace=True),axis=1)
        paired[str(k)]=dict(mean_gate_advantage=float(differences.mean()),
                           seed_group_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    report=dict(status='completed',states=len(states),rows=len(rows),failures=0,
        split_states={s:len(g) for s,g in groups.items()},selected_model=trials[selected],
        controls=controls,random_expected={str(k):dict(
            gates_saved=float(np.mean([rr['summary'][str(k)]['gates_saved'] for rr in random_runs])),
            states_improved=float(np.mean([rr['summary'][str(k)]['states_improved'] for rr in random_runs])))
            for k in (1,2,4,8)},learned_minus_random=paired,
        oracle_total_gain=sum(s['oracle_gain'] for s in learned['per_state']),
        model_fit_and_validation_seconds=model_fit_seconds,test_scoring_seconds=prediction_seconds,
        all_feature_seconds=sum(r['feature_seconds'] for r in rows),
        all_exact_evaluation_seconds=sum(r['extraction_seconds']+r['cleanup_seconds'] for r in rows),
        total_seconds=time.time()-started,
        limitations='Offline candidate-ranking replay. Not measured online wall-time benefit. Candidate subset capped24. Immediate basic-gate target only. Shared-library consistency, not independent proof. Same-generator held-out seeds, not new-family generalization. No cflow baseline, NAM conversion, PPO or accelerator use.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='controls'},indent=2))


if __name__=='__main__':
    main()
