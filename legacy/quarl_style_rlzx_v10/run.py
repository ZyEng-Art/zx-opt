"""Same-walltime ablation of restart archive and on-target PPO; CPU research pilot."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def objective_score(counts, objective):
    return (counts['cx'], counts['gates']) if objective == 'cx' else (counts['gates'], counts['cx'])


def trial(spec, method, checkpoint, source_dir, config, seed, budget, folder, worker_fn=None):
    # Spawn avoids unsafe forking of an initialized Torch runtime on macOS.
    from runtime import worker
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(config, budget_seconds=budget, adapted_checkpoint=str((folder / 'adapted_latest.pt').resolve()))
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker_fn or worker, args=(child, spec, method, str(checkpoint),
                                              str(source_dir), config, seed))
    setup = time.perf_counter()
    process.start()
    child.close()
    events, updates, refreshes, candidates = [], [], [], {}
    status, stage, started = 'budget_capped', 'setup', None
    report = {}
    try:
        if not parent.poll(90):
            raise TimeoutError('worker setup exceeded 90 seconds')
        ready = parent.recv()
        if ready['kind'] != 'ready':
            raise RuntimeError(ready)
        setup_seconds = time.perf_counter() - setup
        (folder / 'input.qasm').write_text(ready['source'])
        best = {}
        for objective, item in ready['baselines'].items():
            candidates[item['sha256']] = item['qasm']
            best[objective] = {k: v for k, v in item.items() if k != 'qasm'}
            (folder / ('baseline_' + objective + '.qasm')).write_text(item['qasm'])
        started = time.perf_counter()
        parent.send(started + budget)
        while True:
            remaining = started + budget - time.perf_counter()
            if remaining <= 0 or not parent.poll(remaining):
                break
            try:
                item = parent.recv()
            except EOFError:
                status = 'worker_failed'
                break
            elapsed = time.perf_counter() - started
            if elapsed > budget:
                break
            if item['kind'] == 'candidate':
                assert sha(item['qasm']) == item['sha256']
                candidates[item['sha256']] = item.pop('qasm')
                item['received_seconds'] = elapsed
                events.append(item)
                for objective in best:
                    if objective_score(item['cost'], objective) < objective_score(best[objective]['cost'], objective):
                        best[objective] = {k: item[k] for k in ['sha256', 'cost']}
            elif item['kind'] == 'update':
                updates.append(dict(**item, received_seconds=elapsed))
            elif item['kind'] == 'refresh':
                refreshes.append(dict(**item, received_seconds=elapsed))
            elif item['kind'] == 'stage':
                stage = item['stage']
            elif item['kind'] == 'error':
                status = 'worker_error'
                report['error'] = item
                break
            elif item['kind'] == 'finished':
                report['worker_summary'] = item
                status = 'finished'
                break
        cutoff = time.perf_counter() - started
        report.update(spec=spec, method=method, search_seed=seed, budget_seconds=budget,
                      objective=config.get('objective', 'nam'),
                      input_sha256=ready['input_sha256'], baselines=ready['baselines'],
                      baseline_errors=ready['baseline_errors'], incumbent=best,
                      events=events, updates=updates, refreshes=refreshes, status=status, last_stage=stage,
                      setup_seconds=setup_seconds, parent_cutoff_seconds=cutoff)
    finally:
        shutdown = time.perf_counter()
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()
    report['shutdown_seconds'] = time.perf_counter() - shutdown
    adapted = folder / 'adapted_latest.pt'
    if adapted.exists():
        report['adapted_checkpoint'] = dict(path=str(adapted.resolve()),
                                            sha256=hashlib.sha256(adapted.read_bytes()).hexdigest(),
                                            eligibility='Diagnostic model only; final incumbent selection '
                                                        'uses parent-received pre-deadline candidates exclusively.')
    accepted = folder / 'accepted'
    accepted.mkdir()
    for digest, text in candidates.items():
        (accepted / (digest + '.qasm')).write_text(text)
    for objective, item in report['incumbent'].items():
        (folder / ('best_' + objective + '.qasm')).write_text(candidates[item['sha256']])
    (folder / 'trial.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--objective', choices=['nam', 'cx'], default='nam')
    parser.add_argument('--backend', choices=['reference', 'optimized'], default='optimized')
    parser.add_argument('--source-dir', type=Path, default=HERE.parent / 'upstream_source_v1')
    parser.add_argument('--budget', type=float, default=30.)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--seed-count', type=int, default=2)
    parser.add_argument('--seed-start', type=int, default=32000000)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.budget <= 0 or args.reps < 1 or args.seed_count < 1:
        parser.error('positive budget, repetitions and input seed count required')
    from runtime import METHODS
    methods = args.methods or list(METHODS)
    assert set(methods) <= set(METHODS)
    config = dict(objective=args.objective, backend=args.backend, buffer_capacity=64, cache_capacity=256, alpha=1.2,
                  rollout=8 if args.smoke else 128, minibatch=4 if args.smoke else 32,
                  epochs=2 if args.smoke else 4, lr=.0003,
                  collaboration_block=8, model_refresh_seconds=5., mask_capacity=256)
    specs = [dict(case=f'q{q}_d{d}_s{s}', qubits=q, depth=d, seed=s)
             for s in range(args.seed_start, args.seed_start + args.seed_count)
             for q, d in [(3, 30), (5, 70)]]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(scope=__doc__, methods=methods, config=config, specs=specs, objective=args.objective,
                    repetitions=args.reps, budget_seconds=args.budget,
                    checkpoint=str(args.checkpoint),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py', '.cpp', '.so']] +
                                   [args.source_dir / 'zx_env.py', args.source_dir / 'rl_agent.py',
                                    HERE.parent / 'cflow_transfer_probe.py']},
                    smoke=args.smoke, device='cpu',
                    timing='Includes actor, native Env.step, fixed-wire completion at EVERY transition, cache, '
                           'buffer copies/sampling, PPO updates, adapted-model checkpoints and IPC. '
                           'Excludes common reset/baselines, model loading, process startup/shutdown, '
                           'candidate/report file writes and independent proof.',
                    buffer='64 states, exact pickle SHA dedup; admit objective-first score<=trajectory start; '
                           'retain root and best; diverse restart mix .2root/.4best/.4cost-stratified.',
                    reward='(previous fixed-wire completion primary cost - next primary cost)/10; '
                           'plain global-value PPO/GAE, not Quarl HAE. All legal actions including STOP.',
                    limitations='Quarl-inspired mechanism ablation, not Quarl reproduction. '
                                'One predeclared pretrained checkpoint. Cooperative variants use two '
                                'interleaved environments, flat RL-ZX actions and global GAE; not Quarl HAE. '
                                'Tiny local CPU pilot; no claim of stable learning gain.')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, input_hashes = [], {}
    for spec in specs:
        order = [(method, rep) for rep in range(args.reps) for method in methods]
        random.Random(spec['case']).shuffle(order)
        for method, rep in order:
            folder = args.output / spec['case'] / f'{method}_r{rep}'
            result = trial(spec, method, args.checkpoint, args.source_dir, config,
                           33000000 + spec['seed'] * 10 + rep, args.budget, folder)
            assert input_hashes.setdefault(spec['case'], result['input_sha256']) == result['input_sha256']
            row = dict(case=spec['case'], method=method, rep=rep, status=result['status'],
                       transitions=len(result['events']), updates=len(result['updates']),
                       optimizer_steps=sum(u['optimizer_steps'] for u in result['updates']),
                       cx_gain=result['baselines']['cx']['cost']['cx'] - result['incumbent']['cx']['cost']['cx'],
                       nam_gain=result['baselines']['nam']['cost']['gates'] - result['incumbent']['nam']['cost']['gates'],
                       resumed=sum(e['start_distance'] > 0 for e in result['events']),
                       max_distance=max([e['distance'] for e in result['events']] + [0]),
                       last_buffer=result['events'][-1]['buffer'] if result['events'] else None,
                       folder=str(folder.resolve()))
            rows.append(row)
            with (args.output / 'rows.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    # Independent verifier runs AFTER all timed trials and imports no ZX/Torch.
    subprocess.run([sys.executable, str(HERE / 'verify.py'), str(args.output)], check=True)
    proof = json.loads((args.output / 'verification.json').read_text())
    summary = {method: dict(trials=sum(r['method'] == method for r in rows),
                            cx_gain=sum(r['cx_gain'] for r in rows if r['method'] == method),
                            nam_gain=sum(r['nam_gain'] for r in rows if r['method'] == method),
                            transitions=sum(r['transitions'] for r in rows if r['method'] == method),
                            updates=sum(r['updates'] for r in rows if r['method'] == method),
                            resumed=sum(r['resumed'] for r in rows if r['method'] == method))
               for method in methods}
    result = dict(status='passed' if proof['status'] == 'passed' and
                  all(r['status'] not in ('worker_error', 'worker_failed') for r in rows) else 'failed',
                  trials=len(rows), verification=proof, by_method=summary)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise RuntimeError(result)


if __name__ == '__main__':
    main()
