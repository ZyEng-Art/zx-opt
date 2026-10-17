"""Independent full-matrix checks, including every archived intermediate candidate."""
import hashlib
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'quarl_style_rlzx_v10'))
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator


def error(source, target):
    if source.shape != target.shape:
        return float('inf')
    inner = np.vdot(source, target)
    phase = inner / abs(inner) if abs(inner) > 1e-15 else 1.
    return float(np.linalg.norm(target - phase * source) / np.linalg.norm(source))


def controls():
    c = QuantumCircuit(2)
    c.h(0)
    c.cx(0, 1)
    reference = Operator(c).data
    wrong = c.copy()
    wrong.x(0)
    swapped = QuantumCircuit(2)
    swapped.h(1)
    swapped.cx(1, 0)
    assert error(reference, reference * np.exp(.37j)) < 1e-9
    assert error(reference, Operator(wrong).data) > 1e-9
    assert error(reference, Operator(swapped).data) > 1e-9
    return dict(global_phase_accepted=True, wrong_x_rejected=True, wrong_wires_rejected=True)


def main():
    root = Path(sys.argv[1])
    check = controls()
    assert 'pyzx' not in sys.modules and 'torch' not in sys.modules
    memo, records = {}, [];bridges=[]
    for folder in sorted(root.glob('*/*')):
        if not (folder / 'trial.json').exists():
            continue
        record = json.loads((folder / 'trial.json').read_text())
        source_text = (folder / 'input.qasm').read_text()
        source_hash = hashlib.sha256(source_text.encode()).hexdigest()
        assert source_hash == record['input_sha256']
        source = Operator(qasm2.loads(source_text)).data
        supplied=Path(record['spec']['input_path']).read_text();assert hashlib.sha256(supplied.encode()).hexdigest()==record['spec']['supplied_input_sha256']
        bridge_error=error(Operator(qasm2.loads(supplied)).data,source);assert bridge_error<1e-9
        bridges.append(dict(folder=str(folder),error=bridge_error))
        for path in sorted((folder / 'accepted').glob('*.qasm')):
            text = path.read_text()
            target_hash = hashlib.sha256(text.encode()).hexdigest()
            assert target_hash == path.stem
            target = qasm2.loads(text)
            ops = target.count_ops()
            assert set(ops) <= {'h', 'rz', 'x', 'cx'}
            pair = (source_hash, target_hash)
            if pair not in memo:
                memo[pair] = error(source, Operator(target).data)
            records.append(dict(path=str(path), error=memo[pair], passed=memo[pair] < 1e-9))
        # Replay costs, exact reward and monotonic incumbent selection.
        known = {p.stem: qasm2.loads(p.read_text()) for p in (folder / 'accepted').glob('*.qasm')}
        for event in record['events']:
            c = known[event['sha256']]
            assert event['cost'] == dict(cx=int(c.count_ops().get('cx', 0)), gates=len(c), depth=int(c.depth()))
            assert event['received_seconds'] <= record['budget_seconds']
            objective = record['objective']
            assert event['objective'] == objective
            metric = 'gates' if objective == 'nam' else 'cx'
            assert event['record_best']==min(event['previous_record'],event['cost']['gates'])
            expected=(event['previous_cost'][metric]-event['cost'][metric])/10 if event['reward_mode']=='current'else(event['previous_record']-event['record_best'])/10
            assert abs(event['reward']-expected)<1e-12
            assert not event['legal_nonstop']or event['action']!=9003001
            scale=max(1,event['root_gates']);ctx=[(event['cost']['gates']-event['start_gates'])/scale,(event['record_best']-event['start_gates'])/scale,event['episode_len']/35,event['start_gates']/scale]
            assert np.allclose(event['context'][0],ctx,atol=1e-7)
            assert event['buffer']['objective'] == objective
        for objective, item in record['incumbent'].items():
            c = known[item['sha256']]
            assert item['cost']['cx'] == int(c.count_ops().get('cx', 0)) and item['cost']['gates'] == len(c)
            assert (folder / ('best_' + objective + '.qasm')).read_text() == qasm2.dumps(c) + '\n'
            scores = [x['cost'] for x in record['events']] + [record['baselines'][objective]['cost']]
            key = (lambda x: (x['cx'], x['gates'])) if objective == 'cx' else (lambda x: (x['gates'], x['cx']))
            assert key(item['cost']) == min(map(key, scores))
    result = dict(status='passed' if records and all(r['passed'] for r in records) else 'failed',
                  original_input_bridge=dict(status='passed',checks=len(bridges),max_error=max(b['error']for b in bridges)),controls=check, checks=len(records), unique_pairs=len(memo),
                  max_error=max(memo.values(), default=None),
                  pyzx_loaded=False, torch_loaded=False, records=records)
    (root / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('independent equivalence verification failed')


if __name__ == '__main__':
    main()
