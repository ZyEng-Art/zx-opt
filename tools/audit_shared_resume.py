"""Replay a resumed ZX segment without pretending it starts at transition zero."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qiskit import qasm2
from zxopt.complete import Diagram
from zxopt.general_kernel import GeneralKernel,Views,HybridAction
from zxopt.general_archive import SearchState,following_state
from zxopt.legacy import environment
from zxopt.verify import qiskit_matrix_zx_order,matrix_error


def audit(folder):
    protocol=json.loads((folder/'protocol.json').read_text());source=Path(protocol['resumed_from']).parent
    reference=qiskit_matrix_zx_order(qasm2.load(str(folder/'input.qasm')))
    views=Views(protocol['num_qubits'],reference);kernel=GeneralKernel(environment(folder/'input.qasm'),jpv=False)
    def graph(key):
        for path in (folder,source):
            p=path/'objects'/(key+'.json.gz')
            if p.exists():
                d=Diagram.from_record(json.loads(gzip.decompress(p.read_bytes())));assert d.digest()==key;return d
        raise ValueError('Missing source-bound graph '+key)
    rows=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()];before=protocol['resume_segment_start_counts'][0]
    assert [r['transition'] for r in rows]==list(range(before+1,before+1+len(rows)))
    checked=0;circuits=set()
    for r in rows:
        if r['status'] not in ('accepted','accepted_unscored'):continue
        parent=graph(r['parent_graph_key']);child,receipts=kernel.step(parent,views.get(parent),HybridAction.from_record(r['action']))
        assert child.digest()==r['graph_key'] and receipts==r['auxiliary'];views.get(child)
        result=None
        if r['status']=='accepted':
            text=(folder/'candidates'/(r['sha256']+'.qasm')).read_text()
            circuit=qasm2.loads(text)
            assert matrix_error(reference,qiskit_matrix_zx_order(circuit))<1e-9
            counts=dict(gates=len(circuit),cx=int(circuit.count_ops().get('cx',0)),depth=circuit.depth())
            assert counts==r['cost'];result=dict(cost=counts);circuits.add(r['sha256'])
        previous=SearchState(parent,r['last_scored_cost'],r['parent_cost'],r['unscored_depth_before'])
        _,reward=following_state(previous,child,result);assert abs(reward-r['reward'])<1e-12;checked+=1
    return dict(status='passed',segment_events=len(rows),accepted_replayed=checked,unique_qasm_checked=len(circuits),source_checkpoint=protocol['resumed_from'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=audit(a.folder);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r))
