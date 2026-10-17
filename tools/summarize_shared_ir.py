"""Read-only audit and summary of a completed shared-IR pilot and resume."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
from collections import Counter
from qiskit import qasm2
from qiskit.quantum_info import Operator
import torch
import pyzx


def summarize(path):
    root=Operator(qasm2.load(str(path/'input.qasm')))
    db=sqlite3.connect('file:'+str(path/'shared.sqlite')+'?mode=ro',uri=True)
    objects=db.execute('SELECT key,qasm FROM objects').fetchall()
    for key,text in objects:
        assert hashlib.sha256(text.encode()).hexdigest()==key
        assert root.equiv(Operator(qasm2.loads(text)),atol=1e-9,rtol=0),key
    events=[dict(producer=p,kind=k,key=key,detail=json.loads(detail)) for p,k,key,detail in db.execute('SELECT producer,kind,key,detail FROM events ORDER BY id')]
    db.close()
    (path/'exchange-after-resume.json').write_text(json.dumps(events,indent=2))
    workers=json.loads((path/'status.json').read_text());resume=json.loads((path/'resume-status.json').read_text())
    assert all(v['returncode']==0 for v in workers['workers'].values())
    assert all(v['returncode']==0 for v in resume.values())
    zx=json.loads((path/'zx/report.json').read_text());zr=json.loads((path/'zx-resume/report.json').read_text())
    qa=[json.loads(x) for x in (path/'quarl/progress.jsonl').read_text().splitlines()]
    qr=[json.loads(x) for x in (path/'quarl-resume/progress.jsonl').read_text().splitlines()]
    assert qr[0]['iteration']==qa[-1]['iteration']+1
    assert zr['transitions']>zx['transitions'] and zr['updates']>zx['updates']
    for name in ('quarl','quarl-resume'):
        latest=json.loads((path/name/'full_state/latest.json').read_text());snap=Path(latest['path'])
        manifest=json.loads((snap/'manifest.json').read_text())
        for filename,record in manifest['files'].items():assert hashlib.sha256((snap/filename).read_bytes()).hexdigest()==record['sha256']
        assert {'shared-buffer.sqlite','native-trajectory.sqlite'}<=set(manifest['files'])
    result=dict(status='passed',input_sha256=hashlib.sha256((path/'input.qasm').read_bytes()).hexdigest(),
      input_gates=len(qasm2.load(str(path/'input.qasm'))),
      training=dict(quarl_updates=len(qa),zx_updates=zx['updates'],zx_transitions=zx['transitions'],quarl_best=qa[-1]['best_gates'],zx_best=zx['best']['cost']),
      resume=dict(quarl_updates=len(qr),quarl_iterations=[x['iteration'] for x in qr],zx_transitions_before=zx['transitions'],zx_transitions_after=zr['transitions'],zx_updates_after=zr['updates']),
      exchange=dict(events=dict(Counter(x['producer']+':'+x['kind'] for x in events)),unique_qasm_verified=len(objects)),
      native_audit=json.loads((path/'native-audit.json').read_text()),zx_audit=json.loads((path/'zx-audit.json').read_text()),
      environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,pyzx=pyzx.__version__,packages={n:importlib.metadata.version(n) for n in ('qiskit','torch-geometric','gymnasium','networkx','numpy')}),
      limits='Single exposed barenco_tof3 pilot, not a quality/speedup/generalization comparison; CPU only. Numerical full-matrix boundary1e-9. Checkpoints preserve search state but do not guarantee concurrent wall-budget bitwise replay.')
    (path/'summary.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(summarize(a.run),indent=2))
