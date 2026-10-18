"""Run on the node host: inspect or recover a stopped owned native NAM study.

Default is read-only. Execution requires its original controller to be absent and
its owned container stopped. Original segments/checkpoints are never overwritten.
This is recovery, not a final replay-audit certificate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PYTHON = '/root/miniconda/envs/python310_torch29_cuda/bin/python'
MOUNT = Path('/workspace/task')

# Evaluated inside the original container/checkout, so source-bound checkpoints
# continue to match. This helper does not edit the frozen zxopt package.
RUN_SEGMENT = r'''
import argparse, hashlib, json, sys, time
from pathlib import Path
from zxopt.general_search import parser, run
import torch
source, destination, exclude = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]=='1'
protocol=json.loads((source/'protocol.json').read_text())
checkpoint=source/'search-state.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
args=parser().parse_args([str(destination)])
for key,value in protocol['config'].items():
    if not hasattr(args,key): raise ValueError('Unknown saved option: '+key)
    if key in ('output','input','resume_state','resume_model','shared_buffer','native_flow_module') and value is not None:
        value=Path(value)
    setattr(args,key,value)
args.output=destination; args.input=source/'input.qasm'; args.resume_model=None
args.resume_state=checkpoint; args.seconds=0; args.max_steps=None
assert args.rule_profile=='native' and args.policy=='online'
receipt=dict(original_checkpoint=str(checkpoint),original_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             exclude_downtime=exclude,source_segment=str(source))
if exclude:
    # mtime belongs to the atomic checkpoint itself; checkpoint.json can lag a
    # replaced .pt if the process dies between the two atomic replacements.
    elapsed=max(0.,time.time()-checkpoint.stat().st_mtime)
    saved['patience']['last_improvement_unix']+=elapsed
    adjusted=source/('recovery-'+destination.name+'.pt')
    assert not adjusted.exists()
    torch.save(saved,adjusted)
    args.resume_state=adjusted
    receipt.update(excluded_seconds=elapsed,adjusted_checkpoint=str(adjusted),
                   adjusted_sha256=hashlib.sha256(adjusted.read_bytes()).hexdigest())
receipt['start_counts']=saved['counts']; receipt['remaining_at_resume']=max(0.,args.no_improvement_seconds-(time.time()-saved['patience']['last_improvement_unix']))
# Persist before run: fatal initialization failures must retain recovery lineage.
(source/('recovery-'+destination.name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
run(args)
'''


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--exclude-downtime', action='store_true', help='Explicitly exclude time since each last atomic checkpoint; records the adjustment')
    p.add_argument('--workers', type=int, default=13)
    a = p.parse_args()
    if a.workers < 1:
        raise ValueError('workers must be positive')
    base = a.base.absolute()
    owner = json.loads((base/'ownership.json').read_text())
    launch = json.loads((base/'launch.json').read_text())
    campaign = base/'source/runs/comparison'
    plan = json.loads((campaign/'campaign.json').read_text())
    assert len(plan['plan']) == 13 and plan['seconds'] == 0 and plan['no_improvement_seconds'] == 10800
    rows = []
    for spec in plan['plan']:
        assert spec['mode'] == 'rules_native' and spec['policy'] == 'online'
        original = campaign/spec['folder']
        segments = [original]
        recovery = base/'source/runs/recovery'
        if recovery.exists():
            segments += sorted(recovery.glob('*/'+spec['folder']))
        finished = []
        for segment in segments:
            f = segment/'report.json'
            if f.exists():
                r = json.loads(f.read_text())
                if r['status'] == 'no_improvement_timeout':
                    assert r['finished_unix']-r['patience']['last_improvement_unix'] >= 10800
                    finished.append(segment)
        available = [s for s in segments if (s/'search-state.pt').exists()]
        latest = max(available, key=lambda s: (s/'search-state.pt').stat().st_mtime) if available else None
        row = dict(folder=spec['folder'],status='completed' if finished else 'recoverable' if latest else 'missing_checkpoint',
                   source_segment=None if latest is None else str(latest.relative_to(base)))
        if latest:
            cp = latest/'search-state.pt'
            row.update(checkpoint_bytes=cp.stat().st_size,checkpoint_mtime=cp.stat().st_mtime)
        rows.append(row)
    info = json.loads(subprocess.check_output(['docker','inspect',owner['container_id']],text=True))[0]
    assert info['Id']==owner['container_id'] and info['Name'].lstrip('/')==owner['container_name']
    assert info['Config']['Labels']['owner']=='dengzengyong' and info['Config']['Labels']['task']=='rlzx-anchor-20261018'
    controller = Path('/proc')/str(launch['controller_pid'])/'cmdline'
    controller_alive = controller.exists() and b'node_anchor_campaign.py' in controller.read_bytes()
    summary = dict(source_commit=owner['source_commit'],container_running=info['State']['Running'],
                   original_controller_alive=controller_alive,exclude_downtime=a.exclude_downtime,
                   records=rows,execute=a.execute)
    print(json.dumps(summary,indent=2),flush=True)
    if not a.execute:
        return
    if info['State']['Running'] or controller_alive:
        raise RuntimeError('Refusing recovery while original container or controller is active; no processes stopped')
    if any(r['status']=='missing_checkpoint' for r in rows):
        raise RuntimeError('Missing checkpoint: refuses silently restarting a target from scratch')
    jobs = [r for r in rows if r['status']=='recoverable']
    if not jobs:
        return
    # Exclusive recovery lock: a crash leaves a visible lock requiring inspection.
    lock=base/'recovery.lock'
    with lock.open('x') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time()))+'\n')
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    dest=base/'source/runs/recovery'/stamp;dest.mkdir(parents=True,exist_ok=False)
    (dest/'plan.json').write_text(json.dumps(summary,indent=2)+'\n')
    try:
        subprocess.run(['docker','start',owner['container_id']],check=True)
        def one(row):
            source=MOUNT/row['source_segment']
            target=MOUNT/'source/runs/recovery'/stamp/row['folder']
            command=['docker','exec','-w','/workspace/task/source','-e','PYTHONPATH=/workspace/pin:/workspace/offline',
                     '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
                     owner['container_id'],PYTHON,'-c',RUN_SEGMENT,str(source),str(target),'1' if a.exclude_downtime else '0']
            log=dest/(row['folder'].replace('/','__')+'.log')
            with log.open('x') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            return dict(folder=row['folder'],returncode=result.returncode,checkpoint_source=row['source_segment'])
        records=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(one,row) for row in jobs]):
                records.append(future.result())
                (dest/'recovery-status.json').write_text(json.dumps(dict(records=records,expected=len(jobs),
                    audit_status='pending independent resumed-segment audit; not certified'),indent=2)+'\n')
        if any(r['returncode'] for r in records):
            raise RuntimeError('Failed recovery segments retained; inspect logs')
    finally:
        subprocess.run(['docker','stop','-t','30',owner['container_id']],check=True)
        lock.unlink()


if __name__=='__main__':
    main()
