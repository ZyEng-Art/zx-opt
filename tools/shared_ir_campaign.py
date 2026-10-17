"""Bounded concurrent Quarl/ZX on-policy workers with durable restart exchange."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--quarl-root',type=Path,required=True)
    p.add_argument('--quarl-entry',type=Path,help='Optional isolated overlay run_original_quarl.py')
    p.add_argument('--quarl-python',required=True)
    p.add_argument('--zx-python',required=True)
    p.add_argument('--seconds',type=int,default=120)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--horizon',type=int,default=12)
    p.add_argument('--seed',type=int,default=11)
    a=p.parse_args()
    if min(a.seconds,a.iterations,a.episodes,a.horizon)<1:raise ValueError('positive budgets required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    zxroot=Path(__file__).resolve().parents[1];grs=a.quarl_root.resolve()
    root=out/'input.qasm';root.write_bytes(a.input.read_bytes());database=out/'shared.sqlite'
    env=os.environ.copy();env.update(GRS_DEVICE='cpu',CUDA_VISIBLE_DEVICES='',XPU_VISIBLE_DEVICES='',QUARL_GRAPH_BACKEND='torch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled')
    env['LD_LIBRARY_PATH']=str(grs/'.quartz/lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['PYTHONPATH']=str(grs)+os.pathsep+str(zxroot)+os.pathsep+env.get('PYTHONPATH','')
    commands={
      'quarl':[a.quarl_python,str(a.quarl_entry or grs/'run_original_quarl.py'),'--output',str(out/'quarl'),'--qasm',str(root),'--target-circuit','shared_target','--target','0','--random-init','--iterations',str(a.iterations),'--episodes',str(a.episodes),'--max-eps-len',str(a.horizon),'--seconds',str(a.seconds),'--seed',str(a.seed),'--matcher','cached','--output-full-seq','--full-state-every','1','--full-state-keep','2','--shared-buffer',str(database),'--zx-opt-root',str(zxroot),'--shared-root',str(root)],
      'zx':[a.zx_python,'-m','zxopt.general_search',str(out/'zx'),'--input',str(root),'--jpv','--policy','online','--seconds',str(a.seconds),'--horizon',str(a.horizon),'--rollout','8','--seed',str(a.seed),'--shared-buffer',str(database)]}
    metadata=dict(commands=commands,input_sha256=hashlib.sha256(root.read_bytes()).hexdigest(),cpu_only=True,wall_budget_per_worker=a.seconds,foreign_ppo_replay=False,shared_pool='verified fixed-wire NAM Clifford+T restarts; native latent states retained separately')
    metadata['zx_sources']={str(p.relative_to(zxroot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (zxroot/'zxopt').rglob('*.py')}
    metadata['quarl_overlay_sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(commands['quarl'][1]).parent.glob('*.py')}
    metadata['quarl_entry_sha256']=hashlib.sha256(Path(commands['quarl'][1]).read_bytes()).hexdigest()
    (out/'launch.json').write_text(json.dumps(metadata,indent=2))
    jobs={};logs={};started=time.monotonic()
    for name,cmd in commands.items():
        logs[name]=(out/(name+'.log')).open('w')
        jobs[name]=subprocess.Popen(cmd,cwd=grs if name=='quarl' else zxroot,env=env,stdout=logs[name],stderr=subprocess.STDOUT,start_new_session=True)
    statuses={}
    try:
        for name,job in jobs.items():
            try:statuses[name]=dict(returncode=job.wait(timeout=max(1,a.seconds+180-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(job.pid,signal.SIGTERM)
                try:job.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
                statuses[name]=dict(returncode=job.returncode,status='watchdog_timeout',resume_complete=False)
    finally:
        for log in logs.values():log.close()
        if database.exists():
            source=sqlite3.connect(str(database));dest=sqlite3.connect(str(out/'shared-final.sqlite'))
            source.backup(dest);dest.close()
            events=[dict(id=i,time=t,producer=p,kind=k,key=key,detail=json.loads(d)) for i,t,p,k,key,d in source.execute('SELECT * FROM events ORDER BY id')]
            (out/'exchange.json').write_text(json.dumps(events,indent=2));source.close()
        (out/'status.json').write_text(json.dumps(dict(workers=statuses,seconds=time.monotonic()-started),indent=2))
    for name,filename in [('quarl','result.json'),('zx','report.json')]:
        if not (out/name/filename).exists():
            statuses[name]['status']='missing_completion_receipt'
        elif name=='zx':
            statuses[name]['training_updates']=json.loads((out/name/filename).read_text())['updates']
        else:
            statuses[name]['training_updates']=len((out/name/'progress.jsonl').read_text().splitlines())
    (out/'status.json').write_text(json.dumps(dict(workers=statuses,seconds=time.monotonic()-started),indent=2))
    if any(r['returncode']!=0 or r.get('status') or not r.get('training_updates') for r in statuses.values()):raise SystemExit(1)


if __name__=='__main__':main()
