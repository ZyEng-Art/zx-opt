"""Read live owned-process handles, study progress and the preserved NAM jobs.

This observer never starts, restarts, stops or edits remote jobs. A transient
SSH failure is recorded as an observation failure, not as a terminal job.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time


REMOTE = r'''
from pathlib import Path
import datetime,json,subprocess,os
config=json.loads(CONFIG)
node=config['node']
result={'node':node,'time':datetime.datetime.now().astimezone().isoformat(),'studies':[]}
sv=os.statvfs('/tmp');result['disk']={'free_bytes':sv.f_bavail*sv.f_frsize,'free_inodes':sv.f_favail}
for spec in config['studies']:
 p=Path(spec['base']);r={'tag':spec['tag'],'base':str(p),'expected_source':spec['source']}
 if not (p/'launch.json').exists():r['observation_error']='Launch record missing';result['studies'].append(r);continue
 launch=json.loads((p/'launch.json').read_text());pid=launch['controller_pid'];cid=launch['container_id']
 try:cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
 except FileNotFoundError:cmd=''
 r.update(controller_pid=pid,controller_live=str(p) in cmd,container_id=cid)
 d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True,timeout=8))[0]
 r.update(container_running=d['State']['Running'],same_incarnation=d['State']['StartedAt']==launch['started_at'])
 if d['State']['Running']:
  top=subprocess.check_output(['docker','top',cid,'-eo','pid,args'],text=True,timeout=8)
  r['search_processes']=sum('zxopt.general_search' in s for s in top.splitlines())
  r['audit_processes']=sum('audit_general.py' in s for s in top.splitlines())
 if (p/'controller.json').exists():r['controller']=json.loads((p/'controller.json').read_text())
 q=p/'source/runs/comparison'
 r['case_failure_files']=sum(1 for _ in q.glob('*/*/failure.json'))
 for name in ['summary.json','audit/progress.json','finished.json']:
  if (q/name).exists():
   row=json.loads((q/name).read_text());r[name]={k:v for k,v in row.items() if k not in ['rows','records']}
   if name=='summary.json':r['training_failures']=sum(x.get('status')=='failed' for x in row.get('rows',[]))
 if r.get('controller',{}).get('phase')=='failed' and (p/'controller.log').exists():
  r['failure_tail']=(p/'controller.log').read_text()[-3000:]
 result['studies'].append(r)
old=Path('/tmp/dengzengyong_rlzx_nam_node'+str(node)+'_20261017')
name='dengzengyong_rlzx_nam_node'+str(node)+'_20261017'
d=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=8))[0]
jobs=list((old/'zx-opt-f5874e497ce1/runs/nam-study').glob('*/jpv_online_s11'))
r={'container_running':d['State']['Running'],'jobs':len(jobs),'finished':sum((p/'report.json').exists() for p in jobs),'failed':sum((p/'failure.json').exists() for p in jobs)}
if d['State']['Running']:
 top=subprocess.check_output(['docker','top',d['Id'],'-eo','pid,args'],text=True,timeout=8)
 r['search_processes']=sum('zxopt.general_search' in s for s in top.splitlines())
result['preserved_nam']=r
print(json.dumps(result))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("state", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args(); state = json.loads(a.state.read_text())

    def one(node):
        config = dict(node=node, studies=[s for s in state["studies"] if s["node"] == node])
        source = REMOTE.replace("CONFIG", repr(json.dumps(config)))
        ssh = ["ssh", "-S", state["ssh_sockets"][str(node)], "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=8", "dengzengyong@wxtky02-p800-backup-8nic-vd-node"+str(node)+".wxtky02", "python3 -"]
        try:
            result = subprocess.run(ssh, input=source, text=True, capture_output=True, timeout=45)
            return json.loads(result.stdout)
        except Exception as exc:
            return dict(node=node, observation_error=repr(exc), terminal=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, [22, 23]))
    result = dict(time_unix=time.time(), nodes=records)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        temp = a.output.with_suffix(".tmp"); temp.write_text(json.dumps(result, indent=2)+"\n"); temp.replace(a.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
