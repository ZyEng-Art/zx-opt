import subprocess,pathlib,json,datetime
base=pathlib.Path('/tmp/dengzengyong_rlzx_policy_device_node23_20261018')
meta=json.loads((base/'ownership.json').read_text());cid=meta['container_id']
def save(phase,**kw):
 r=dict(phase=phase,time=datetime.datetime.now().astimezone().isoformat(),**meta,**kw)
 (base/'controller.json').write_text(json.dumps(r,indent=2)+'\n')
def inspect():
 d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
 assert d['Name'].lstrip('/')==meta['container_name'] and d['State']['StartedAt']==meta['started_at']
 assert d['Config']['Labels']['task']=='rlzx-policy-device-20261018'
 return d
try:
 inspect();save('benchmark')
 args=['docker','exec','-w','/workspace/task/source-attempt2','-e','PYTHONPATH=/workspace/pin:/workspace/offline','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',cid,'/root/miniconda/envs/python310_torch29_cuda/bin/python','tools/benchmark_anchor_device.py','/workspace/task/benchmark.json','--device','kunlun','--batch-sizes','1','8','32','--repeats','5','--warmup','2']
 with (base/'benchmark.log').open('x') as stream:subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
 result=json.loads((base/'benchmark.json').read_text());assert result['status']=='passed';save('passed')
except BaseException as exc:
 save('failed',error=repr(exc));raise
finally:
 if inspect()['State']['Running']:subprocess.run(['docker','stop','-t','20',cid],check=True)
 with (base/'resource-after.log').open('w') as stream:subprocess.run(['xpu-smi'],stdout=stream,stderr=subprocess.STDOUT,check=True)
