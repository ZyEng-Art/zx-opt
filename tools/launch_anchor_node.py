"""Launch a sealed CPU comparison through an existing persistent SSH socket."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--node", type=int, choices=[22, 23], required=True)
    p.add_argument("--tag", default="anchor_prior")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--circuits", nargs="+", default=["barenco_tof_3"])
    p.add_argument("--seeds", nargs="+", default=["11", "29", "47", "71", "101"])
    p.add_argument("--modes", nargs="+", default=["global32", "anchor8", "anchor8_nodes",
                                                  "anchor8_lr1e3", "anchor8_nodes_lr1e3"])
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9_]{1,32}", a.tag):
        raise ValueError("Use a short task tag")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Commit the source before launching a sealed comparison")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    bundle = out/"source.bundle"
    subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(bundle), "HEAD"], check=True)
    spec = dict(node=a.node, tag=a.tag, source_commit=revision, seconds=a.seconds,
                workers=a.workers, circuits=a.circuits, seeds=a.seeds, modes=a.modes,
                base="/tmp/dengzengyong_rlzx_"+a.tag+"_node"+str(a.node)+"_20261018",
                bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
    host = "dengzengyong@wxtky02-p800-backup-8nic-vd-node"+str(a.node)+".wxtky02"
    socket = "/tmp/quarl-nam-node"+str(a.node)+"-live"
    ssh = ["ssh", "-S", socket, "-o", "ProxyCommand=false", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, "python3 -"]

    def remote(script):
        result = subprocess.run(ssh, input=script, text=True, capture_output=True, timeout=60)
        with (out/"setup.log").open("a") as stream:
            stream.write(result.stdout+"\n"+result.stderr)
        return result

    prepared = remote("from pathlib import Path\np=Path("+repr(spec["base"])+")\np.mkdir(exist_ok=False)\nprint('prepared')\n")
    if "prepared" not in prepared.stdout:
        raise RuntimeError("Remote directory creation failed; inspect setup.log without overwriting another task")
    subprocess.run(["scp", "-o", "ControlPath="+socket, "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
                    str(bundle), host+":"+spec["base"]+"/source.bundle"], check=True, timeout=120)
    script = "spec=json.loads("+repr(json.dumps(spec))+")\n"
    script = "import json,subprocess,pathlib,hashlib\n"+script+r'''
base=pathlib.Path(spec['base'])
if hashlib.sha256((base/'source.bundle').read_bytes()).hexdigest()!=spec['bundle_sha256']:
    raise RuntimeError('Bundle transport checksum mismatch')
run=lambda cmd: subprocess.check_output(cmd,text=True).strip()
run(['git','clone',str(base/'source.bundle'),str(base/'source')])
revision=run(['git','-C',str(base/'source'),'rev-parse','HEAD'])
if revision!=spec['source_commit']: raise RuntimeError('Source revision mismatch')
old=pathlib.Path('/tmp/dengzengyong_rlzx_nam_node'+str(spec['node'])+'_20261017')
image=json.loads(run(['docker','inspect','dengzengyong_rlzx_nam_node'+str(spec['node'])+'_20261017']))[0]['Image']
name=base.name
cid=run(['docker','create','--name',name,'--label','owner=dengzengyong','--label','task=rlzx-anchor-20261018',
         '--cpus','16','--memory','48g','--shm-size','4g','--pids-limit','4096',
         '-v',str(base)+':/workspace/task','-v',str(old/'pyzx-pin')+':/workspace/pin:ro',
         '-v',str(old/'offline-deps')+':/workspace/offline:ro','--entrypoint','/bin/bash',image,'-lc','sleep infinity'])
run(['docker','start',cid])
data=json.loads(run(['docker','inspect',cid]))[0]
meta=dict(container_name=name,container_id=cid,started_at=data['State']['StartedAt'],image=image,
          source_commit=revision,cpus=16,memory_gib=48,workers=spec['workers'],gpu=False,node=spec['node'])
(base/'ownership.json').write_text(json.dumps(meta,indent=2)+'\n')
run(['docker','exec',cid,'git','config','--global','--add','safe.directory','/workspace/task/source'])
command=['python3',str(base/'source/tools/node_anchor_campaign.py'),str(base),'--container',name,
         '--commit',revision,'--seconds',str(spec['seconds']),'--workers',str(spec['workers']),
         '--circuits',*spec['circuits'],'--seeds',*spec['seeds'],'--modes',*spec['modes']]
with (base/'controller.log').open('x') as stream:
    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
meta['controller_pid']=proc.pid
(base/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta))
'''
    result = remote(script)
    lines = [s for s in result.stdout.splitlines() if s.startswith("{") and "controller_pid" in s]
    if not lines:
        raise RuntimeError("No verified controller launch; inspect retained remote setup artifacts")
    launch = dict(**spec, runtime=json.loads(lines[-1]))
    (out/"launch.json").write_text(json.dumps(launch, indent=2)+"\n")
    with (out/"collector.log").open("x") as stream:
        proc = subprocess.Popen([sys.executable, str(ROOT/"tools/mirror_anchor_nodes.py"), str(out/"mirror"),
                                 "--task-prefix", "rlzx_"+a.tag, "--nodes", str(a.node), "--interval", "300"],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (out/"collector-launch.json").write_text(json.dumps(dict(pid=proc.pid, interval=300), indent=2)+"\n")
    print(json.dumps(launch))


if __name__ == "__main__":
    main()
