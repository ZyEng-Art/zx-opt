"""Launch a sealed CPU comparison through an existing persistent SSH socket."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--node", type=int, choices=[22, 23], required=True)
    p.add_argument("--tag", default="anchor_prior")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--no-improvement-seconds", type=float, default=0.)
    p.add_argument("--policies", nargs="+", choices=["frozen", "online"], default=["frozen", "online"])
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--cpus", type=int, default=16)
    p.add_argument("--memory-gib", type=int, default=48)
    p.add_argument("--experiment", choices=["search", "policy-probe", "model-probe"], default="search")
    p.add_argument("--model-manifest", type=Path, help="Flat hash-bound demonstration model package")
    p.add_argument("--probe-source-tag", help="Completed owned study on the same host; mounted read-only")
    p.add_argument("--training-seeds", nargs="+", default=["137", "163", "197"])
    p.add_argument("--audit-workers", type=int, default=2)
    p.add_argument("--audit-native-replay", choices=["reference", "rooted"], default="reference")
    p.add_argument("--audit-crosscheck-native-every", type=int, default=0)
    p.add_argument("--build-native-flow", action="store_true")
    p.add_argument("--save-improvement-models", action="store_true")
    p.add_argument("--metadata-only-mirror", action="store_true")
    p.add_argument("--storage-root", help="Remote durable directory; task keeps a /tmp symlink for existing collectors")
    p.add_argument("--remote-archive-only", action="store_true")
    p.add_argument("--circuits", nargs="+", default=["barenco_tof_3"])
    p.add_argument("--seeds", nargs="+", default=["11", "29", "47", "71", "101"])
    p.add_argument("--modes", nargs="+", default=["global32", "anchor8", "anchor8_nodes",
                                                  "anchor8_lr1e3", "anchor8_nodes_lr1e3"])
    a = p.parse_args()
    if a.seconds < 0 or a.no_improvement_seconds < 0 or (a.seconds == 0 and a.no_improvement_seconds == 0):
        raise ValueError("Provide a positive wall budget or positive per-target patience")
    if a.audit_crosscheck_native_every < 0 or (a.audit_crosscheck_native_every and a.audit_native_replay != "rooted"):
        raise ValueError("Native crosschecks require rooted replay and a nonnegative interval")
    if not re.fullmatch(r"[a-z0-9_]{1,32}", a.tag):
        raise ValueError("Use a short task tag")
    if a.experiment == "policy-probe" and (not a.probe_source_tag or
            not re.fullmatch(r"[a-z0-9_]{1,32}", a.probe_source_tag)):
        raise ValueError("Policy probes require a valid completed source-study tag")
    if a.workers < 1 or a.cpus < 1 or a.memory_gib < 1:
        raise ValueError("Resource limits must be positive")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Commit the source before launching a sealed comparison")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    model_bundle = None; model_files = []; development_models = False
    if a.experiment == "model-probe":
        if a.model_manifest is None:
            raise ValueError("Model probes require a local models.json manifest")
        manifest = json.loads(a.model_manifest.read_text()); development_models = manifest["development_source"]
        model_bundle = out/"models.tar.gz"
        with tarfile.open(model_bundle, "w:gz", compresslevel=1) as tar:
            tar.add(a.model_manifest, arcname="models.json"); model_files.append("models.json")
            for model in manifest["models"]:
                name = model["model"]
                if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or name in model_files:
                    raise ValueError("Model package must use unique flat filenames")
                path = a.model_manifest.resolve().parent/name
                if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]:
                    raise ValueError("Model capsule checksum mismatch")
                tar.add(path, arcname=name); model_files.append(name)
    bundle = out/"source.bundle"
    subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(bundle), "HEAD"], check=True)
    spec = dict(node=a.node, tag=a.tag, source_commit=revision, seconds=a.seconds,
                no_improvement_seconds=a.no_improvement_seconds, policies=a.policies, workers=a.workers, cpus=a.cpus, memory_gib=a.memory_gib, circuits=a.circuits, seeds=a.seeds, modes=a.modes,
                storage_root=a.storage_root, experiment=a.experiment, probe_source_tag=a.probe_source_tag,
                training_seeds=a.training_seeds, audit_workers=a.audit_workers,
                audit_native_replay=a.audit_native_replay,
                audit_crosscheck_native_every=a.audit_crosscheck_native_every,
                build_native_flow=a.build_native_flow,
                model_bundle_sha256=None if model_bundle is None else hashlib.sha256(model_bundle.read_bytes()).hexdigest(),
                model_files=model_files, development_models=development_models,
                save_improvement_models=a.save_improvement_models,
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

    preparation = "from pathlib import Path\np=Path("+repr(spec["base"])+")\n"
    if a.storage_root:
        preparation += "target=Path("+repr(a.storage_root)+")/p.name\ntarget.mkdir(parents=True,exist_ok=False)\np.symlink_to(target,target_is_directory=True)\n"
    else:
        preparation += "p.mkdir(exist_ok=False)\n"
    prepared = remote(preparation+"print('prepared')\n")
    if "prepared" not in prepared.stdout:
        raise RuntimeError("Remote directory creation failed; inspect setup.log without overwriting another task")
    subprocess.run(["scp", "-o", "ControlPath="+socket, "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
                    str(bundle), host+":"+spec["base"]+"/source.bundle"], check=True, timeout=120)
    if model_bundle is not None:
        subprocess.run(["scp", "-o", "ControlPath="+socket, "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
                        str(model_bundle), host+":"+spec["base"]+"/models.tar.gz"], check=True, timeout=120)
    script = "spec=json.loads("+repr(json.dumps(spec))+")\n"
    script = "import json,subprocess,pathlib,hashlib,tarfile\n"+script+r'''
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
extra_mounts=[]; probe_source=None
if spec['experiment']=='policy-probe':
    training=pathlib.Path('/tmp/dengzengyong_rlzx_'+spec['probe_source_tag']+'_node'+str(spec['node'])+'_20261018')
    controller=json.loads((training/'controller.json').read_text())
    training_owner=json.loads((training/'ownership.json').read_text())
    if controller['phase']!='passed' or controller['commit']!=training_owner['source_commit']:
        raise RuntimeError('Training source has not reached an audited terminal pass')
    probe_source=training/'source/runs/comparison'
    extra_mounts=['-v',str(probe_source)+':/workspace/probe-source:ro']
if spec['experiment']=='model-probe':
    if hashlib.sha256((base/'models.tar.gz').read_bytes()).hexdigest()!=spec['model_bundle_sha256']:
        raise RuntimeError('Model bundle transport checksum mismatch')
    models=base/'models';models.mkdir(exist_ok=False);seen=set()
    with tarfile.open(base/'models.tar.gz','r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in spec['model_files'] or member.name in seen:
                raise ValueError('Unexpected model archive member')
            seen.add(member.name);tar.extract(member,path=models)
    if seen!=set(spec['model_files']): raise ValueError('Incomplete model package')
    extra_mounts=['-v',str(models)+':/workspace/models:ro']
cid=run(['docker','create','--name',name,'--label','owner=dengzengyong','--label','task=rlzx-anchor-20261018',
         '--cpus',str(spec['cpus']),'--memory',str(spec['memory_gib'])+'g','--shm-size','4g','--pids-limit','4096',
         '-v',str(base)+':/workspace/task','-v',str(old/'pyzx-pin')+':/workspace/pin:ro',
         '-v',str(old/'offline-deps')+':/workspace/offline:ro',*extra_mounts,
         '--entrypoint','/bin/bash',image,'-lc','sleep infinity'])
run(['docker','start',cid])
data=json.loads(run(['docker','inspect',cid]))[0]
meta=dict(container_name=name,container_id=cid,started_at=data['State']['StartedAt'],image=image,
          source_commit=revision,cpus=spec['cpus'],memory_gib=spec['memory_gib'],workers=spec['workers'],gpu=False,node=spec['node'],
          experiment=spec['experiment'],probe_source=None if probe_source is None else str(probe_source))
(base/'ownership.json').write_text(json.dumps(meta,indent=2)+'\n')
run(['docker','exec',cid,'git','config','--global','--add','safe.directory','/workspace/task/source'])
command=['python3',str(base/'source/tools/node_anchor_campaign.py'),str(base),'--container',name,
         '--commit',revision,'--seconds',str(spec['seconds']),'--workers',str(spec['workers']),
         '--no-improvement-seconds',str(spec['no_improvement_seconds']),'--policies',*spec['policies'],
         '--circuits',*spec['circuits'],'--seeds',*spec['seeds'],'--modes',*spec['modes'],
         '--experiment',spec['experiment'],'--audit-workers',str(spec['audit_workers']),
         '--audit-native-replay',spec['audit_native_replay'],
         '--audit-crosscheck-native-every',str(spec['audit_crosscheck_native_every'])]
if probe_source is not None:
    command+=['--probe-source','/workspace/probe-source','--training-seeds',*spec['training_seeds']]
if spec['experiment']=='model-probe':
    command+=['--model-manifest','/workspace/models/models.json']
    if spec['development_models']: command+=['--allow-development-models']
if spec['build_native_flow']:
    command+=['--build-native-flow']
if spec['save_improvement_models']:
    command+=['--save-improvement-models']
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
                                 "--task-prefix", "rlzx_"+a.tag, "--nodes", str(a.node), "--interval", "300",
                                 *(["--metadata-only"] if a.metadata_only_mirror else []),
                                 *(["--remote-archive-only"] if a.remote_archive_only else [])],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (out/"collector-launch.json").write_text(json.dumps(dict(pid=proc.pid, interval=300), indent=2)+"\n")
    print(json.dumps(launch))


if __name__ == "__main__":
    main()
