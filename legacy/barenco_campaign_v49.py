"""Single supplied circuit: online learning is part of the timed search itself."""
import argparse,hashlib,json,multiprocessing as mp,random,subprocess,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
R=Path(__file__).resolve().parent;HERE=R/'single_circuit_rlzx_v49'
sys.path.insert(0,str(R/'quarl_style_rlzx_v10'));from run import trial
sys.path.insert(0,str(HERE));from worker import worker
METHODS=['uniform','frozen','online_current','online_record']
CONFIG=dict(objective='nam',buffer_capacity=64,cache_capacity=256,alpha=1.5,rollout=128,minibatch=32,epochs=4,lr=.0003)
def one(task):
    spec,method,seed,block,budget,output=task;folder=Path(output)/spec['case']/f'{method}_b{block}_m{seed}'
    d=trial(spec,method,HERE/f'initial_seed{seed}.pt',R/'upstream_source_v1',dict(CONFIG,model_seed=seed),219000000+seed,budget,folder,worker_fn=worker)
    assert d['status']not in['worker_error','worker_failed'],d.get('error');e=d['events'];assert e
    assert all(x['cost_mode']=='clean'and(x['action']!=9003001 or not x['legal_nonstop'])for x in e)
    assert all(x['cost']==x['clean_cost']for x in e)
    if method.startswith('online_'):assert d['updates']and any(u['optimizer_steps']for u in d['updates'])
    else:assert not d['updates']
    first_passage={str(k):next((x['received_seconds']for x in e if x['cost']['gates']<=k),None)for k in[39,38,37,36,35,34,33,32,31]}
    return dict(case=spec['case'],method=method,block=block,model_seed=seed,input_sha256=d['input_sha256'],supplied_input_sha256=spec['supplied_input_sha256'],baseline_gates=d['baselines']['nam']['cost']['gates'],final_gates=d['incumbent']['nam']['cost']['gates'],gate_gain=d['baselines']['nam']['cost']['gates']-d['incumbent']['nam']['cost']['gates'],transitions=len(e),optimizer_updates=len(d['updates']),optimizer_steps=sum(u['optimizer_steps']for u in d['updates']),first_passage_seconds=first_passage,unique_states=e[-1]['unique_states'],folder=str(folder),status=d['status'])
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--seconds',type=float,default=600.);p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False);input_path=R/'single_barenco_input_v47/barenco_tof_3.qasm';h=hashlib.sha256(input_path.read_bytes()).hexdigest();assert h=='fdbc0a1b82dbb2c1f9f05a7501a708e1a220cebb015c41dea9ee0d175694b5ea'
    spec=dict(case='barenco_tof_3',qubits=5,depth=46,seed=218000000,input_path=str(input_path),supplied_input_sha256=h)
    seeds=[11]if args.smoke else[11,29,47];budget=12. if args.smoke else args.seconds
    tasks=[(spec,m,s,b,budget,str(out))for b,s in enumerate(seeds)for m in METHODS];random.Random(220000000).shuffle(tasks)
    paths=[Path(__file__),R/'circuit_env_v25.py',R/'completion_feedback_v37.py',R/'nam_postprocess_v33.py',R/'verify_single_barenco_v49.py']+list(HERE.glob('*.py'))+[p for p in(R/'quarl_style_rlzx_v10').iterdir()if p.is_file()and p.suffix in['.py','.cpp','.so']]
    paths+=[R/'upstream_source_v1/zx_env.py',R/'upstream_source_v1/rl_agent.py',R/'reconstructed_ppo.py',R/'cflow_transfer_probe.py',R/'rank_probe.py',R/'action_probe.py',R/'short_path_probe.py']
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths};models={str(s):dict(path=str(HERE/f'initial_seed{s}.pt'),sha256=hashlib.sha256((HERE/f'initial_seed{s}.pt').read_bytes()).hexdigest())for s in seeds}
    protocol=dict(specs=[spec],methods=METHODS,model_seeds=seeds,budget_seconds=budget,trials=len(tasks),parallel_parents=args.workers,config=CONFIG,sources=sources,models=models,
        scope='UserexplicitlyrequestsONLYbarenco_tof_3 on-targetlearnedsearch, no validationcircuits/generalization. Same5q46gateQuarlNAMrm input; commonstrongfixedwirebaselinechargedsetup, allfourmethods fromsameZXroot/64diversebuffer/cleanedNAMfeedback. ActualH/RZ/X/CXgateobjective,CXtiebreak. Freshstep0originalGAT+zero-lastfullcontextactorresidual, seed11/29/47; frozenvsbothonline rewards sameinitialfunction. Uniformcontrol samepolicyactionmask/archive/costoracle.',
        learning='OnlinePPO partoftimedsearch; rollout128/mini32/4epochs/lr.0003/gamma.99/lambda.95/clip.2/entropy.01/grad.5. Oneglobalcritic. online_current:deltaNAM/10; online_record:trajectoryrecordbestimprovement/10. NonSTOPmaskinsideactor ensures sampling/PPOlogprob match; STOP retainedwhen nolegalrewrites. Allarms samealpha1.5/rootcutoff, original35stephorizon. No HAE/fullQuarlreproductionclaim.',
        selection='Everyarm/seedreported onTHIScircuit. Compareanytimebest andfirst-passage gates vsclock primarily, transitionssecondarily. Nooptimized35/31witness entersinitial/buffer. Noheldouttest orpopulationgeneralizationclaim. PPOloss/entropy/KL/valuefit retainedaslearningdiagnostics, actualbestcircuit decidesutility.',
        timing='Parentreceivedstrictwalltime. Includesactor/critic/Env/exactcompletion/NAMcleanup/buffer/PPO/onlinecheckpoints/IPC. Initialparse/model/baselines andfinalreport/matrixproofexcluded. Eachrun independentbuffer, no cross-armcandidate sharing.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];inputs=set();baselines=set()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'))as pool:
        for f in as_completed([pool.submit(one,t)for t in tasks]):
            row=f.result();rows.append(row);inputs.add(row['input_sha256']);baselines.add(row['baseline_gates']);assert len(inputs)==len(baselines)==1
            with(out/'rows.jsonl').open('a')as w:w.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    subprocess.run([sys.executable,str(R/'verify_single_barenco_v49.py'),str(out)],check=True)
    proof=json.loads((out/'verification.json').read_text());assert proof['status']=='passed'
    if args.smoke:
        folders={m:out/spec['case']/f'{m}_b0_m11'for m in METHODS if m!='uniform'}
        traces={m:json.loads((f/'trial.json').read_text())for m,f in folders.items()};n=min(128,min(len(t['events'])for t in traces.values()));assert n==128
        fields=['graph_key','sha256','cost','action','start_key','distance','buffer']
        sequences=[[{k:e[k]for k in fields}for e in traces[m]['events'][:128]]for m in folders];assert all(s==sequences[0]for s in sequences)
        assert len({traces[m]['events'][0]['initial_tensor_sha256']for m in folders})==1
    for name,h in sources.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    (out/'report.json').write_text(json.dumps(dict(status='passed',trials=len(rows),source_unchanged=True,smoke_initial_and_first128_exact=args.smoke,verification={k:v for k,v in proof.items()if k!='records'}),indent=2))
if __name__=='__main__':main()
