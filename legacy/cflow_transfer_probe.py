"""Frozen ranking transfer to actual NAM cost with strong cflow controls.

Research on fresh synthetic <=5qubit inputs, no optimizer integration or PPO.
Uses the paper's pinned PyZX fork. NAM=(H,RZ,X,CX). Preserve wire order.
"""
import hashlib,json,os,random,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import pyzx as zx
from pyzx.rules import apply_rule,lcomp,pivot,match_lcomp_parallel,match_pivot_parallel
from qiskit import qasm2,transpile
from rank_probe import features,predict

COMMIT='298693bed2608e84fcb44a672282f873321c70b7'


def timeout_handler(signum,frame):raise TimeoutError('Candidate exceeded 5s operation budget')


def nam(circuit):
    c=transpile(qasm2.loads(circuit.to_basic_gates().to_qasm()),basis_gates=['h','rz','x','cx'],optimization_level=0)
    assert set(c.count_ops()).issubset({'h','rz','x','cx'})
    return c


def cost(c):return dict(gates=len(c),cx=int(c.count_ops().get('cx',0)),depth=int(c.depth()))


def key(c,objective):
    v=cost(c)
    return (v['gates'],v['cx']) if objective=='nam' else (v['cx'],v['gates'])


def clean(c):return zx.basic_optimization(c.to_basic_gates().split_phase_gates(),do_swaps=False).to_basic_gates()


def direct_cflow(graph):
    g=graph.clone()
    zx.flow_2Q_simp(g,cFlow=True,max_lc_unfusions=2,max_p_unfusions=2)
    return nam(clean(zx.extract_simple(g,up_to_perm=False)))


def complete(graph):
    start=time.perf_counter()
    g=graph.clone();g.normalize()
    extracted=clean(zx.extract_circuit(g,up_to_perm=False))
    extraction=time.perf_counter()-start
    g=extracted.to_graph()
    zx.teleport_reduce(g)
    zx.to_graph_like(g)
    candidate=direct_cflow(g)
    return candidate,dict(gflow_extraction_basic_seconds=extraction,
                          subsequent_teleport_cflow_nam_seconds=time.perf_counter()-start-extraction)


def choices(graph):
    result=[('lc',(m[0],),m) for m in match_lcomp_parallel(graph,allow_interacting_matches=True)]
    result += [('pivot',(m[0],m[1]),m) for m in match_pivot_parallel(graph,allow_interacting_matches=True)]
    return result


def rewrite(graph,action):
    kind,vertices,match=action
    g=graph.clone()
    apply_rule(g,lcomp if kind=='lc' else pivot,[match])
    return g


def main():
    model_path,out=map(Path,sys.argv[1:3])
    seeds=int(sys.argv[3]) if len(sys.argv)>3 else 20
    seed_origin=int(sys.argv[4]) if len(sys.argv)>4 else 26020000
    out.mkdir(parents=True,exist_ok=False)
    model=json.loads(model_path.read_text())
    assert hasattr(zx,'flow_2Q_simp') and COMMIT in zx.__file__
    signal.signal(signal.SIGALRM,timeout_handler)
    protocol=dict(scope=__doc__,fork_commit=COMMIT,pyzx_file=zx.__file__,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('verify_qasm_batch.py')]},
        seeds=seeds,seed_origin=seed_origin,shapes=[[3,30],[3,60],[5,30],[5,60]],
        first_branches=4,continuations=4,candidate_cap=24,candidate_timeout_seconds=5,
        input_preprocessing='split_phase_gates,basic_optimization(do_swaps=False),teleport_reduce,to_graph_like',
        candidate_completion='fixed-wire gflow extraction,basic,teleport,cflow(unfusions2),fixed-wire simple extraction,basic,NAM conversion',
        baseline='best separately for NAM/CX of raw NAM, initial basic NAM, direct cflow, zero-action hybrid completion',
        model='frozen previous synthetic-basic-gate two-step model; no NAM/CX retraining',
        timings='single sequential run, randomized method order; preprocessing/baselines separate, verification separate; search includes ranking/matching/completion',
        up_to_perm=False,do_swaps=False,scope_limit='synthetic <=5qubit transfer diagnostic, not new pass/all600/family generalization')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    tasks=[];records=[];baseline_records=[]
    start_all=time.perf_counter()
    for qubits,depth in protocol['shapes']:
        for seed in range(seed_origin,seed_origin+seeds):
            case=f'q{qubits}_d{depth}_s{seed}';path=out/case;path.mkdir()
            random.seed(seed);np.random.seed(seed)
            source=zx.Circuit.from_graph(zx.generate.cliffordT(qubits,depth,p_t=.25,p_s=.25,p_hsh=.25)).to_basic_gates()
            original=nam(source);source_path=path/'input.qasm';source_path.write_text(qasm2.dumps(original)+'\n')
            prep_start=time.perf_counter()
            basic=clean(source)
            basic_nam=nam(basic)
            graph=basic.to_graph();zx.teleport_reduce(graph);zx.to_graph_like(graph)
            prep_seconds=time.perf_counter()-prep_start
            baseline_start=time.perf_counter()
            baseline_candidates=[('raw',original),('basic',basic_nam)]
            baseline_errors=[]
            for name,fn in [('direct_cflow',lambda:direct_cflow(graph)),('zero_action_hybrid',lambda:complete(graph)[0])]:
                operation_start=time.perf_counter()
                try:
                    signal.setitimer(signal.ITIMER_REAL,5)
                    c=fn();baseline_candidates.append((name,c))
                    (path/(name+'.qasm')).write_text(qasm2.dumps(c)+'\n')
                    tasks.append(dict(case=case,kind=name,source=str(source_path),target=str(path/(name+'.qasm'))))
                except Exception as exc:
                    baseline_errors.append(dict(name=name,error_type=type(exc).__name__,error=str(exc)))
                finally:signal.setitimer(signal.ITIMER_REAL,0)
            baselines={objective:min(baseline_candidates,key=lambda item:key(item[1],objective)) for objective in ['nam','cx']}
            baseline_seconds=time.perf_counter()-baseline_start
            for objective,(name,c) in baselines.items():
                f=path/('baseline_'+objective+'.qasm');f.write_text(qasm2.dumps(c)+'\n')
                tasks.append(dict(case=case,kind='baseline_'+objective,source=str(source_path),target=str(f)))
            base_row=dict(case=case,qubits=qubits,generator_depth=depth,seed=seed,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                input=cost(original),preprocessing_seconds=prep_seconds,baseline_seconds=baseline_seconds,
                candidates={name:cost(c) for name,c in baseline_candidates},errors=baseline_errors,
                baseline={obj:dict(name=name,**cost(c)) for obj,(name,c) in baselines.items()})
            baseline_records.append(base_row)
            with (out/'baselines.jsonl').open('a') as handle:handle.write(json.dumps(base_row)+'\n')
            methods=['random','edges','learned'];random.Random(case).shuffle(methods)
            for method in methods:
                started=time.perf_counter()
                available=choices(graph);random.Random(seed).shuffle(available);available=available[:24]
                ranked=[]
                if method=='random':
                    order=list(range(len(available)));random.Random(seed+99000).shuffle(order)
                else:
                    for action in available:
                        trial=rewrite(graph,action)
                        ranked.append(dict(graph=trial,features=features(graph,trial,action[0],action[1],qubits,len(baselines['nam'][1]))))
                    scores=predict(model,ranked) if method=='learned' and ranked else np.asarray([r['features'][18] for r in ranked])
                    order=np.argsort(-scores,kind='stable').tolist()
                preparation=time.perf_counter()-started
                best={obj:c for obj,(name,c) in baselines.items()};evaluations=[]
                for index in order[:4]:
                    first=available[index]
                    trial=rewrite(graph,first) if method=='random' else ranked[index]['graph']
                    second_choices=choices(trial)
                    random.Random(case+str(first[1])+first[0]).shuffle(second_choices)
                    branches=[('first',trial,first)]
                    branches += [('second',rewrite(trial,a),a) for a in second_choices[:4]]
                    for stage,g,action in branches:
                        event=dict(stage=stage,first_kind=first[0],first_vertices=first[1],kind=action[0],vertices=action[1])
                        evaluations.append(event);op_start=time.perf_counter()
                        try:
                            signal.setitimer(signal.ITIMER_REAL,5)
                            c,stages=complete(g)
                            event.update(status='completed',cost=cost(c),stages=stages)
                            for obj in ['nam','cx']:
                                if key(c,obj)<key(best[obj],obj):best[obj]=c
                        except Exception as exc:
                            event.update(status='error',error_type=type(exc).__name__,error=str(exc))
                        finally:signal.setitimer(signal.ITIMER_REAL,0)
                        event['seconds']=time.perf_counter()-op_start
                search_seconds=time.perf_counter()-started
                row=dict(case=case,seed=seed,method=method,candidates=len(available),search_seconds=search_seconds,
                    ranking_preparation_seconds=preparation,evaluations=evaluations,
                    baseline_errors=baseline_errors,baseline={obj:cost(c) for obj,(name,c) in baselines.items()},
                    after={obj:cost(c) for obj,c in best.items()},status='pending_independent_verification')
                for obj,c in best.items():
                    f=path/(method+'_'+obj+'.qasm');f.write_text(qasm2.dumps(c)+'\n')
                    tasks.append(dict(case=case,kind=method+'_'+obj,source=str(source_path),target=str(f)))
                records.append(row)
                with (out/'rows.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
            print(case,'baseline',base_row['baseline'],'actions',len(available),flush=True)
    tasks_path=out/'verification_tasks.json';tasks_path.write_text(json.dumps(tasks,indent=2)+'\n')
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    with (out/'verification.log').open('w') as handle:
        verifier=subprocess.run([sys.executable,str(Path(__file__).with_name('verify_qasm_batch.py')),str(tasks_path),str(out/'verification.json')],
                                env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=180)
    verification=json.loads((out/'verification.json').read_text())
    proof={(r['case'],r['kind']):r['passed'] for r in verification['rows']}
    summary={}
    for method in ['random','edges','learned']:
        selected=[r for r in records if r['method']==method]
        summary[method]=dict(cases=len(selected),
            verified=sum(all(proof.get((r['case'],method+'_'+obj),False) for obj in ['nam','cx']) for r in selected),
            baseline_error_cases=sum(bool(r['baseline_errors']) for r in selected),
            candidate_errors=sum(sum(e['status']=='error' for e in r['evaluations']) for r in selected),
            nam_saved=sum(r['baseline']['nam']['gates']-r['after']['nam']['gates'] for r in selected),
            nam_improved=sum(r['after']['nam']['gates']<r['baseline']['nam']['gates'] for r in selected),
            cx_saved=sum(r['baseline']['cx']['cx']-r['after']['cx']['cx'] for r in selected),
            cx_improved=sum(r['after']['cx']['cx']<r['baseline']['cx']['cx'] for r in selected),
            search_seconds=sum(r['search_seconds'] for r in selected),evaluations=sum(len(r['evaluations']) for r in selected))
    for row in records:
        row['status']='verified' if all(proof.get((row['case'],row['method']+'_'+obj),False) for obj in ['nam','cx']) else 'verification_failed'
    (out/'verified_rows.json').write_text(json.dumps(records,indent=2)+'\n')
    report=dict(status='completed' if verifier.returncode==0 else 'verification_failed',cases=len(baseline_records),summary=summary,
        baseline_errors=sum(len(r['errors']) for r in baseline_records),
        preprocessing_seconds=sum(r['preprocessing_seconds'] for r in baseline_records),
        baseline_seconds=sum(r['baseline_seconds'] for r in baseline_records),verification_seconds=verification['total_seconds'],
        verification_tasks=verification['tasks'],verification_passed=verification['passed'],verifier_returncode=verifier.returncode,
        total_seconds=time.perf_counter()-start_all,
        limitation='Frozen-model transfer only, not retrained on NAM/CX/cflow. Standard lc/Pauli pivot subset. Synthetic small circuits. No all600/new-family/default integration claim.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if verifier.returncode:sys.exit(1)


if __name__=='__main__':main()
