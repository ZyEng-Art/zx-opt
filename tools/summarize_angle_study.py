"""Aggregate audited study outputs, retaining each mode/target/arm/seed."""
import argparse
from collections import Counter
import hashlib,json,statistics,shutil
from pathlib import Path


def load(p): return json.loads(p.read_text())
def dump(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def paired(rows,left,right):
    records=[]
    for target in sorted({r['target'] for r in rows}):
        for seed in sorted({r['seed'] for r in rows}):
            a=next(r for r in rows if r['target']==target and r['seed']==seed and (r['mode'],r['arm'])==left)
            b=next(r for r in rows if r['target']==target and r['seed']==seed and (r['mode'],r['arm'])==right)
            delta=a['best']['cost']['gates']-b['best']['cost']['gates']
            records.append(dict(target=target,seed=seed,left_gates=a['best']['cost']['gates'],right_gates=b['best']['cost']['gates'],gate_delta=delta,anytime_delta=a['mean_anytime_gates']-b['mean_anytime_gates']))
    return dict(left=left,right=right,lower_is_better=True,wins=sum(r['gate_delta']<0 for r in records),ties=sum(r['gate_delta']==0 for r in records),losses=sum(r['gate_delta']>0 for r in records),records=records,mean_anytime_delta=statistics.mean(r['anytime_delta'] for r in records))


def main():
 p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('output',type=Path);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
 manifest=load(a.runs/'angle-study/study.json'); raw_summary=load(a.runs/'angle-study/summary.json')
 assert raw_summary['completed']==raw_summary['expected']==48
 rows=[]; audits=[]; probes=[]; totals=Counter(); errors=Counter()
 for target in manifest['targets']:
  for mode in ('conditional','general'):
   name=mode+'-'+target['name']; campaign=a.runs/'angle-study'/name; analysis=load(a.runs/'angle-analysis'/name/'analysis.json'); audit=load(a.runs/'angle-audit'/name/'verification.json'); source=load(a.runs/'angle-audit'/name/'source-audit.json'); probe=load(a.runs/'angle-audit'/name/'policy-probe.json')
   assert audit['status']==source['status']==probe['status']=='passed'
   audits.append(dict(name=name,**audit)); probes.append(dict(name=name,**probe))
   for k in ('accepted_rewrites','saved_qasm','trials'): totals[k]+=audit[k]
   for r in analysis['trials']:
    folder=campaign/r['folder']; protocol=load(folder/'protocol.json'); events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]; updates=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
    selected=Counter(e['action']['family'] for e in events if e['status'] in ('accepted','accepted_unscored'))
    row=dict(r,mode=mode,target=target['name'],input_gates=protocol['input_gates'],num_qubits=protocol['num_qubits'],A_accepted=sum(v for k,v in selected.items() if k.startswith('A:')),A_accepted_families={k:v for k,v in selected.items() if k.startswith('A:')},rewards=dict(nonzero=sum(e.get('reward',0)!=0 for e in events),positive=sum(e.get('reward',0)>0 for e in events),negative=sum(e.get('reward',0)<0 for e in events)),source_folder=name+'/'+r['folder'])
    if updates:
     row['updates_first']=updates[0]['metrics']; row['updates_last']=updates[-1]['metrics']
    rows.append(row)
    for k in ('max_graph_operator_error','max_qasm_operator_error'): errors[k]=max(errors[k],r[k])
   dest=out/name;dest.mkdir()
   for file in ('analysis.json','findings.txt','anytime.png','anytime.pdf','learning.png','learning.pdf','best.qasm'):
    f=a.runs/'angle-analysis'/name/file
    if f.exists():shutil.copyfile(f,dest/file)
   for file in ('verification.json','source-audit.json','policy-probe.json'):
    shutil.copyfile(a.runs/'angle-audit'/name/file,dest/file)
 table=[]
 for t in manifest['targets']:
  target=t['name']; group=[r for r in rows if r['target']==target]
  table.append(dict(target=target,qubits=group[0]['num_qubits'],input_gates=group[0]['input_gates'],baseline=group[0]['root_cost']['gates'],groups=[dict(mode=mode,arm=arm,gates=[r['best']['cost']['gates'] for r in sorted(group,key=lambda r:r['seed']) if r['mode']==mode and r['arm']==arm],updates=[r['updates'] for r in sorted(group,key=lambda r:r['seed']) if r['mode']==mode and r['arm']==arm]) for mode in ('conditional','general') for arm in ('jpv_frozen','jpv_online')]))
 pairs=[paired(rows,(mode,'jpv_online'),(mode,'jpv_frozen')) for mode in ('conditional','general')]
 pairs+=[paired(rows,('general',arm),('conditional',arm)) for arm in ('jpv_frozen','jpv_online')]
 timing={mode:dict(Counter({k:sum(r['timers'].get(k,0) for r in rows if r['mode']==mode) for k in ('matching','feasibility','inference','rewrite_verify','score','ppo')})) for mode in ('conditional','general')}
 result=dict(status='passed',timing_seconds=timing,manifest=manifest,table=table,paired=pairs,trials=rows,totals=dict(totals),max_errors=dict(errors),A_accepted=sum(r['A_accepted'] for r in rows),all_models_and_inputs_audited=True)
 dump(out/'summary.json',result);dump(out/'policy-probes.json',probes)
 print(json.dumps(dict(table=table,paired=[{k:v for k,v in p.items() if k!='records'} for p in pairs],totals=dict(totals),max_errors=dict(errors),A_accepted=result['A_accepted']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
