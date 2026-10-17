"""Post-hoc bounded first-visit proposal coverage; does not alter search results."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def one(spec):
 campaign,count=spec;campaign=Path(campaign)
 from zxopt.complete import Diagram
 from zxopt.complete.angles import Phase
 from zxopt.complete.ring import is_quarter
 from zxopt.jpv_actions import Proposals
 files={}
 for f in sorted(campaign.glob('jpv_*_s*')):
  if not f.is_dir():continue
  protocol=json.loads((f/'protocol.json').read_text())
  for g in sorted((f/'objects').glob('*.json.gz')):files.setdefault(g.stem.split('.')[0],g)
 root=protocol['root_graph'];keys=sorted(set(files)-{root});random.Random(20261017).shuffle(keys);keys=[root]+keys[:count-1]
 menu=Proposals(11,32,2048,general=True);rows=[];families=Counter()
 for key in keys:
  f=files[key];d=Diagram.from_record(json.loads(gzip.decompress(f.read_bytes())));assert d.digest()==key
  actions=menu.build(d);hist=Counter(a.axiom.name+':'+a.direction for a in actions);families.update(hist)
  rows.append(dict(graph=key,source=str(f),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),nodes=len(d.nodes),families=dict(hist),symbolic_nodes=sum(isinstance(n.phase,Phase) for n in d.nodes.values()),non_quarter_nodes=sum(n.kind in ('Z','X') and not is_quarter(n.phase) for n in d.nodes.values())))
 return dict(campaign=campaign.name,unique_states=len(files),samples=len(rows),A_proposals=sum(v for k,v in families.items() if k.startswith('A:')),families=dict(families),rows=rows,stats=menu.stats)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);p.add_argument('--samples',type=int,default=64);a=p.parse_args()
 specs=[(str(f),a.samples) for f in sorted(a.study.glob('general-*')) if (f/'campaign.json').exists()]
 with ProcessPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,specs))
 report=dict(status='passed',scope='Post-hoc sample: each distinct raw SHA once, proposal seed 11 and first visit; cap32, work2048. No extraction/cost lookahead; not exhaustive topology coverage.',records=records,samples=sum(r['samples'] for r in records),A_proposals=sum(r['A_proposals'] for r in records))
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':main()
