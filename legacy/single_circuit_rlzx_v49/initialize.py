"""Fresh single-circuit base weights; no broad-training or optimized-circuit seed."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent;sys.path.insert(0,str(R/'quarl_style_rlzx_v10'))
import torch
from runtime import source_classes
def main():
    torch.set_num_threads(1);_,Agent=source_classes(R/'upstream_source_v1');rows=[]
    for seed in [11,29,47]:
        path=HERE/f'initial_seed{seed}.pt';assert not path.exists();torch.manual_seed(seed);net=Agent(None,'cpu',c_hidden=32,c_hidden_v=32)
        torch.save(dict(state_dict=net.state_dict(),steps=0,width=32,seed=seed,objective='nam',scope='FreshoriginalGATweights, singlebarenco on-targetsearch study; no pretraining oroptimizedcircuitseed.'),path)
        rows.append(dict(seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE/'initial_models.json').write_text(json.dumps(dict(status='passed',models=rows),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
