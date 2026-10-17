"""Compare initial/final policies on the same original-circuit observation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch_geometric
from zxopt.legacy import environment
from zxopt.kernel import Kernel
from zxopt.policy import ExpandedAgent, observation, batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    campaign=json.loads((args.campaign/"campaign.json").read_text())
    arms=[arm for arm in campaign["arms"]if arm.endswith("_online")]
    assert len(arms)==1
    env=environment();base,old=Kernel(env).prepare(env.graph)
    obs,actions=observation(base,old,[],[0.,0.,0.,1.])
    packed=batch([obs]);records=[]
    for seed in campaign["seeds"]:
        folder=args.campaign/(arms[0]+"_s"+str(seed));probs=[];values=[]
        for name in ["initial.pt","final.pt"]:
            net=ExpandedAgent();net.load_state_dict(torch.load(folder/name,map_location="cpu",weights_only=True)["state_dict"]);net.eval()
            with torch.no_grad():
                logits=net.actor(packed[0]).flatten()[packed[0].y!=-1].double()
                probs.append(logits.softmax(0));values.append(float(net.get_value(packed[1])[0]))
        initial,final=probs
        entropy=lambda q:float(-(q*q.log()).sum())/math.log(len(q))
        records.append(dict(seed=seed,action_count=len(actions),initial_probabilities=initial.tolist(),
                            final_probabilities=final.tolist(),initial_normalized_entropy=entropy(initial),
                            final_normalized_entropy=entropy(final),total_variation=float((initial-final).abs().sum()/2),
                            kl_final_to_initial=float((final*(final.log()-initial.log())).sum()),
                            initial_value=values[0],final_value=values[1]))
    result=dict(status="passed",scope="Same original barenco_tof_3 root, fixed full legal-action set and context; offline parameter-change diagnostic only",
                actions=[a.record()for a,_ in actions],context=[0.,0.,0.,1.],records=records,
                offline_runtime=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__),
                probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({'status':'passed','fixed_root_policy_changes':[{k:r[k]for k in ['seed','total_variation','initial_normalized_entropy','final_normalized_entropy']}for r in records]}))


if __name__=="__main__":main()
