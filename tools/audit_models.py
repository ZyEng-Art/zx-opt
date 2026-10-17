"""Check frozen controls, online weight changes and the common initial prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("output",type=Path);args=p.parse_args()
    torch.set_num_threads(1)
    records=[]
    seeds=json.loads((args.campaign/"campaign.json").read_text())["seeds"]
    for seed in seeds:
        initial_hashes=[];prefixes=[]
        for arm in ["expanded_frozen","expanded_online"]:
            folder=args.campaign/(arm+"_s"+str(seed))
            initial=torch.load(folder/"initial.pt",map_location="cpu",weights_only=True)["state_dict"]
            final=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)["state_dict"]
            assert initial.keys()==final.keys()
            delta=sum(float((final[k]-initial[k]).double().square().sum())for k in initial)**.5
            assert all(torch.isfinite(v).all()for v in final.values())
            assert (delta==0.)if arm=="expanded_frozen"else(delta>0.)
            initial_hashes.append(json.loads((folder/"protocol.json").read_text())["initial_tensor_sha256"])
            events=[json.loads(l)for l in(folder/"events.jsonl").read_text().splitlines()][:128]
            fields=["action","parent_graph_key","graph_key","status","cost","reward","available","restart_key","selection"]
            prefixes.append([{k:e.get(k)for k in fields}for e in events])
            records.append(dict(arm=arm,seed=seed,parameter_l2_delta=delta,
                                initial_file_sha256=hashlib.sha256((folder/"initial.pt").read_bytes()).hexdigest(),
                                final_file_sha256=hashlib.sha256((folder/"final.pt").read_bytes()).hexdigest()))
        assert len(set(initial_hashes))==1
        assert len(prefixes[0])==len(prefixes[1])==128 and prefixes[0]==prefixes[1]
    output=dict(status="passed",seeds=seeds,frozen_online_initial_and_first128_identical=True,records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output))


if __name__=="__main__":main()
