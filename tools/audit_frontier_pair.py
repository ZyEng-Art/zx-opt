"""Check that the reserve changes search only after elite-frontier exhaustion."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();campaign=json.loads((args.campaign/"campaign.json").read_text())
    records=[]
    for seed in campaign["seeds"]:
        folders=[args.campaign/(arm+"_s"+str(seed))for arm in ["serialized","serialized_frontier"]]
        events=[[json.loads(line)for line in(folder/"events.jsonl").read_text().splitlines()]for folder in folders]
        reports=[json.loads((folder/"report.json").read_text())for folder in folders]
        first=next((i for i,e in enumerate(events[1])if e.get("restart_source")=="deferred"),None)
        prefix=events[1]if first is None else events[1][:first]
        usable=[[e for e in es if e["status"]!="excluded_after_deadline"]for es in [events[0],prefix]]
        fields=["action","parent_graph_key","graph_key","cost","status","reward",
                "available","restart_key","selection","context","admitted"]
        n=min(map(len,usable))
        assert n>0
        for i in range(n):
            a,b=({key:e.get(key)for key in fields}for e in [usable[0][i],usable[1][i]])
            assert a==b,(seed,i+1,a,b)
        if reports[0]["status"]=="retained_frontier_exhausted" and first is not None:
            assert first==len(events[0])==n
            assert events[1][first]["transition"]==n+1
            assert reports[1]["deferred"]["resumptions"]>0
        records.append(dict(seed=seed,common_prefix_transitions=n,
                            first_deferred_transition=None if first is None else first+1,
                            control_status=reports[0]["status"],reserve_status=reports[1]["status"],
                            control_best=reports[0]["best"]["cost"],reserve_best=reports[1]["best"]["cost"],
                            deferred_resumptions=reports[1]["deferred"]["resumptions"],status="passed"))
    result=dict(status="passed",checks="Same uniform action/state/cost/reward prefix before deferred recovery; first resumed transition follows baseline exhaustion",records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))


if __name__=="__main__":main()
