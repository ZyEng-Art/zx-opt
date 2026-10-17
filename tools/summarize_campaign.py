"""Export all-seed pilot evidence and standalone search/learning figures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_mean(updates,key):
    values=[u["metrics"][key]for u in updates if u["metrics"].get(key)is not None]
    return float(np.mean(values)) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--refresh",action="store_true",help="Regenerate presentation files from unchanged raw data")
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=args.refresh)
    summary=json.loads((args.campaign/"summary.json").read_text())
    campaign=json.loads((args.campaign/"campaign.json").read_text())
    rows=summary["rows"]
    if len(rows)!=summary["expected"] or any(r["status"]=="failed" for r in rows):
        raise RuntimeError("Cannot summarize an incomplete or failed campaign")
    grid=np.linspace(0,campaign["seconds"],361);curves={};trials=[];losses={}
    fig,ax=plt.subplots(figsize=(8,4.5))
    for row in rows:
        folder=args.campaign/row["folder"]
        protocol=json.loads((folder/"protocol.json").read_text())
        if protocol["tracked_dirty"]:raise AssertionError("Dirty experiment checkout")
        events=[json.loads(l)for l in(folder/"events.jsonl").read_text().splitlines()]
        state_costs={}
        parent_keys={e["parent_graph_key"] for e in events}
        unexpanded_unretained=set()
        registry_remaining={}
        for event in events:
            registry_key=event.get("action_registry_key")
            if registry_key is not None:
                registry_remaining[registry_key]=event["available"]-1
            if event["status"]=="accepted":
                state_costs.setdefault(event["state"],set()).add((event["cost"]["gates"],event["cost"]["cx"]))
                if not event["admitted"] and event["graph_key"] not in parent_keys:
                    unexpanded_unretained.add(event["graph_key"])
        curve=np.full(len(grid),protocol["root_cost"]["gates"],dtype=float)
        area=0.;previous_time=0.;incumbent=protocol["root_cost"]["gates"]
        for event in events:
            if event["status"]=="accepted":
                area+=(event["seconds"]-previous_time)*incumbent
                previous_time=event["seconds"];incumbent=min(incumbent,event["cost"]["gates"])
                curve[grid>=event["seconds"]]=np.minimum(curve[grid>=event["seconds"]],event["cost"]["gates"])
        area+=(campaign["seconds"]-previous_time)*incumbent
        curves.setdefault(row["arm"],[]).append(curve)
        trial=dict(row,git_commit=protocol["git_commit"],input_sha256=protocol["input_sha256"],
                   initial_tensor_sha256=protocol.get("initial_tensor_sha256"),software=protocol.get("software"),
                   events_sha256=sha(folder/"events.jsonl"),updates_sha256=sha(folder/"updates.jsonl"),
                   protocol_sha256=sha(folder/"protocol.json"),
                   invalid_proposals=sum(e["status"]=="rejected" for e in events),
                   deadline_exclusions=sum(e["status"]=="excluded_after_deadline" for e in events),
                   isomorphic_states_with_different_extraction_cost=sum(len(v)>1 for v in state_costs.values()),
                   max_isomorphic_gate_spread=max((max(c[0] for c in v)-min(c[0] for c in v) for v in state_costs.values()),default=0),
                   unretained_successors_without_logged_outgoing_actions=len(unexpanded_unretained),
                   representative_replacements=row.get("buffer",{}).get("representative_replacements",0),
                   registries_with_untried_logged_actions=sum(n>0 for n in registry_remaining.values()),
                   untried_logged_actions=sum(registry_remaining.values()),
                   deferred_resumptions=row.get("deferred",{}).get("resumptions",0),
                   retained_frontier_exhausted=row["status"]=="retained_frontier_exhausted",
                   mean_anytime_gates=area/campaign["seconds"])
        updates=[json.loads(l)for l in(folder/"updates.jsonl").read_text().splitlines()]
        if updates:
            losses.setdefault(row["arm"],[]).append((row["seed"],updates))
            n=max(1,len(updates)//4)
            keys=["policy_loss","value_loss","entropy","approx_kl","clip_fraction",
                  "value_explained_variance_after","normalized_entropy"]
            trial["learning"]={k:dict(first=metric_mean(updates[:n],k),last=metric_mean(updates[-n:],k))
                               for k in keys}
            trial["learning"]["optimizer_steps"]=sum(u["optimizer_steps"]for u in updates)
        trials.append(trial)
    aggregate=[]
    for arm,values in curves.items():
        a=np.array(values);line=ax.plot(grid,a.mean(0),label=arm)[0]
        for v in a:ax.plot(grid,v,color=line.get_color(),alpha=.15,lw=.7)
        group=[r for r in trials if r["arm"]==arm]
        aggregate.append(dict(arm=arm,seeds=[r["seed"]for r in group],
            final_gates=[r["best"]["cost"]["gates"]for r in group],
            mean_anytime_gates=float(np.mean([r["mean_anytime_gates"]for r in group])),
            mean_transitions=float(np.mean([r["transitions"]for r in group])),
            statuses=[r["status"]for r in group],
            actual_total_loop_seconds=[r["total_loop_seconds"]for r in group],
            early_exhausted_runs=sum(r["retained_frontier_exhausted"]for r in group),
            representative_replacements=sum(r["representative_replacements"]for r in group),
            unretained_successors_without_logged_outgoing_actions=sum(r["unretained_successors_without_logged_outgoing_actions"]for r in group),
            rejection_fraction=sum(r["invalid_proposals"]for r in group)/max(1,sum(r["transitions"]for r in group)),
            deadline_exclusions=sum(r["deadline_exclusions"]for r in group),
            duplicate_successor_fraction=sum(r["duplicate_successors"]for r in group)/max(1,sum(r["accepted"]for r in group))))
    ax.set(xlabel="Search wall time (seconds)",ylabel="Best total NAM gates",title=f"barenco_tof_3: {len(campaign['seeds'])} seed(s) per arm")
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    for suffix in ["png","pdf"]:fig.savefig(out/("anytime."+suffix),dpi=170)
    plt.close(fig)
    if losses:
        fig,axs=plt.subplots(1,3,figsize=(12,3.6))
        for arm,runs in losses.items():
            for seed,updates in runs:
                xs=[u["seconds"]for u in updates]
                for axis,key in zip(axs,["value_loss","normalized_entropy","value_explained_variance_after"]):
                    axis.plot(xs,[u["metrics"].get(key)for u in updates],label=f"{arm} s{seed}")
        for axis,label in zip(axs,["Value MSE","Normalized policy entropy","Value explained variance"]):
            axis.set(xlabel="Search wall time (seconds)",ylabel=label);axis.grid(alpha=.2)
        axs[0].set_yscale("log")
        axs[1].set_ylim(0,1.05)
        explained=[u["metrics"].get("value_explained_variance_after")
                   for runs in losses.values() for _,updates in runs for u in updates]
        explained=[v for v in explained if v is not None and np.isfinite(v)]
        axs[2].set_ylim(min([-.05]+[v-.05 for v in explained]),max([1.05]+[v+.05 for v in explained]))
        axs[0].legend(fontsize=7);fig.suptitle("PPO diagnostics on barenco_tof_3");fig.tight_layout()
        for suffix in ["png","pdf"]:fig.savefig(out/("learning."+suffix),dpi=170)
        plt.close(fig)
    best=min(trials,key=lambda r:(r["best"]["cost"]["gates"],r["best"]["cost"]["cx"]))
    shutil.copyfile(args.campaign/best["folder"]/"best.qasm",out/"best.qasm")
    result=dict(status="passed",scope="same-case pilot only; all arms/seeds; no generalization or completeness claim",
                campaign_sha256=sha(args.campaign/"campaign.json"),summary_sha256=sha(args.campaign/"summary.json"),
                budget_seconds=campaign["seconds"],trials=trials,aggregate=sorted(aggregate,key=lambda r:r["arm"]),
                best_source=dict(arm=best["arm"],seed=best["seed"],sha256=sha(out/"best.qasm")))
    (out/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["barenco_tof_3 pilot: every arm and seed retained.","Final gates (seeds in the JSON report), time-averaged best gates, rejection fraction:"]
    for r in result["aggregate"]:lines.append(f"{r['arm']}: {r['final_gates']}; anytime={r['mean_anytime_gates']:.3f}; rejected={r['rejection_fraction']:.1%}; repeated-successors={r['duplicate_successor_fraction']:.1%}; early-exhausted={r['early_exhausted_runs']}; representative-replacements={r['representative_replacements']}")
    lines += ["Loss and value fit diagnose training; compare the search curves to assess useful learning.",
              "Early retained-frontier exhaustion is not global exhaustion or training convergence; the last best is held until the fixed budget when integrating curves.",
              "An unretained successor with no logged outgoing action is only a diagnostic count; it may have no legal outgoing rewrites.",
              "Bounded proposals and extraction rejection do not give a completeness guarantee."]
    (out/"findings.txt").write_text("\n".join(lines)+"\n")
    print(json.dumps(result["aggregate"],indent=2))


if __name__=="__main__":main()
