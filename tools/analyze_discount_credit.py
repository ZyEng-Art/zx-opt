"""Compare endpoint gate improvements with discounted recorded episode rewards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(folder, gamma):
    protocol = json.loads((folder/"protocol.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    counters = Counter(); examples = []; trajectory = None; rewards = []; start = last = None
    peak = 0; final_known = start_known = True; first_transition = None

    def finish(completed, explicit_terminal):
        if not rewards: return
        total = sum(rewards); discounted = sum(gamma**i*r for i,r in enumerate(rewards))
        assert abs(total-(start-last)/10) < 1e-9
        counters["episodes"] += 1
        if not completed:
            counters["open_final_episode"] += 1; return
        counters["completed_episodes"] += 1
        if not start_known:
            counters["completed_with_unknown_start_cost"] += 1; return
        if not final_known:
            counters["completed_with_unknown_final_cost"] += 1; return
        if last < start:
            counters["gate_improving_episodes"] += 1
            if discounted < -1e-10:
                counters["gate_improving_but_negative_discounted_return"] += 1
                if len(examples) < 5:
                    examples.append(dict(trajectory=trajectory, first_transition=first_transition,
                        steps=len(rewards), start_gates=start, final_gates=last, peak_gates=peak,
                        undiscounted_return=total, discounted_return=discounted,
                        explicit_terminal=explicit_terminal))
        elif last == start:
            counters["gate_neutral_episodes"] += 1
            if discounted < -1e-10: counters["gate_neutral_but_negative_discounted_return"] += 1

    prior_terminal = False
    with (folder/"events.jsonl").open() as stream:
        for line in stream:
            event=json.loads(line)
            if event["status"] == "excluded_after_deadline": continue
            if event["trajectory"] != trajectory:
                finish(True, prior_terminal)
                trajectory=event["trajectory"];rewards=[]
                start=last=event["last_scored_cost"]["gates"];peak=start
                first_transition=event["transition"];start_known=event["parent_cost"] is not None;final_known=start_known
            rewards.append(event["reward"])
            if event["status"] == "accepted":
                last=event["cost"]["gates"];peak=max(peak,last);final_known=True
            elif event["status"] == "accepted_unscored": final_known=False
            prior_terminal=event["terminal"]
    finish(prior_terminal, prior_terminal)
    return dict(trial=str(folder), source_commit=protocol["git_commit"], policy=protocol["config"]["policy"],
                seed=protocol["config"]["seed"], counters=dict(counters), examples=examples,
                events_sha256=sha(folder/"events.jsonl"), report_sha256=sha(folder/"report.json"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--gamma",type=float,default=.99)
    a=p.parse_args()
    if not 0 < a.gamma <= 1: raise ValueError("Discount must be in (0,1]")
    summary=json.loads((a.campaign/"summary.json").read_text())
    assert summary["completed"]==summary["expected"]==len(summary["rows"])
    assert not any(r.get("status")=="failed" for r in summary["rows"])
    records=[analyze(a.campaign/r["folder"],a.gamma) for r in summary["rows"]]
    totals=Counter()
    for row in records: totals.update(row["counters"])
    finished_path=a.campaign/"finished.json";audit_path=a.campaign/"audit/verification.json"
    audited=(finished_path.exists() and audit_path.exists()
             and json.loads(finished_path.read_text())["status"]=="passed"
             and json.loads(audit_path.read_text())["status"]=="passed")
    result=dict(status="descriptive_closed_training_reward_analysis",gamma=a.gamma,records=records,totals=dict(totals),
        analyzer_sha256=sha(Path(__file__)),source_whole_audit_passed=audited,
        scope="Arithmetic on recorded gate-delta rewards, not a PPO efficacy/convergence claim. Complete episode boundaries follow explicit terminals or subsequent observed restarts; final open episodes and unknown start/final costs excluded from endpoint comparisons. Shared frozen controls/related seeds introduce dependence. Discounted return shown has zero terminal bootstrap; lambda/value estimates are not evaluated here. Whole source audit may still be pending.")
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],gamma=a.gamma,totals=dict(totals))))


if __name__=="__main__":main()
