"""Check saved NAM circuits with Qiskit and audit all recorded transitions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Operator


def error(a,b):
    if a.shape!=b.shape or not np.isfinite(b).all():raise AssertionError("Invalid saved operator")
    overlap=np.vdot(a,b);phase=overlap/abs(overlap)if abs(overlap)else 1.
    return float(np.max(np.abs(a-b/phase)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path);p.add_argument("input",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();reference=Operator(qasm2.loads(args.input.read_text())).data
    expected=hashlib.sha256(args.input.read_bytes()).hexdigest()
    rows=json.loads((args.campaign/"summary.json").read_text())["rows"]
    checked={};records=[];total=0;max_error=0.
    for row in rows:
        folder=args.campaign/row["folder"]
        protocol=json.loads((folder/"protocol.json").read_text())
        assert protocol["input_sha256"]==expected and not protocol["tracked_dirty"]
        events=[json.loads(l)for l in(folder/"events.jsonl").read_text().splitlines()]
        assert len(events)==row["transitions"]
        accepted=[e for e in events if e["status"]=="accepted"]
        assert len(accepted)==row["accepted"]
        costs={}
        for path in sorted((folder/"candidates").glob("*.qasm"))+[folder/"baseline.qasm",folder/"best.qasm"]:
            digest=hashlib.sha256(path.read_bytes()).hexdigest();total+=1
            if path.parent.name=="candidates":assert digest==path.stem
            if digest not in checked:
                c=qasm2.loads(path.read_text())
                assert c.num_qubits==5 and set(c.count_ops())<={"h","rz","x","cx"}
                err=error(reference,Operator(c).data);assert err<2e-8,(path,err)
                max_error=max(max_error,err)
                checked[digest]=dict(gates=len(c),cx=int(c.count_ops().get("cx",0)),depth=int(c.depth()))
            costs[digest]=checked[digest]
        assert checked[row["best"]["sha256"]]==row["best"]["cost"]
        best=protocol["root_cost"]["gates"];chain={};record_chain={}
        serialized_scope = protocol["config"].get("dedup") and protocol["config"].get("dedup_scope") == "serialized"
        attempted=set();serialized_costs={};last_parent={};deferred_trajectories=set()
        reserve=protocol["config"].get("frontier_reserve",0)
        if protocol.get("root_graph_key"):
            serialized_costs[protocol["root_graph_key"]]=protocol["root_cost"]
        for e in events:
            trajectory=e["trajectory"]
            parent=e["parent_graph_key"]
            if "deferred" in e:
                assert e["deferred"]["capacity"]==reserve
                assert 0<=e["deferred"]["size"]<=reserve
                assert 0<e["buffer"]["size"]<=e["buffer"]["capacity"]==64
                if e.get("restart_source")=="deferred" and trajectory not in deferred_trajectories:
                    assert reserve>0 and e["restart_key"] in last_parent
                    assert last_parent[e["restart_key"]]["available"]>1
                    assert e["trajectory_step"]==1
                    deferred_trajectories.add(trajectory)
                assert e["deferred"]["resumptions"]==len(deferred_trajectories)
            if parent in serialized_costs:assert serialized_costs[parent]==e["previous_cost"]
            serialized_costs[parent]=e["previous_cost"]
            if serialized_scope:
                assert e["action_registry_key"]==parent
                action=e["action"]
                signature=(parent,action["kind"],tuple(action["vertices"]),tuple(sorted(action["subset"])),action["phase"])
                assert signature not in attempted,(folder,signature)
                attempted.add(signature)
            last_parent[parent]=e
            if trajectory in chain:assert e["previous_cost"]==chain[trajectory]
            if trajectory in record_chain:assert e["previous_record"]==record_chain[trajectory]
            if e["status"]=="accepted":
                assert e["seconds"]<=row["budget_seconds"]
                assert e["sha256"]in costs and e["cost"]==costs[e["sha256"]]
                if e["graph_key"]in serialized_costs:assert serialized_costs[e["graph_key"]]==e["cost"]
                serialized_costs[e["graph_key"]]=e["cost"]
                new_record=min(e["previous_record"],e["cost"]["gates"])
                expected_reward=((e["previous_cost"]["gates"]-e["cost"]["gates"])if protocol["config"]["reward"]=="current"
                                 else(e["previous_record"]-new_record))/10.
                assert abs(e["reward"]-expected_reward)<1e-12
                assert e["vertices"]<=protocol["max_vertices"]
                chain[trajectory]=e["cost"];best=min(best,e["cost"]["gates"])
                record_chain[trajectory]=new_record
            elif e["status"]=="rejected":
                assert e["reward"]==0.;chain[trajectory]=e["previous_cost"]
                record_chain[trajectory]=e["previous_record"]
        assert best==row["best"]["cost"]["gates"]
        records.append(dict(arm=row["arm"],seed=row["seed"],transitions=len(events),accepted=len(accepted),
                            serialized_action_pruning_checked=bool(serialized_scope),
                            deferred_restart_trajectories_checked=len(deferred_trajectories),status="passed"))
    output=dict(status="passed",input_sha256=expected,trials=len(rows),saved_qasm_files_checked=total,
                distinct_qasm_operators_checked=len(checked),max_operator_error=max_error,
                checks="NAM gate counts, fixed named wires, all 32x32 operator entries, unit-modulus global phase only, rewards, deadlines, node bounds, transition chains, exact-key cost consistency and serialized action-registry uniqueness",records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n");print(json.dumps(output))


if __name__=="__main__":main()
