"""Compare initial/final preferences on the same root and fixed proposal list."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args()
    import numpy as np
    import torch
    from qiskit import qasm2
    from zxopt.complete import Diagram
    from zxopt.legacy import environment
    from zxopt.general_kernel import Views, GeneralKernel
    from zxopt.general_policy import GeneralAgent, observation, batch
    from zxopt.general_search import candidate_digest
    from zxopt.general_feasibility import Feasibility
    from zxopt.verify import qiskit_matrix_zx_order
    from zxopt.experiment_input import recorded_input
    torch.set_num_threads(1)
    summary = json.loads((a.campaign/"summary.json").read_text()); results = []
    for row in summary["rows"]:
        if not row["arm"].endswith("online"):
            continue
        folder = a.campaign/row["folder"]; protocol = json.loads((folder/"protocol.json").read_text())
        config = protocol["config"]
        input_path = recorded_input(folder, protocol)
        original = qasm2.loads(input_path.read_text()); n = original.num_qubits
        ref = qiskit_matrix_zx_order(original)
        raw = Diagram.from_record(json.loads(gzip.decompress((folder/"objects"/(protocol["root_graph"]+".json.gz")).read_bytes())))
        views = Views(n, ref); view = views.get(raw)
        general_zx = config.get("general_zx", False)
        kernel = GeneralKernel(environment(input_path), config["seed"], config["jpv"], config["candidate_cap"], config["matching_work"], general_zx)
        available = kernel.prepare(raw, view, protocol["root_view_vertices"]+config["max_growth"])
        if config.get("feasibility_mask", False):
            available, _ = Feasibility(protocol["max_vertices"], len(raw.edges)+config["max_edge_growth"]).prepare(raw, view, available, kernel)
        first = json.loads((folder/"events.jsonl").read_text().splitlines()[0])
        assert candidate_digest(available) == first["candidate_sha256"]
        obs = observation(raw, view.graph, available, first["context"]); packed = batch([obs])
        net = GeneralAgent(general_zx=general_zx); values = []; distributions = []
        for checkpoint in ("initial.pt", "final.pt"):
            net.load_state_dict(torch.load(folder/checkpoint, map_location="cpu", weights_only=True)["state_dict"])
            with torch.no_grad():
                scores = net.actor(packed[0]).flatten()[packed[0].y != -1].double().numpy()
                q = np.exp(scores-scores.max()); q /= q.sum()
                distributions.append(q); values.append(float(net.get_value(packed[1])[0]))
        before, after = distributions
        def entropy(q):
            return float(-(q*np.log(np.maximum(q, 1e-300))).sum()/np.log(len(q))) if len(q) > 1 else 0.
        family_before = {}; family_after = {}
        for i, action in enumerate(available):
            family_before[action.family] = family_before.get(action.family, 0.)+float(before[i])
            family_after[action.family] = family_after.get(action.family, 0.)+float(after[i])
        results.append(dict(arm=row["arm"], seed=row["seed"], proposals=len(available),
                            candidate_sha256=first["candidate_sha256"],
                            total_variation=float(np.abs(after-before).sum()/2),
                            kl_initial_to_final=float((before*np.log(np.maximum(before, 1e-300)/np.maximum(after, 1e-300))).sum()),
                            normalized_entropy_initial=entropy(before), normalized_entropy_final=entropy(after),
                            critic_initial=values[0], critic_final=values[1],
                            family_probabilities_initial=family_before, family_probabilities_final=family_after))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(status="passed", fixed_state="original root, original context and exact first proposal list",
                                      proof_limit="Preference change is not a search-benefit claim", records=results), indent=2)+"\n")
    print(json.dumps(dict(status="passed", probes=len(results))))


if __name__ == "__main__":
    main()
