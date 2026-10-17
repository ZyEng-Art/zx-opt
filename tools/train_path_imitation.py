"""Distill reverified successful same-circuit actions into the existing actor.

This is supervised demonstration learning, not PPO or a generalization test.
Critic parameters and the search objective remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--max-updates", type=int, default=6000)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--minibatch", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--seed", type=int, default=1221)
    p.add_argument("--allow-development-dataset", action="store_true")
    a = p.parse_args()
    if min(a.max_updates, a.seconds, a.minibatch, a.learning_rate) <= 0:
        raise ValueError("Positive imitation budgets required")
    import torch
    from zxopt.anchor_policy import AnchorAgent, batch
    from zxopt.model_snapshot import state_dict_digest
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    metadata_path = a.dataset/"demonstration.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "demonstration_path_reverified"
    if metadata["development_unaudited_source"] and not a.allow_development_dataset:
        raise ValueError("Whole source audit pending; explicitly request development training")
    assert sha(a.dataset/"dataset.pt") == metadata["dataset_sha256"]
    assert sha(a.dataset/"initial-model.pt") == metadata["initial_model_sha256"]
    # Only our hash-bound local PyG observation capsule is loaded as objects.
    data = torch.load(a.dataset/"dataset.pt", map_location="cpu", weights_only=False)
    observations, targets = data["observations"], torch.tensor(data["actions"], dtype=torch.long)
    assert len(observations) == len(targets) == metadata["examples"] > 0
    initial = torch.load(a.dataset/"initial-model.pt", map_location="cpu", weights_only=True)
    model = AnchorAgent(general_zx=metadata["config"]["general_zx"])
    model.load_state_dict(initial["state_dict"])
    before = {k: value.clone() for k, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.actor_gnn.parameters(), lr=a.learning_rate, eps=1e-5)
    generator = torch.Generator().manual_seed(a.seed)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/"initial-model.pt").write_bytes((a.dataset/"initial-model.pt").read_bytes())
    started = time.perf_counter(); deadline = started+a.seconds
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    protocol = dict(source_commit=revision, trainer_sha256=sha(Path(__file__)),
                    demonstration_sha256=sha(metadata_path), dataset_sha256=metadata["dataset_sha256"],
                    initial_model_sha256=sha(out/"initial-model.pt"), examples=len(targets),
                    max_updates=a.max_updates, seconds=a.seconds, minibatch=a.minibatch,
                    learning_rate=a.learning_rate, seed=a.seed,
                    development_dataset=metadata["development_unaudited_source"],
                    scope="Successful-path supervised joint anchor/action likelihood, actor only. Training-set diagnostics only; later model-only fresh-pool frozen searches determine utility. This does not validate the original PPO.")
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")

    def terms(indices):
        aa, pp, _ = batch([observations[i] for i in indices.tolist()])
        la, _ = model.distribution(aa, aa.selected)
        lp, _ = model.distribution(pp, targets[indices])
        return la+lp, aa, pp

    def accuracy(graph, choices):
        scores = model.actor(graph).flatten(); correct = 0
        for i, choice in enumerate(choices):
            positions = ((graph.y != -1) & (graph.batch == i)).nonzero(as_tuple=True)[0]
            correct += int(int(scores[positions].argmax()) == int(choice))
        return correct

    @torch.no_grad()
    def evaluate(step):
        nll = anchors = rules = 0.
        for start in range(0, len(targets), a.minibatch):
            ix = torch.arange(start, min(start+a.minibatch, len(targets)))
            logs, aa, pp = terms(ix)
            nll -= float(logs.sum()); anchors += accuracy(aa, aa.selected); rules += accuracy(pp, targets[ix])
        return dict(update=step, seconds=time.perf_counter()-started, joint_nll=nll/len(targets),
                    anchor_accuracy=anchors/len(targets), local_action_accuracy=rules/len(targets))

    first = evaluate(0); last = first; updates = 0; successful_checks = 0
    with (out/"updates.jsonl").open("x", buffering=1) as stream:
        stream.write(json.dumps(first)+"\n"); print(json.dumps(first), flush=True)
        while updates < a.max_updates and time.perf_counter() < deadline:
            ix = torch.randperm(len(targets), generator=generator)[:min(a.minibatch, len(targets))]
            logs, _, _ = terms(ix); loss = -logs.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite imitation loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actor_gnn.parameters(), .5)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite imitation gradient")
            optimizer.step(); updates += 1
            if updates % 200 == 0:
                last = evaluate(updates); stream.write(json.dumps(last)+"\n"); print(json.dumps(last), flush=True)
                successful_checks = successful_checks+1 if last["joint_nll"] < .05 and last["anchor_accuracy"] >= .99 and last["local_action_accuracy"] >= .99 else 0
                if successful_checks >= 2:
                    break
        if last["update"] != updates:
            last = evaluate(updates); stream.write(json.dumps(last)+"\n"); print(json.dumps(last), flush=True)
    values = model.state_dict()
    assert all(bool(torch.isfinite(value).all()) for value in values.values())
    critic_keys = [key for key in before if key.startswith(("critic_encoder.", "value_head."))]
    assert critic_keys and all(torch.equal(before[key], values[key]) for key in critic_keys)
    actor_delta = float(sum((value-before[key]).double().square().sum()
                            for key, value in values.items() if key.startswith(("actor_encoder.", "actor_head."))).sqrt())
    assert updates > 0 and actor_delta > 0
    package = {k: v for k, v in initial.items() if k != "state_dict"}
    package.update(state_dict=values, steps=updates*a.minibatch, updates=updates,
                   optimizer_state_dict=optimizer.state_dict(),
                   probe_origin=dict(stage="path_imitation", training_protocol_sha256=sha(out/"protocol.json"),
                                     demonstration_sha256=sha(metadata_path), source_initial_sha256=sha(out/"initial-model.pt"),
                                     development_dataset=metadata["development_unaudited_source"]))
    torch.save(package, out/"trained-model.pt")
    result = dict(status="imitation_training_complete", updates=updates,
                  first=first, final=last, actor_l2_delta=actor_delta, critic_tensors_bitwise_unchanged=True,
                  initial_tensor_sha256=state_dict_digest(before), final_tensor_sha256=state_dict_digest(values),
                  trained_model_sha256=sha(out/"trained-model.pt"), initial_model_sha256=sha(out/"initial-model.pt"),
                  demonstration_sha256=sha(metadata_path), development_dataset=metadata["development_unaudited_source"],
                  scope="Training-set fit only; no utility/convergence/generalization claim without separate frozen initial/imitated model searches from the original circuit and fresh pools.")
    (out/"report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
