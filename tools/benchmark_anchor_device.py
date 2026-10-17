"""CPU/P800 policy-kernel and host-transfer timing, separate from search.

Fixed-shape replicated graph batches are a hardware diagnostic, not evidence
of search speed or learning quality. Optimizer steps only mutate benchmark
model copies. Every search/circuit equivalence check remains on CPU.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--device", choices=["cpu", "kunlun"], default="cpu")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    a = p.parse_args()
    if a.device == "kunlun":
        import torch_xmlir  # registers the vendor device before any tensors move
        if not torch.cuda.is_available():
            raise RuntimeError("P800 was requested but the vendor accelerator is unavailable")
        name = torch.cuda.get_device_name(0)
        if "P800" not in name:
            raise RuntimeError("Expected P800, found "+name)
    else:
        name = "CPU"
    from zxopt.anchor_policy import AnchorAgent, anchor_observation, batch, catalog, conditional_observation, empty_action
    from zxopt.complete.adapter import from_pyzx
    from zxopt.general_kernel import GeneralKernel, Views
    from zxopt.legacy import environment
    from zxopt.scalable_verify import reference_for
    torch.set_num_threads(1)
    path = ROOT/"inputs/quarl-nam-raw/barenco_tof_3.qasm"
    env = environment(path); raw = from_pyzx(env.graph, allow_arbitrary=True)
    view = Views(len(env.graph.inputs()), reference_for(path.read_text())).get(raw)
    anchors = catalog(raw, view.graph); context = [0., 0., 0., 0., 1., 1., 0., 0.]
    kernel = GeneralKernel(env, general_zx=True, candidate_cap=8, total_candidate_cap=8)
    observations = []
    for i in range(max(a.batch_sizes)):
        slot = (i*31) % len(anchors); anchor = anchors[slot]
        available = kernel.prepare(raw, view, view.graph.num_vertices()+64, anchor.query()) or [empty_action()]
        ap = anchor_observation(raw, view.graph, anchors, context)
        ap.selected = torch.tensor([slot])
        observations.append((ap, *conditional_observation(raw, view.graph, available, context, anchor)))
    torch.manual_seed(11); reference = AnchorAgent(general_zx=True)
    result = dict(status="running", requested=a.device, device_name=name, torch=torch.__version__,
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), batches=[],
                  scope="Fixed graph shapes, kernel/transfer microbenchmark; no search or learning-quality claim")
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(result, indent=2)+"\n")

    def sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize()

    def measure(fn, device):
        for _ in range(a.warmup):
            fn()
        sync(device); values = []
        for _ in range(a.repeats):
            started = time.perf_counter(); fn(); sync(device); values.append(time.perf_counter()-started)
        return dict(median_seconds=statistics.median(values), minimum_seconds=min(values), repeats=len(values))

    save()
    try:
        for size in a.batch_sizes:
            cpu_inputs = batch(observations[:size]); actions = torch.zeros(size, dtype=torch.long)
            with torch.no_grad():
                expected_anchor = reference.actor(cpu_inputs[0]).detach()
                expected_logs = reference.get_action_and_value(cpu_inputs, action=actions)[1].detach()
            for label in (["cpu", "kunlun"] if a.device == "kunlun" else ["cpu"]):
                device = torch.device("cpu" if label == "cpu" else "cuda:0")
                model = copy.deepcopy(reference).to(device)
                inputs = tuple(g.clone().to(device) for g in cpu_inputs)
                acts = actions.to(device)
                with torch.no_grad():
                    anchor = model.actor(inputs[0]); logs = model.get_action_and_value(inputs, action=acts)[1]
                    anchor_error = float((anchor.cpu()-expected_anchor).abs().max())
                    log_error = float((logs.cpu()-expected_logs).abs().max())
                row = dict(batch_size=size, backend=label, policy_nodes=[len(g.x) for g in cpu_inputs],
                           max_anchor_logit_error=anchor_error, max_joint_log_probability_error=log_error)
                result["batches"].append(row); save()
                if not np.isfinite(anchor_error+log_error) or max(anchor_error, log_error) > 2e-4:
                    raise AssertionError("Accelerator forward differs from CPU beyond the diagnostic tolerance")

                def anchor_forward():
                    with torch.no_grad():
                        return model.actor(inputs[0])

                def joint_forward():
                    with torch.no_grad():
                        return model.get_action_and_value(inputs, action=acts)

                def anchor_roundtrip():
                    with torch.no_grad():
                        graph = cpu_inputs[0].clone().to(device)
                        return model.actor(graph).flatten()[graph.y != -1].cpu()

                row["anchor_preloaded"] = measure(anchor_forward, device); save()
                row["joint_preloaded"] = measure(joint_forward, device); save()
                row["anchor_with_transfer"] = measure(anchor_roundtrip, device); save()
                optimizer = torch.optim.Adam(model.parameters(), lr=.0003, eps=1e-5)
                with torch.no_grad():
                    old_logs = model.get_action_and_value(inputs, action=acts)[1].detach()
                advantage = torch.linspace(-1, 1, size).to(device)
                returns = torch.linspace(-.5, .5, size).to(device)

                def train_step():
                    _, lp, entropy, values, _, _ = model.get_action_and_value(inputs, action=acts)
                    ratio = (lp-old_logs).exp()
                    loss = torch.maximum(-advantage*ratio, -advantage*ratio.clamp(.8, 1.2)).mean()
                    loss = loss+.5*(values-returns).square().mean()-.01*entropy.mean()
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .5); optimizer.step()
                    return loss.detach()

                row["optimizer_step_preloaded"] = measure(train_step, device)
                row["parameters_finite"] = all(bool(torch.isfinite(t).all()) for t in model.parameters())
                if not row["parameters_finite"]:
                    raise FloatingPointError("Nonfinite benchmark parameters")
                save()
                del model, inputs, optimizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"; result["failure"] = dict(type=type(exc).__name__, message=str(exc))
        save(); raise
    save(); print(json.dumps(dict(status=result["status"], batches=len(result["batches"]), device=name)))


if __name__ == "__main__":
    main()
