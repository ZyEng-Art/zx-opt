"""Portable provenance for trusted same-target model continuations and probes."""
import hashlib
import io

import torch


def state_dict_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_model_snapshot(path, input_sha256, selection, anchor_prior, anchor_resource_mask, neutral_stop_steps=0, rule_profile="full", rule_schedule="static", native_steps=256, expanded_steps=64, candidate_allocation="balanced"):
    """Hash and decode exactly the same bytes, before any state is installed."""
    payload = path.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if saved.get("input_sha256") != input_sha256:
        raise ValueError("Resume model input mismatch")
    if saved.get("selection", "global") != selection:
        raise ValueError("Resume model uses a different action factorization")
    if saved.get("anchor_prior", "uniform") != anchor_prior:
        raise ValueError("Resume model uses a different anchor prior")
    if saved.get("anchor_resource_mask", False) != anchor_resource_mask:
        raise ValueError("Resume model uses a different anchor resource mask")
    if saved.get("neutral_stop_steps", 0) != neutral_stop_steps:
        raise ValueError("Resume model uses a different neutral-stop context")
    if saved.get("rule_profile", "full") != rule_profile:
        raise ValueError("Resume model uses a different rule profile")
    for key, default, value in (("rule_schedule", "static", rule_schedule),
                                ("native_steps", 256, native_steps),
                                ("expanded_steps", 64, expanded_steps),
                                ("candidate_allocation", "balanced", candidate_allocation)):
        if saved.get(key, default) != value:
            raise ValueError("Resume model uses a different rule allocation: "+key)
    if not saved.get("state_dict") or not all(
            isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            for value in saved["state_dict"].values()):
        raise ValueError("Resume model has missing or non-finite weights")
    info = dict(sha256=hashlib.sha256(payload).hexdigest(), input_sha256=input_sha256,
                tensor_sha256=state_dict_digest(saved["state_dict"]),
                steps=saved.get("steps"), updates=saved.get("updates"),
                snapshot="resume-model.pt", original_path=str(path.resolve()))
    if "probe_origin" in saved:
        info["probe_origin"] = saved["probe_origin"]
    return saved, payload, info
