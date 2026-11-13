"""Pure detector-only control and trace checks for the independent v16 candidate."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch

DETECTION_ORDER = ("fasterrcnn", "ssd300")
DUAL_ORDER = (*DETECTION_ORDER, "monodepth2", "dpt_large")
CANDIDATE_PREFIX = "normalized-v17-roadpatch-20260920-ours-"


def model_order(task_variant):
    if task_variant == "det-only":
        return DETECTION_ORDER
    if task_variant == "dual":
        return DUAL_ORDER
    raise ValueError("unsupported ours task variant")


@dataclass(frozen=True)
class DetectorControl:
    joint_gradient: torch.Tensor
    gradient_norms: torch.Tensor
    model_weights: torch.Tensor
    zero_model_gradient_indices: tuple


def combine_detector_gradients(gradients: Sequence[torch.Tensor], eps=1e-8):
    """Equal mean of active per-model L2-unit gradients, then latent RMS scaling.

    Detector-only has no artificial depth slots, AP weights or task projection.
    A missing detection may give one zero gradient; both zero fails explicitly.
    """
    if len(gradients) != 2:
        raise ValueError("detector-only control requires exactly two gradients")
    shape, device = gradients[0].shape, gradients[0].device
    if any(g.shape != shape or g.device != device for g in gradients):
        raise ValueError("gradients must share shape and device")
    flat = torch.stack([g.detach().float().reshape(-1) for g in gradients])
    norms = torch.linalg.vector_norm(flat, dim=1)
    if not torch.isfinite(flat).all() or not torch.isfinite(norms).all():
        raise ValueError("nonfinite detector gradients")
    active = norms > eps
    if not active.any():
        raise ValueError("all detector gradients are zero or near-zero")
    unit = flat / torch.where(active, norms, torch.ones_like(norms))[:, None]
    weights = active.to(flat) / active.sum()
    joint = (weights[:, None] * unit).sum(0)
    rms = joint.square().mean().sqrt()
    if not torch.isfinite(rms) or rms <= eps:
        raise ValueError("combined detector gradient cancels or is near-zero")
    return DetectorControl((joint / rms).reshape(shape), norms, weights,
                           tuple(torch.nonzero(~active).flatten().tolist()))


def detector_progress(before, after, eps=1e-8):
    if len(before) != 2 or len(after) != 2:
        raise ValueError("detector-only progress requires two losses")
    a, b = [torch.stack([v.detach().float().reshape(()) for v in values])
            for values in (before, after)]
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("progress losses must be finite and nonnegative")
    return ((a - b) / a.abs().clamp_min(eps)).clamp(0, 1)


def audit_candidate_trace(rows, config, task_variant):
    order = model_order(task_variant)
    if len(rows) != config.adversarial_updates:
        raise ValueError("trace update count differs from the candidate budget")
    if [row.get("global_step") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("trace steps must be contiguous")
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("optimization trace contains a non-train row")
        if row.get("model_count") != len(order) or tuple(row.get("training_models", [])) != order:
            raise ValueError("trace model slots differ from actual task variant")
        if set(row["losses_pre"]) != set(order) or set(row["losses_post"]) != set(order):
            raise ValueError("trace contains missing or extra model losses")
        if len(row["gradient_norms"]) != len(order):
            raise ValueError("trace gradient count differs from model count")
        if row.get("task_variant") != task_variant:
            raise ValueError("trace mixes task variants")
        if task_variant == "det-only" and (row.get("task_projection_applied") or row.get("task_cosine") is not None):
            raise ValueError("detector-only trace contains artificial depth-task control")


def depth_asset_checks(order, manifest):
    """Verify only selected depth assets; detector-only never inspects depth files."""
    import json
    from pathlib import Path
    from road_runtime import checked_file
    selected = [x for x in order if x in ("monodepth2", "dpt_large")]
    if not selected:
        return {"inputs_ready": True, "checks": [], "applicable": False}
    definitions = {x["id"]: x for x in json.loads(Path(manifest).read_text())["models"]}
    checks = []
    for model_id in selected:
        entry = definitions[model_id]
        if not entry.get("artifacts"):
            raise ValueError("depth model assets are unbound: " + model_id)
        for artifact in entry["artifacts"]:
            checks.append(dict(checked_file(artifact["path"], artifact["sha256"]), model_id=model_id))
        if model_id == "monodepth2":
            directory = Path(entry["repository"]["path"])
            checks.append({"path": str(directory), "model_id": model_id,
                           "status": "pass" if (directory / "networks").is_dir() else "missing",
                           "role": "local source directory; revision must be verified before formal freeze"})
    return {"inputs_ready": all(x["status"] == "pass" for x in checks), "checks": checks, "applicable": True}


def require_candidate_run_path(path, root):
    from pathlib import Path
    path = Path(path).resolve()
    allowed = (Path(root) / "runs").resolve()
    try:
        relative = path.relative_to(allowed)
    except ValueError as error:
        raise ValueError("v16 runtime outputs/inputs must stay in this candidate runs directory") from error
    if not relative.parts:
        raise ValueError("use a named run subdirectory, never runs itself")
    return path


def verify_patch_bindings(run_directory, summary):
    from pathlib import Path
    from data import sha256
    bindings = summary.get("patch_bindings", {})
    for name in ("zero_step_patch.pt", "final_patch.pt"):
        if not bindings.get(name) or sha256(Path(run_directory) / name) != bindings[name]:
            raise ValueError("candidate patch hash binding failed: " + name)
