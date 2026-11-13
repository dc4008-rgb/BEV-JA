"""Unintegrated meeting components: explicit-temperature loss and raw gradient export.

This file does not choose normalization, adaptive weights, EOT ranges, or a clamp.
It does not execute training. Historical sealed packages must remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def soft_score_iou_loss(scores, overlaps, *, temperature):
    """Meeting image 2: tau * logsumexp(score * IoU / tau), all supplied candidates.

    The caller must supply the existing detector's selected vehicle candidates.
    No top-k selection or candidate-count normalization is silently introduced.
    Empty candidates yield graph-connected zero where input tensors are connected;
    the caller must separately report this event (not claim successful attack).
    """
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("an explicit finite positive temperature is required")
    if scores.ndim != 1 or scores.shape != overlaps.shape:
        raise ValueError("scores and overlaps must be aligned one-dimensional tensors")
    if scores.device != overlaps.device or scores.dtype != overlaps.dtype:
        raise ValueError("scores and overlaps must share dtype and device")
    if not scores.is_floating_point() or not overlaps.is_floating_point():
        raise ValueError("floating point inputs required")
    if not torch.isfinite(scores).all() or not torch.isfinite(overlaps).all():
        raise ValueError("nonfinite candidate values")
    if not scores.numel():
        return scores.sum() * 0.0 + overlaps.sum() * 0.0
    return temperature * torch.logsumexp(scores * overlaps / temperature, dim=0)


def tensor_statistics(value):
    """Read-only CPU float64 summaries; the separately saved tensor is not rounded."""
    raw = value.detach().cpu().contiguous()
    if not raw.is_floating_point() or not raw.numel():
        raise ValueError("a nonempty floating point gradient/update tensor is required")
    x = raw.double().reshape(-1)
    finite = torch.isfinite(x)
    result = {
        "dtype": str(raw.dtype), "shape": list(raw.shape), "numel": raw.numel(),
        "raw_bytes_sha256": hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        "finite_count": int(finite.sum()), "nan_count": int(torch.isnan(x).sum()),
        "positive_inf_count": int(torch.isposinf(x).sum()),
        "negative_inf_count": int(torch.isneginf(x).sum()),
        "exact_zero_count": int((x == 0).sum()),
        "all_finite": bool(finite.all()), "all_exact_zero": bool((x == 0).all()),
        "negative_finite_count": int(((x < 0) & finite).sum()),
        "positive_finite_count": int(((x > 0) & finite).sum()),
        "statistics_scope": "finite_components_only",
    }
    finite_x = x[finite]
    result["finite_component_statistics"] = None
    if finite_x.numel():
        quantiles = torch.tensor([0.0, .01, .1, .5, .9, .95, .99, .999, 1.0], dtype=torch.float64)
        names = ["p0", "p1", "p10", "p50", "p90", "p95", "p99", "p99_9", "p100"]
        norm = torch.linalg.vector_norm(finite_x)
        result["finite_component_statistics"] = {
            "mean": float(finite_x.mean()), "std_population": float(finite_x.std(correction=0)),
            "l2": float(norm), "rms": float(norm / math.sqrt(finite_x.numel())),
            "signed_quantiles": dict(zip(names, torch.quantile(finite_x, quantiles).tolist())),
            "absolute_quantiles": dict(zip(names, torch.quantile(finite_x.abs(), quantiles).tolist())),
        }
    return result


def export_gradient_snapshot(output, tensors, *, metadata):
    """Export named stages unchanged; never perform an update, normalize, or clip.

    Meeting diagnostic stages: raw/reference losses, detached adaptive weights,
    weighted scalar loss, latent, its raw weighted-loss gradient, dtype-cast
    gradient and lr-scaled proposed update before any clamp.
    """
    if not tensors or any(not isinstance(k, str) for k in tensors):
        raise ValueError("nonempty string-keyed tensor mapping required")
    # Validate JSON before creating output; preserve native tensor dtype in PT.
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    raw = {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}
    report = {
        "status": "gradient_snapshot_only", "metadata": metadata,
        "no_clamp_selected": True, "no_tensor_rounding": True,
        "tensors": {name: tensor_statistics(tensor) for name, tensor in raw.items()},
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    torch.save(raw, path / "raw_tensors.pt")
    (path / "statistics.json").write_text(encoded, encoding="utf-8")
    return report
