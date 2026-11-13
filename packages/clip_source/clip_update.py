"""Candidate-only L2 clipping of the *amplified total* latent gradient.

Order: cast -> coefficient multiplication -> optional global L2 clipping ->
learning-rate multiplication -> subtraction.  Off/inactive clipping takes the
historical arithmetic path without a multiply-by-one.  Active clipping computes
its norm and scale in FP32, then casts the scaled gradient back to latent dtype;
BF16 rounding can put the measured post-cast norm slightly above the requested
threshold.  This is not a strict mathematical bound on the realized update.

This module does not change the diffusion chain, task weighting, or RGB bounds.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import torch


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, not a tensor or bool")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _threshold(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("max_norm must be None or a real scalar, not a tensor or bool")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    return result


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a nonfinite value")


def _norm(value: torch.Tensor, name: str) -> torch.Tensor:
    """Global norm, with FP32 accumulation, never a per-channel norm."""
    result = torch.linalg.vector_norm(value.detach().float().reshape(-1), ord=2)
    _finite(result, name)
    return result


def clipped_scaled_update(
    latent: torch.Tensor,
    gradient: torch.Tensor,
    coefficient: float = 100000.0,
    lr: float = 0.05,
    *,
    max_norm: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Return an independent candidate, detached diagnostics, and JSON metadata.

    ``gradient`` must already be the gradient of the combined scalar loss.
    ``max_norm`` applies after coefficient multiplication.  ``None`` disables
    clipping; zero removes only the adversarial gradient update, not denoising.
    Neither inputs nor RNG state are modified.  Invalid/nonfinite arithmetic
    raises rather than silently replacing values or making a partial update.

    ``realized_delta_float32`` uses the historical subtraction-update sign:
    latent.float() - candidate.float().  It is a reporting tensor only.
    """
    for name, value in (("latent", latent), ("gradient", gradient)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided or not value.is_floating_point():
            raise TypeError(f"{name} must be a dense, real floating-point tensor")
        if value.device.type == "meta":
            raise ValueError(f"{name} must have materialized storage")
        if value.numel() == 0:
            raise ValueError(f"{name} must not be empty")
    if latent.shape != gradient.shape:
        raise ValueError("latent and gradient shapes must match")
    if latent.device != gradient.device:
        raise ValueError("latent and gradient devices must match")
    for name, value in (("latent", latent), ("gradient", gradient)):
        _finite(value, name)
    scale = _positive_scalar(coefficient, "coefficient")
    learning_rate = _positive_scalar(lr, "lr")
    threshold = _threshold(max_norm)

    with torch.no_grad():
        cast = gradient.detach().to(latent)
        _finite(cast, "cast_gradient")
        amplified = scale * cast
        _finite(amplified, "amplified_gradient")
        norm_before = _norm(amplified, "norm_before")
        # A nonzero BF16/FP32 vector can have a zero FP32 norm after underflow.
        # The zero endpoint must still remove every adversarial update.
        zero_endpoint = threshold == 0.0 and bool(torch.count_nonzero(amplified).item())
        clipped = zero_endpoint or (threshold is not None and float(norm_before.item()) > threshold)
        clip_scale = torch.ones((), device=latent.device, dtype=torch.float32)
        if clipped:
            if threshold == 0.0:
                clip_scale = torch.zeros((), device=latent.device, dtype=torch.float32)
            else:
                # No epsilon is needed: the positive-cap active branch has norm > 0.
                clip_scale = torch.tensor(threshold, device=latent.device, dtype=torch.float32) / norm_before
            _finite(clip_scale, "clip_scale")
            used_gradient = (amplified.float() * clip_scale).to(latent.dtype)
            _finite(used_gradient, "clipped_gradient")
        else:
            # Exact historical path: do NOT multiply amplified by one here.
            used_gradient = amplified
        norm_after = _norm(used_gradient, "norm_after")
        proposed = learning_rate * used_gradient
        _finite(proposed, "proposed_update")
        candidate = (latent.detach() - proposed).detach()
        _finite(candidate, "candidate_latent")
        realized = latent.detach().float() - candidate.float()
        _finite(realized, "realized_delta_float32")
        changed = int(torch.count_nonzero(candidate != latent.detach()).item())
        count = latent.numel()
        root_count = math.sqrt(count)
        latent_rms = float((_norm(latent, "latent_norm") / root_count).item())
        proposed_rms = float((_norm(proposed, "proposed_update_norm") / root_count).item())
        realized_rms = float((_norm(realized, "realized_update_norm") / root_count).item())

    metadata: dict[str, Any] = {
        "coefficient": scale,
        "lr": learning_rate,
        "latent_dtype": str(latent.dtype),
        "input_gradient_dtype": str(gradient.dtype),
        "arithmetic_dtype": str(latent.dtype),
        "norm_accumulation_dtype": "torch.float32",
        "clip_scale_dtype": "torch.float32",
        "device": str(latent.device),
        "shape": list(latent.shape),
        "numel": count,
        "changed_count": changed,
        "unchanged_count": count - changed,
        "changed_fraction": changed / count,
        "unchanged_fraction": (count - changed) / count,
        "max_norm": threshold,
        "norm_before": float(norm_before.item()),
        "norm_after": float(norm_after.item()),
        "clip_scale": float(clip_scale.item()),
        "clipped": bool(clipped),
        "latent_rms": latent_rms,
        "proposed_update_rms": proposed_rms,
        "realized_update_rms": realized_rms,
        "realized_to_latent_rms_ratio": realized_rms / latent_rms if latent_rms > 0 else None,
        "clipping": "off" if threshold is None else "global_l2",
        "norm_scope": "all_elements_of_combined_amplified_latent_gradient",
        "clip_stage": "after_coefficient_before_lr",
        "arithmetic_order": [
            "cast = gradient.detach().to(latent)",
            "amplified = coefficient * cast",
            "used_gradient = amplified if off/inactive else (amplified.float() * clip_scale).to(latent.dtype)",
            "proposed = lr * used_gradient",
            "candidate = (latent.detach() - proposed).detach()",
        ],
        "realized_delta_definition": "latent.float() - candidate.float()",
        "normalization": "none",
        "clamp": "none",
        "post_cast_threshold_note": "low_precision_rounding_may_slightly_exceed_max_norm",
        "inputs_modified": False,
        "independent_one_step": True,
    }
    tensors = {
        "cast_gradient": cast,
        "amplified_gradient": amplified,
        "clipped_gradient": used_gradient,
        "proposed_update": proposed,
        "candidate_latent": candidate,
        "realized_delta_float32": realized,
    }
    return candidate, tensors, metadata
