"""Loss-space adaptive weighting for the September 23 meeting revision.

This module only weights the two raw, nonnegative detector losses.  It performs
no loss calibration, min-max transform, per-model gradient normalization, or
joint-gradient normalization.  Call ``torch.autograd.grad`` once on the returned
weighted loss to obtain the unnormalized joint gradient.

The FOA-style rule used here is ``r = current / (previous + epsilon)`` and
``w = N * softmax(r / temperature)``, with N = 2.  The factor N is deliberate:
equal weights are [1, 1], not [0.5, 0.5].  Weight computation is detached, so it
does not introduce an additional derivative through the loss-ratio controller.

The caller owns the meaning of current/previous.  In a temporal experiment they
may be successive observations.  The no-update diagnostic pilot instead uses
paired saved before/after states with the same scene, target and EOT draws,
re-evaluated under the new loss.  That paired diagnostic is an adaptation, not a
claim to have reproduced the source algorithm's complete optimization loop.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor


def _loss_vector(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool((value < 0).any()):
        raise ValueError(f"{name} contains negative losses")


def foa_weights(
    current: Tensor,
    previous: Tensor | None,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[Tensor, dict[str, Any]]:
    """Return detached weights summing to two and JSON-compatible diagnostics.

    Missing history produces ones.  Existing history always uses the displayed
    epsilon rule, including zeros: 0/(0+eps) is zero and x/(0+eps) can be large.
    Zero-history events are explicitly flagged rather than assigned a fabricated
    ratio or hidden behind an active-model threshold.  Finite negative losses
    and non-finite inputs are rejected.  Non-finite ratios raise rather than
    silently clipping them or changing the formula.

    Computation preserves float32/float64 inputs; float16/bfloat16 inputs are
    promoted to float32 so the specified epsilon does not underflow in float16.
    This dtype promotion is reported and is not a gradient normalization.
    """
    _loss_vector(current, "current")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    if previous is not None:
        _loss_vector(previous, "previous")
        if previous.device != current.device:
            raise ValueError("current and previous must be on the same device")
        if previous.dtype != current.dtype:
            raise ValueError("current and previous must have the same dtype")

    work_dtype = (
        current.dtype if current.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    with torch.no_grad():
        now = current.detach().to(dtype=work_dtype)
        before = None if previous is None else previous.detach().to(dtype=work_dtype)
        if before is None:
            ratio = None
            weights = torch.ones_like(now)
        else:
            ratio = now / (before + epsilon)
            logits = ratio / temperature
            if not bool(torch.isfinite(ratio).all() and torch.isfinite(logits).all()):
                raise FloatingPointError("loss ratios or temperature-scaled ratios are non-finite")
            weights = 2.0 * torch.softmax(logits, dim=0)
        if not bool(torch.isfinite(weights).all()):
            raise FloatingPointError("FOA weights are non-finite")
        diagnostics: dict[str, Any] = {
            "rule": "2 * softmax((current / (previous + epsilon)) / temperature)",
            "initial_equal_weights": before is None,
            "temperature": float(temperature),
            "epsilon": float(epsilon),
            "current": now.cpu().tolist(),
            "previous": None if before is None else before.cpu().tolist(),
            "ratios": None if ratio is None else ratio.cpu().tolist(),
            "weights": weights.cpu().tolist(),
            "weight_sum": float(weights.sum().item()),
            "weight_sum_target": 2.0,
            "zero_current": (now == 0).cpu().tolist(),
            "zero_previous": None if before is None else (before == 0).cpu().tolist(),
            "all_current_zero": bool((now == 0).all()),
            "all_previous_zero": None if before is None else bool((before == 0).all()),
            "input_dtype": str(current.dtype),
            "calculation_dtype": str(work_dtype),
            "weights_detached": True,
            "loss_or_gradient_normalization": "none",
        }
    return weights.detach(), diagnostics


def weighted_loss(losses: Sequence[Tensor], weights: Tensor) -> Tensor:
    """Sum two scalar raw losses using fixed weights; never divide by their sum.

    The controller weights are detached even if a caller supplies a tensor that
    requires gradients.  The returned scalar retains the two original loss
    graphs.  No model is omitted for having a zero loss/gradient, and this helper
    neither computes nor clips gradients or updates latent variables.
    """
    if len(losses) != 2:
        raise ValueError("exactly two detector losses are required")
    for loss in losses:
        if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.is_floating_point():
            raise TypeError("each loss must be a scalar floating point tensor")
    if any(loss.device != losses[0].device or loss.dtype != losses[0].dtype for loss in losses):
        raise ValueError("both losses must have the same device and dtype")
    vector = torch.stack(tuple(losses))
    _loss_vector(vector, "losses")
    _loss_vector(weights, "weights")
    if weights.device != vector.device:
        raise ValueError("weights and losses must be on the same device")
    fixed_weights = weights.detach().to(dtype=vector.dtype)
    return (vector * fixed_weights).sum()
