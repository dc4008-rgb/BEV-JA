"""Independent one-step gradient multiplier diagnostic, without clipping/scaling norms.

The operation order is deliberately explicit: cast to latent dtype, multiply by
the chosen coefficient, multiply by the learning rate, then subtract. In BF16
each of these tensor operations produces BF16 output; a combined multiplier or
an FP32 master latent would be a different experiment.
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


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a nonfinite value")


def scaled_update(
    latent: torch.Tensor,
    gradient: torch.Tensor,
    coefficient: float,
    lr: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Return an independent candidate and raw tensors; never modify inputs.

    ``realized_delta_float32`` is ``latent.float() - candidate.float()`` and
    therefore has the same subtraction-update sign convention as ``proposed``.
    It is a reporting tensor only; it does not participate in the update.
    """
    for name, value in (("latent", latent), ("gradient", gradient)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided or not value.is_floating_point():
            raise TypeError(f"{name} must be a dense, real floating-point tensor")
        if value.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        _finite(value, name)
    if latent.shape != gradient.shape:
        raise ValueError("latent and gradient shapes must match")
    if latent.device != gradient.device:
        raise ValueError("latent and gradient devices must match")
    scale = _positive_scalar(coefficient, "coefficient")
    learning_rate = _positive_scalar(lr, "lr")

    with torch.no_grad():
        cast = gradient.detach().to(latent)
        _finite(cast, "cast_gradient")
        amplified = scale * cast
        _finite(amplified, "amplified_gradient")
        proposed = learning_rate * amplified
        _finite(proposed, "proposed_update")
        candidate = (latent.detach() - proposed).detach()
        _finite(candidate, "candidate_latent")
        realized = latent.detach().float() - candidate.float()
        _finite(realized, "realized_delta_float32")
        changed = int(torch.count_nonzero(candidate != latent.detach()).item())

    count = latent.numel()
    metadata: dict[str, Any] = {
        "coefficient": scale,
        "lr": learning_rate,
        "latent_dtype": str(latent.dtype),
        "input_gradient_dtype": str(gradient.dtype),
        "arithmetic_dtype": str(latent.dtype),
        "device": str(latent.device),
        "shape": list(latent.shape),
        "numel": count,
        "changed_count": changed,
        "unchanged_count": count - changed,
        "changed_fraction": changed / count,
        "unchanged_fraction": (count - changed) / count,
        "arithmetic_order": [
            "cast = gradient.detach().to(latent)",
            "amplified = coefficient * cast",
            "proposed = lr * amplified",
            "candidate = (latent.detach() - proposed).detach()",
        ],
        "realized_delta_definition": "latent.float() - candidate.float()",
        "normalization": "none",
        "clamp": "none",
        "inputs_modified": False,
        "independent_one_step": True,
    }
    tensors = {
        "amplified_gradient": amplified,
        "proposed_update": proposed,
        "candidate_latent": candidate,
        "realized_delta_float32": realized,
    }
    return candidate, tensors, metadata
