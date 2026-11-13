"""Pure protocol helpers for the stateful SD3.5 ours candidate.

This module deliberately has no diffusers dependency.  It is used by the GPU
runner and by local protocol tests so that the denoising/update budget can be
audited before any model is loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


CORE_MODEL_ORDER = ("fasterrcnn", "ssd300", "monodepth2", "dpt_large")
TRANSFER_MODEL_ORDER = ("maskrcnn", "fcos", "depth_anything_v1", "zoedepth")


@dataclass(frozen=True)
class StatefulOursConfig:
    num_inference_steps: int = 100
    strength: float = 0.25
    outer_rounds: int = 4
    latent_lr: float = 0.05
    step_feedback_mode: str = "latent_chain"
    guidance_scale: float = 4.5
    seed: int = 20260918
    prompt: str = "top-down aerial view of clean asphalt road surface"
    negative_prompt: str = "vehicle, car, obstacle"

    def validate(self) -> None:
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        if self.outer_rounds < 1:
            raise ValueError("outer_rounds must be positive")
        if self.latent_lr <= 0.0:
            raise ValueError("latent_lr must be positive")
        if self.step_feedback_mode not in {"latent_chain", "rgb_reencode"}:
            raise ValueError("unsupported step_feedback_mode")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")

    @property
    def active_steps_per_round(self) -> int:
        self.validate()
        return min(
            int(self.num_inference_steps * self.strength),
            self.num_inference_steps,
        )

    @property
    def adversarial_updates(self) -> int:
        return self.active_steps_per_round * self.outer_rounds

    @property
    def denoise_steps(self) -> int:
        return self.adversarial_updates

    @property
    def noise_samples(self) -> int:
        return self.outer_rounds

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            active_steps_per_round=self.active_steps_per_round,
            adversarial_updates=self.adversarial_updates,
            denoise_steps=self.denoise_steps,
            noise_samples=self.noise_samples,
        )
        return payload


@dataclass(frozen=True)
class PlannedStep:
    global_step: int
    outer_round: int
    inner_step: int
    sample_id: str
    dataset: str
    noise_seed: int
    render_seed: int


@dataclass(frozen=True)
class GradientControl:
    joint_gradient: torch.Tensor
    model_weights: torch.Tensor
    attack_progress: torch.Tensor
    gradient_conflict: torch.Tensor
    cosine_matrix: torch.Tensor
    gradient_norms: torch.Tensor
    task_cosine: torch.Tensor
    task_projection_applied: bool
    weight_fallback: bool
    zero_model_gradient_indices: tuple[int, ...]


def build_execution_plan(
    config: StatefulOursConfig,
    records: Sequence[Any],
) -> tuple[PlannedStep, ...]:
    """Bind every active diffusion step to one frozen train record."""
    config.validate()
    if len(records) < config.adversarial_updates:
        raise ValueError("not enough scheduled train records for the stateful run")
    plan = []
    active = config.active_steps_per_round
    for index, record in enumerate(records[: config.adversarial_updates]):
        split = getattr(record, "split", None)
        if split != "train":
            raise ValueError(f"optimization record is not train: {split!r}")
        outer = index // active + 1
        inner = index % active + 1
        plan.append(
            PlannedStep(
                global_step=index + 1,
                outer_round=outer,
                inner_step=inner,
                sample_id=str(getattr(record, "sample_id")),
                dataset=str(getattr(record, "dataset")),
                noise_seed=config.seed + 1000 * (outer - 1),
                render_seed=config.seed + 100_000 + index + 1,
            )
        )
    return tuple(plan)


def reject_test_input(path: Path, records: Iterable[Any]) -> None:
    """Fail closed if a tuning run is pointed at test data."""
    if "test" in path.name.lower():
        raise ValueError("ours validation runner refuses a test manifest")
    forbidden = sorted(
        {
            str(getattr(record, "sample_id", "<unknown>"))
            for record in records
            if getattr(record, "split", None) == "test"
        }
    )
    if forbidden:
        raise ValueError(f"test rows are forbidden during tuning: {forbidden[:3]}")


def compute_attack_progress(
    losses_before: Sequence[torch.Tensor],
    losses_after: Sequence[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    if len(losses_before) != 4 or len(losses_after) != 4:
        raise ValueError("attack progress requires exactly four losses")
    before = torch.stack([value.detach().float().reshape(()) for value in losses_before])
    after = torch.stack([value.detach().float().reshape(()) for value in losses_after])
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("attack progress losses must be finite")
    if (before < 0).any() or (after < 0).any():
        raise ValueError("attack progress requires non-negative losses")
    return ((before - after) / before.abs().clamp_min(eps)).clamp(0.0, 1.0)


def combine_four_model_gradients(
    gradients: Sequence[torch.Tensor],
    attack_progress: Sequence[torch.Tensor | float],
    eps: float = 1e-8,
) -> GradientControl:
    """AP/GC/task projection with finite individual zero-gradient handling."""
    if len(gradients) != 4 or len(attack_progress) != 4:
        raise ValueError("ours requires exactly four gradients and progress values")
    shape = gradients[0].shape
    device = gradients[0].device
    if any(value.shape != shape or value.device != device for value in gradients):
        raise ValueError("all model gradients must share shape and device")
    flat = torch.stack([value.detach().float().reshape(-1) for value in gradients])
    norms = torch.linalg.vector_norm(flat, dim=1)
    if not torch.isfinite(flat).all() or not torch.isfinite(norms).all():
        raise ValueError("model gradients contain NaN/Inf")
    active = norms > eps
    if not active.any():
        raise ValueError("all four model gradients are zero or near-zero")
    safe_norms = torch.where(active, norms, torch.ones_like(norms))
    unit = flat / safe_norms[:, None]
    cosine = (unit @ unit.T).clamp(-1.0, 1.0)
    negative = torch.relu(-cosine)
    negative.fill_diagonal_(0.0)
    conflict = negative.sum(dim=1) / 3.0
    progress = torch.stack(
        [torch.as_tensor(value, device=device, dtype=torch.float32).reshape(()) for value in attack_progress]
    ).detach()
    if not torch.isfinite(progress).all() or ((progress < 0) | (progress > 1)).any():
        raise ValueError("attack progress must be finite and within [0,1]")
    raw = (1.0 - progress) * (1.0 - conflict) * active.to(progress)
    if raw.sum() <= eps:
        weights = active.to(raw) / active.sum()
        fallback = True
    else:
        weights = raw / raw.sum()
        fallback = False

    detection = sum(weights[index] * unit[index] for index in (0, 1))
    depth = sum(weights[index] * unit[index] for index in (2, 3))
    detection_norm2 = detection.square().sum()
    depth_norm2 = depth.square().sum()
    dot = torch.dot(detection, depth)
    denominator = torch.sqrt(detection_norm2 * depth_norm2)
    task_cosine = (
        dot / denominator.clamp_min(eps)
        if denominator > eps
        else dot.new_zeros(())
    )
    projected = bool(dot < 0 and detection_norm2 > eps and depth_norm2 > eps)
    if projected:
        original_detection = detection
        detection = detection - dot / depth_norm2.clamp_min(eps) * depth
        depth = depth - dot / detection_norm2.clamp_min(eps) * original_detection
    joint = detection + depth
    joint_rms = joint.square().mean().sqrt()
    if not torch.isfinite(joint_rms) or joint_rms <= eps:
        raise ValueError("combined four-model gradient is zero or non-finite")
    return GradientControl(
        joint_gradient=(joint / joint_rms).reshape(shape),
        model_weights=weights,
        attack_progress=progress,
        gradient_conflict=conflict,
        cosine_matrix=cosine,
        gradient_norms=norms,
        task_cosine=task_cosine,
        task_projection_applied=projected,
        weight_fallback=fallback,
        zero_model_gradient_indices=tuple(torch.nonzero(~active).flatten().tolist()),
    )


def texture_diagnostics(rgb: torch.Tensor) -> dict[str, float]:
    """Record texture quality without using it to select the first candidate."""
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [1,3,H,W]")
    if not torch.isfinite(rgb).all():
        raise ValueError("rgb contains non-finite values")
    horizontal = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean()
    vertical = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean()
    saturation = ((rgb <= 1e-4) | (rgb >= 1.0 - 1e-4)).float().mean()
    return {
        "rgb_min": float(rgb.min().detach()),
        "rgb_max": float(rgb.max().detach()),
        "total_variation_l1": float((horizontal + vertical).detach()),
        "saturated_fraction": float(saturation.detach()),
    }


def audit_trace(rows: Sequence[dict[str, Any]], config: StatefulOursConfig) -> None:
    if len(rows) != config.adversarial_updates:
        raise ValueError(
            f"trace row count mismatch: {len(rows)} != {config.adversarial_updates}"
        )
    expected = list(range(1, config.adversarial_updates + 1))
    actual = [int(row["global_step"]) for row in rows]
    if actual != expected:
        raise ValueError("trace global_step values are not contiguous")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("trace contains a non-train optimization row")
    if any(int(row.get("model_count", 0)) != 4 for row in rows):
        raise ValueError("every update must contain four core model losses")
