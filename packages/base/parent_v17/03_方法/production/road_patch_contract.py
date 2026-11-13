"""One road RGB contract for v16 training, serialization and evaluation.

All methods project a float32 64x128 RGB patch. Generator-native images are
optional provenance only, never an alternative renderer input after loading.
Only patch brightness can be transformed; scene pixels and geometry are fixed.
This is a project road adaptation, not a 3D source-method reproduction.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from advrm_insertion import SCENE_HEIGHT, SCENE_WIDTH
from carriers import BaseCarrier, CarrierRender, SceneContext, perspective_project, road_patch_destination

CANONICAL_HW = (64, 128)
PATCH_SCHEMA = "road_patch_rgb_v1"
RENDERER_CONTRACT = {
    "canonical_hw": list(CANONICAL_HW),
    "dtype": "float32",
    "range": [0.0, 1.0],
    "canonical_resize": "bilinear_align_corners_false",
    "projection": "inverse_homography_bilinear_align_corners_true",
    "geometry": {"top_fraction": 0.58, "bottom_fraction": 0.92, "width_fraction": 0.72},
    "support": "any_alpha_gt_zero",
    "carrier": "road_only",
}


def _tensor_digest(rgb: torch.Tensor) -> str:
    cpu = rgb.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(cpu.dtype), "shape": list(cpu.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _validate_rgb(rgb: torch.Tensor, *, canonical: bool = False) -> None:
    if not isinstance(rgb, torch.Tensor) or not rgb.is_floating_point():
        raise ValueError("RGB must be an explicitly normalized floating tensor")
    if rgb.ndim != 4 or tuple(rgb.shape[:2]) != (1, 3) or min(rgb.shape[-2:]) < 1:
        raise ValueError("RGB shape must be [1,3,H,W] with nonempty dimensions")
    if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError("RGB values must be finite and in [0,1]; no implicit clamp or rescaling")
    if canonical and (tuple(rgb.shape[-2:]) != CANONICAL_HW or rgb.dtype != torch.float32):
        raise ValueError("serialized canonical_rgb must be float32 [1,3,64,128]")


def canonicalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Canonicalize once without detaching the attack gradient or scaling values."""
    _validate_rgb(rgb)
    rgb = rgb.to(dtype=torch.float32)
    if tuple(rgb.shape[-2:]) == CANONICAL_HW:
        return rgb
    return F.interpolate(rgb, size=CANONICAL_HW, mode="bilinear", align_corners=False)


def sample_patch_brightness(
    generator: torch.Generator, bounds: tuple[float, float] = (0.9, 1.1)
) -> float:
    """One explicit RNG draw; reuse the resulting factor across victim models."""
    if generator is None or len(bounds) != 2:
        raise ValueError("brightness EOT requires a generator and two bounds")
    low, high = map(float, bounds)
    if not math.isfinite(low) or not math.isfinite(high) or not 0 < low <= high:
        raise ValueError("brightness bounds must be finite, positive and ordered")
    draw = float(torch.rand((), generator=generator, device=generator.device))
    return low + (high - low) * draw


def apply_patch_brightness(rgb: torch.Tensor, factor: float) -> torch.Tensor:
    """Resize then transform patch RGB only; no scene/vehicle photometric EOT."""
    if not math.isfinite(float(factor)) or factor <= 0:
        raise ValueError("brightness factor must be positive and finite")
    return (canonicalize_rgb(rgb) * factor).clamp(0.0, 1.0)


def _validate_context(context: SceneContext, rgb: torch.Tensor) -> torch.Tensor:
    _validate_rgb(context.benign)
    if context.benign.shape != (1, 3, SCENE_HEIGHT, SCENE_WIDTH):
        raise ValueError("road contract requires a 320x1024 benign scene")
    if context.benign.device != rgb.device or context.benign.dtype != torch.float32:
        raise ValueError("benign scene must be float32 on the patch device")
    if context.object_mask.shape != (1, 1, SCENE_HEIGHT, SCENE_WIDTH):
        raise ValueError("object mask shape is incompatible")
    if context.object_mask.device != rgb.device or not bool(torch.isfinite(context.object_mask).all()):
        raise ValueError("object mask must be finite on the patch device")
    box = context.object_box
    if len(box) != 4 or any(int(x) != x for x in box):
        raise ValueError("object box must contain four integer coordinates")
    left, top, right, bottom = map(int, box)
    if not 0 <= left < right <= SCENE_WIDTH or not 0 <= top < bottom <= SCENE_HEIGHT:
        raise ValueError("object box must lie within the scene")
    protected = context.object_mask.ne(0).clone()
    protected[..., top:bottom, left:right] = True
    return protected


class RoadRGBView(BaseCarrier):
    """Differentiable view of RGB; caller owns optimization parameters."""

    kind = "road"

    def __init__(self, rgb: torch.Tensor) -> None:
        super().__init__()
        _validate_rgb(rgb)
        self.rgb = rgb

    def canonical_rgb(self) -> torch.Tensor:
        return canonicalize_rgb(self.rgb)

    def render(
        self,
        context: SceneContext,
        *,
        eot: dict[str, Any] | None = None,
        generator: torch.Generator | None = None,
        training: bool = False,
    ) -> CarrierRender:
        rgb = self.canonical_rgb()
        if eot:
            if not training or set(eot) != {"brightness_factor"}:
                raise ValueError("only explicit training patch brightness EOT is supported")
            rgb = apply_patch_brightness(rgb, sample_patch_brightness(generator, eot["brightness_factor"]))
        protected = _validate_context(context, rgb)
        destination = road_patch_destination(context.road_quad).to(rgb)
        if not bool(torch.isfinite(destination).all()):
            raise ValueError("non-finite road geometry")
        projected, alpha = perspective_project(rgb, torch.ones_like(rgb[:, :1]), destination)
        if not bool(torch.isfinite(projected).all()) or not bool(torch.isfinite(alpha).all()):
            raise ValueError("non-finite projected road patch")
        support = alpha > 0
        if not bool(support.any()):
            raise ValueError("road patch has empty image support")
        if bool((support & protected).any()):
            raise RuntimeError("any nonzero road alpha must be outside the vehicle mask AND bounding box")
        composite = (context.benign * (1.0 - alpha) + projected * alpha).clamp(0.0, 1.0)
        # Explicit selection preserves every unsupported pixel bit-for-bit.
        image = torch.where(support.expand_as(composite), composite, context.benign)
        if not torch.equal(image[protected.expand_as(image)], context.benign[protected.expand_as(image)]):
            raise RuntimeError("render changed a protected vehicle pixel")
        if not torch.equal(image[(~support).expand_as(image)], context.benign[(~support).expand_as(image)]):
            raise RuntimeError("render changed a pixel outside the road support")
        return CarrierRender(image, alpha, rgb, self.kind)


def save_patch(
    path: Path,
    rgb: torch.Tensor,
    metadata: dict[str, Any] | None = None,
    *,
    raw_generator_rgb: torch.Tensor | None = None,
) -> Path:
    """Atomic save with an explicit schema; never interpret legacy 512 RGB as canonical."""
    path = Path(path)
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("patch metadata must be a JSON object")
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    payload = {
        "schema": PATCH_SCHEMA,
        "renderer_contract": RENDERER_CONTRACT,
        "canonical_rgb": canonicalize_rgb(rgb).detach().cpu().contiguous(),
        "metadata": metadata,
    }
    if raw_generator_rgb is not None:
        _validate_rgb(raw_generator_rgb)
        canonical_raw = canonicalize_rgb(raw_generator_rgb).detach().cpu()
        if not torch.equal(canonical_raw, payload["canonical_rgb"]):
            raise ValueError("raw generator provenance does not produce this canonical patch")
        payload["raw_generator_rgb"] = raw_generator_rgb.detach().cpu().contiguous()
        payload["raw_generator_sha256"] = _tensor_digest(payload["raw_generator_rgb"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_patch(path: Path, *, map_location: Any = "cpu") -> torch.Tensor:
    """Load only this candidate's exact canonical RGB, with no resizing fallback."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != PATCH_SCHEMA:
        raise ValueError("patch lacks the explicit road_patch_rgb_v1 schema")
    if payload.get("renderer_contract") != RENDERER_CONTRACT:
        raise ValueError("patch renderer contract differs from this candidate")
    rgb = payload.get("canonical_rgb")
    _validate_rgb(rgb, canonical=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("patch metadata must be a JSON object")
    json.dumps(metadata, allow_nan=False)
    raw = payload.get("raw_generator_rgb")
    if raw is not None:
        _validate_rgb(raw)
        actual = _tensor_digest(raw)
        if payload.get("raw_generator_sha256") != actual:
            raise ValueError("raw generator provenance hash differs")
        # Do not resize provenance again: CPU/GPU interpolation can differ in ULPs.
        # Training/save checked the relation on the original execution device.
    return rgb
