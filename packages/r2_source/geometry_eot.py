"""Whole-image geometry EOT, proposal 2, with an auditable CPU sampling path.

The caller must supply ranges explicitly: this module does not select experimental
hyperparameters. A batch contains identity plus three independently selected
single transforms (selection WITH replacement), never a composition of transforms.

Coordinates are continuous pixel BOUNDARIES: image extent is [0, W] x [0, H],
pixel (row, col) has centre (col + .5, row + .5), and boxes are xyxy. Forward
matrices map source boundaries to output boundaries. align_corners=False uses
exactly that convention. Rotation angles are positive clockwise on image axes.

Crop means a real integer crop followed by edge padding onto the original-sized
canvas. It does NOT resize the crop. Its random placement is the padding layout,
not an additional sampled translation transform. Other families use inverse
bilinear sampling with edge/border padding. Border padding repeats scene-edge
pixels; it is not real scene content. transform_support_mask marks the geometric
source footprint so downstream diagnostics can distinguish it from padding.

All resampling is on CPU, including backward, then copied to the original device.
This avoids CUDA grid_sample backward in this module, not nondeterminism in other
models/operators. The runner should set torch.set_num_threads(1) and enforce its
determinism policy before importing/using the experiment. We change no globals.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


_KINDS = ("crop", "scale", "translation", "rotation")


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be an explicit [minimum, maximum] pair")
    low, high = map(float, value)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid {name}: {value}")
    return low, high


def _config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = ("crop_side_fraction", "scale_range", "translation_fraction", "rotation_degrees")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"explicit EOT ranges required: {', '.join(missing)}")
    result = {key: list(_pair(config[key], key)) for key in required}
    if not 0 < result["crop_side_fraction"][0] <= result["crop_side_fraction"][1] <= 1:
        raise ValueError("crop_side_fraction must be within (0, 1]")
    if result["scale_range"][0] <= 0:
        raise ValueError("scale_range must be positive")
    if config.get("padding_mode", "border") != "border":
        raise ValueError("this policy requires edge/border padding")
    kinds = list(config.get("kinds", _KINDS))
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in _KINDS for kind in kinds):
        raise ValueError("kinds must be a nonempty, unique subset of supported transforms")
    if config.get("random_views", 3) != 3:
        raise ValueError("meeting proposal 2 requires exactly three random views plus identity")
    attempts = config.get("max_attempts", 50)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    result.update(kinds=kinds, random_views=3, max_attempts=attempts, padding_mode="border")
    return result


def _check_box(box: Sequence[float], height: int, width: int) -> list[float]:
    if len(box) != 4:
        raise ValueError("box must be xyxy")
    out = list(map(float, box))
    x0, y0, x1, y1 = out
    if not all(map(math.isfinite, out)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"box must have positive area and be fully within the source canvas: {out}")
    return out


def _corners(box: Sequence[float]) -> list[list[float]]:
    x0, y0, x1, y1 = map(float, box)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _map_points(points: list[list[float]], matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
             matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]] for x, y in points]


def _bbox(points: list[list[float]]) -> list[float]:
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    determinant = a * d - b * c
    if determinant == 0 or not math.isfinite(determinant):
        raise ValueError("non-invertible transform")
    return [[d / determinant, -b / determinant, (b * ty - d * tx) / determinant],
            [-c / determinant, a / determinant, (c * tx - a * ty) / determinant],
            [0.0, 0.0, 1.0]]


def _sample_transform(kind: str, height: int, width: int, rng: random.Random,
                      config: Mapping[str, Any]) -> tuple[list[list[float]], dict[str, Any], list[float]]:
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    bounds = [0.0, 0.0, float(width), float(height)]
    params: dict[str, Any] = {}
    if kind == "crop":
        fraction = rng.uniform(*config["crop_side_fraction"])
        crop_w = max(1, min(width, round(width * fraction)))
        crop_h = max(1, min(height, round(height * fraction)))
        left, top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
        pad_left, pad_top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
        bounds = [float(left), float(top), float(left + crop_w), float(top + crop_h)]
        matrix[0][2], matrix[1][2] = float(pad_left - left), float(pad_top - top)
        params = {"sampled_side_fraction": fraction, "crop_xyxy": [left, top, left + crop_w, top + crop_h],
                  "pad_lrtb": [pad_left, width - crop_w - pad_left, pad_top, height - crop_h - pad_top],
                  "crop_resized": False}
    elif kind == "scale":
        scale = rng.uniform(*config["scale_range"])
        matrix = [[scale, 0.0, (1 - scale) * width / 2],
                  [0.0, scale, (1 - scale) * height / 2], [0.0, 0.0, 1.0]]
        params = {"scale": scale, "centre_xy": [width / 2, height / 2]}
    elif kind == "translation":
        fx, fy = rng.uniform(*config["translation_fraction"]), rng.uniform(*config["translation_fraction"])
        matrix[0][2], matrix[1][2] = fx * width, fy * height
        params = {"fraction_xy": [fx, fy], "translation_xy": [fx * width, fy * height]}
    elif kind == "rotation":
        angle = rng.uniform(*config["rotation_degrees"])
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        cx, cy = width / 2, height / 2
        matrix = [[cos, -sin, cx - cos * cx + sin * cy],
                  [sin, cos, cy - sin * cx - cos * cy], [0.0, 0.0, 1.0]]
        params = {"angle_degrees_clockwise": angle, "centre_xy": [cx, cy]}
    elif kind != "identity":
        raise ValueError(f"unknown transform {kind}")
    return matrix, params, bounds


def sample_views(height: int, width: int, target_box: Sequence[float], patch_box: Sequence[float],
                 seed: int, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return four JSON-serializable views; reject lost target/patch content.

    For crop, both complete boxes must be inside the retained source rectangle.
    For every family, all four transformed corners of both boxes must stay in the
    output canvas. The family is selected once per view, then only its parameters
    are resampled. Exhaustion raises; there is no silent identity fallback.
    """
    if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1:
        raise ValueError("height and width must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an explicit integer")
    policy = _config(config)
    boxes = {"target": _check_box(target_box, height, width), "patch": _check_box(patch_box, height, width)}
    rng = random.Random(seed)
    views = []
    for index in range(4):
        rng_before = rng.getstate()
        kind = "identity" if index == 0 else rng.choice(policy["kinds"])
        for attempt in range(1, policy["max_attempts"] + 1):
            matrix, params, bounds = _sample_transform(kind, height, width, rng, policy)
            polygons = {name: _map_points(_corners(box), matrix) for name, box in boxes.items()}
            retained = all(bounds[0] <= box[0] and bounds[1] <= box[1] and
                           box[2] <= bounds[2] and box[3] <= bounds[3] for box in boxes.values())
            visible = all(0 <= x <= width and 0 <= y <= height for points in polygons.values() for x, y in points)
            if retained and visible:
                break
        else:
            raise RuntimeError(f"EOT visibility rejection exhausted: seed={seed}, view={index}, "
                               f"kind={kind}, attempts={policy['max_attempts']}; no fallback applied")
        views.append({"schema": "whole_image_geometry_eot_v1", "index": index, "kind": kind,
                      "seed": seed, "height": height, "width": width, "sampling_attempts": attempt,
                      "sampling_rng_before": rng_before, "sampling_rng_after": rng.getstate(),
                      "config": dict(policy), "parameters": params,
                      "forward_matrix": matrix, "inverse_matrix": _inverse(matrix),
                      "coordinate_convention": "pixel_boundaries_xy; pixel centres=(column+.5,row+.5)",
                      "interpolation": "bilinear", "align_corners": False, "padding_mode": "border",
                      "source_support": {"input_xyxy": bounds,
                                         "output_polygon": _map_points(_corners(bounds), matrix),
                                         "padding_is_real_content": False},
                      "target_source_box": boxes["target"], "patch_source_box": boxes["patch"],
                      "target_polygon": polygons["target"], "patch_polygon": polygons["patch"],
                      "target_box": _bbox(polygons["target"]), "patch_box": _bbox(polygons["patch"]),
                      "all_target_and_patch_corners_visible": True,
                      "clean_pairing": "apply this exact view to both clean and adversarial image"})
    return views


def transform_box(box: Sequence[float], view: Mapping[str, Any], height: int, width: int) -> list[float]:
    """Transform a source xyxy box into its (unclipped) output AABB."""
    if (height, width) != (view["height"], view["width"]):
        raise ValueError("box canvas does not match view")
    checked = _check_box(box, height, width)
    return _bbox(_map_points(_corners(checked), view["forward_matrix"]))


def _source_coordinates(view: Mapping[str, Any], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = view["height"], view["width"]
    yy, xx = torch.meshgrid(torch.arange(height, dtype=dtype) + .5,
                            torch.arange(width, dtype=dtype) + .5, indexing="ij")
    inv = view["inverse_matrix"]
    return inv[0][0] * xx + inv[0][1] * yy + inv[0][2], inv[1][0] * xx + inv[1][1] * yy + inv[1][2]


def transform_tensor(image: torch.Tensor, view: Mapping[str, Any]) -> torch.Tensor:
    """Apply to a CHW/NCHW image or soft mask, with differentiable CPU/device copy.

    Input/output shape, dtype and device match. Identity is returned unchanged.
    Call again with the same view for a paired clean image. Do not round or clamp
    after this function merely to hide numerical differences.
    """
    if image.ndim not in (3, 4) or not image.is_floating_point():
        raise ValueError("image must be floating CHW or NCHW")
    height, width = view["height"], view["width"]
    if tuple(image.shape[-2:]) != (height, width):
        raise ValueError("image canvas does not match view")
    if view["kind"] == "identity":
        return image
    if view.get("padding_mode") != "border" or view.get("align_corners") is not False:
        raise ValueError("unsupported padding/coordinate policy")
    original_dtype, original_device = image.dtype, image.device
    compute_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    source = image.to(device="cpu", dtype=compute_dtype)
    squeezed = source.ndim == 3
    if squeezed:
        source = source.unsqueeze(0)
    if view["kind"] == "crop":
        left, top, right, bottom = view["parameters"]["crop_xyxy"]
        cropped = source[..., top:bottom, left:right]
        out = F.pad(cropped, view["parameters"]["pad_lrtb"], mode="replicate")
    else:
        sx, sy = _source_coordinates(view, compute_dtype)
        grid = torch.stack((2 * sx / width - 1, 2 * sy / height - 1), dim=-1)
        grid = grid.unsqueeze(0).expand(source.shape[0], -1, -1, -1)
        out = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if squeezed:
        out = out.squeeze(0)
    return out.to(device=original_device, dtype=original_dtype)


def transform_support_mask(view: Mapping[str, Any], *, device: Any = "cpu",
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return 1xHxW binary geometric source support, with padding marked zero.

    Pixel centres are tested against the retained source boundary rectangle. This
    is a geometry mask, not an anti-aliased opacity or interpolation weight mask.
    """
    sx, sy = _source_coordinates(view, torch.float64)
    left, top, right, bottom = view["source_support"]["input_xyxy"]
    valid = (sx >= left) & (sx < right) & (sy >= top) & (sy < bottom)
    return valid.unsqueeze(0).to(device=device, dtype=dtype)
