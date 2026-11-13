"""Contract-driven objectives for the normalized attack methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from losses import (
    advrm_primary_hard_maintain_loss,
    apa_non_printability_score,
    apa_target_depth_loss,
    apa_total_variation,
    das_connected_cam_loss,
    detection_score_iou_loss,
    detection_suppression_loss,
    edge_weighted_content_loss,
    fca_detector_head_loss,
    fourier_high_frequency_energy,
    inverse_depth_squared_loss,
    linf_excess,
    saam_squared_sum_tv,
    saam_untargeted_loss,
    signed_log_depth_loss,
    semantic_advrm_loss,
    semantic_target_far_loss,
    semantic_untargeted_loss,
    resize_mask,
    threed2fool_l1_nps,
    threed2fool_squared_sum_tv,
    total_variation,
    upc_fasterrcnn_loss,
)


DETECTION_METHODS = {"upc", "das", "fca", "tt3d_detector_adaptation"}
DEPTH_METHODS = {"apa", "saam", "3d2fool", "advrm"}
JOINT_METHODS = {"bittaa", "ours"}


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def scalars(self) -> dict[str, float]:
        return {
            name: float(value.detach())
            for name, value in self.components.items()
        }


def load_method_contracts(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if not isinstance(methods, list) or len(methods) != 10:
        raise ValueError("method contract must contain exactly ten methods")
    indexed = {entry["id"]: entry for entry in methods}
    expected = DETECTION_METHODS | DEPTH_METHODS | JOINT_METHODS
    if set(indexed) != expected or len(indexed) != len(methods):
        raise ValueError("method contract ids do not match the frozen method set")
    budget = payload.get("normalized_budget", {})
    if budget.get("test_time_optimization") is not False:
        raise ValueError("method contract must forbid test-time optimization")
    return payload, indexed


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} is required for this objective")
    return value


def attack_objective(
    method_id: str,
    model_task: str,
    contract: Mapping[str, Any],
    *,
    target_box: tuple[int, int, int, int] | None = None,
    detection_output: Mapping[str, torch.Tensor] | None = None,
    detection_internals: Mapping[str, torch.Tensor] | None = None,
    detection_cam: torch.Tensor | None = None,
    image: torch.Tensor | None = None,
    adversarial_depth: torch.Tensor | None = None,
    benign_depth: torch.Tensor | None = None,
    object_mask: torch.Tensor | None = None,
    patch_support: torch.Tensor | None = None,
    direction: str = "far",
    depth_semantics: str | None = None,
) -> LossBreakdown:
    if model_task == "detection":
        if method_id not in DETECTION_METHODS | JOINT_METHODS:
            raise ValueError(f"{method_id} does not define a detection objective")
        box = _require(target_box, "target_box")
        if method_id == "upc":
            parameters = contract.get("objective", {})
            attack, components = upc_fasterrcnn_loss(
                _require(detection_internals, "detection_internals"),
                box,
                confidence_threshold=float(parameters["confidence_threshold"]),
                rpn_weight=float(parameters["rpn_weight"]),
                rpn_regression_balance=float(parameters["rpn_regression_balance"]),
                classification_weight=float(parameters["classification_weight"]),
                classification_flip_balance=float(parameters["classification_flip_balance"]),
                regression_weight=float(parameters["regression_weight"]),
                bbox_delta_target=float(parameters["bbox_delta_target"]),
            )
            return LossBreakdown(attack, components)
        if method_id == "das":
            attack = das_connected_cam_loss(_require(detection_cam, "detection_cam"))
            return LossBreakdown(attack, {"connected_detector_cam": attack})
        if method_id == "fca":
            parameters = contract.get("objective", {})
            attack, components = fca_detector_head_loss(
                _require(detection_internals, "detection_internals"),
                box,
                box_weight=float(parameters["box_weight"]),
                objectness_weight=float(parameters["objectness_weight"]),
                classification_weight=float(parameters["classification_weight"]),
            )
            return LossBreakdown(attack, components)
        output = _require(detection_output, "detection_output")
        if method_id == "ours":
            attack = detection_score_iou_loss(output, box)
            name = "target_vehicle_score_iou"
        else:
            attack = detection_suppression_loss(output, box)
            name = "target_vehicle_suppression"
        return LossBreakdown(attack, {name: attack})

    if model_task != "depth":
        raise ValueError(f"unsupported model task: {model_task}")
    if method_id not in DEPTH_METHODS | JOINT_METHODS:
        raise ValueError(f"{method_id} does not define a depth objective")
    adversarial = _require(adversarial_depth, "adversarial_depth")
    mask = _require(object_mask, "object_mask")
    semantics = _require(depth_semantics, "depth_semantics")
    regularizers = contract.get("regularizers", {})
    if method_id == "apa":
        attack = semantic_target_far_loss(
            adversarial,
            _require(benign_depth, "benign_depth"),
            _require(patch_support, "patch_support"),
            semantics=semantics,
            target_shift=float(contract.get("objective", {})["normalized_far_shift"]),
        )
        components = {"patch_support_semantic_far_target": attack}
    elif method_id == "saam":
        raw = semantic_untargeted_loss(
            adversarial,
            _require(benign_depth, "benign_depth"),
            _require(patch_support, "patch_support"),
            semantics=semantics,
        )
        attack = float(regularizers["depth_weight"]) * raw
        components = {
            "unweighted_depth": raw,
            "patch_support_untargeted_clean_depth_difference": attack,
        }
    elif method_id == "3d2fool":
        if semantics == "relative_inverse_depth":
            raw = (adversarial.square())[resize_mask(mask, adversarial).expand_as(adversarial)].mean()
        elif semantics in {"metric_depth", "relative_depth"}:
            raw = inverse_depth_squared_loss(adversarial, mask)
        else:
            raise ValueError(f"unsupported depth semantics: {semantics}")
        attack = float(regularizers["primary_weight"]) * raw
        components = {"semantic_inverse_depth_squared": raw, "weighted_primary": attack}
    elif method_id == "advrm":
        attack, components = semantic_advrm_loss(
            adversarial,
            _require(benign_depth, "benign_depth"),
            mask,
            _require(patch_support, "patch_support"),
            semantics=semantics,
        )
    else:
        beta = float(regularizers["depth_beta"])
        attack = semantic_target_far_loss(
            adversarial,
            _require(benign_depth, "benign_depth"),
            mask,
            semantics=semantics,
            target_shift=beta,
        )
        components = {"semantic_far_target": attack}
    return LossBreakdown(attack, components)


def carrier_regularization(
    method_id: str,
    contract: Mapping[str, Any],
    canonical_rgb: torch.Tensor,
    initial_rgb: torch.Tensor,
    *,
    printable_palette: torch.Tensor | None = None,
    shape_loss: torch.Tensor | None = None,
    color_budget_loss: torch.Tensor | None = None,
    advrm_losses: Mapping[str, torch.Tensor] | None = None,
) -> LossBreakdown:
    regularizers = contract.get("regularizers", {})
    components: dict[str, torch.Tensor] = {}

    if method_id == "upc":
        components["linf_excess"] = float(regularizers["linf_weight"]) * linf_excess(
            canonical_rgb, initial_rgb, float(regularizers["linf_radius"])
        )
        components["tv"] = float(regularizers["tv_weight"]) * total_variation(
            canonical_rgb
        )
    elif method_id == "das":
        components["content"] = float(regularizers["content_weight"]) * edge_weighted_content_loss(
            canonical_rgb, initial_rgb, float(regularizers["edge_weight"])
        )
        components["smooth"] = float(regularizers["smooth_weight"]) * total_variation(
            canonical_rgb, squared=True
        )
    elif method_id == "fca":
        components["smooth"] = float(regularizers["smooth_weight"]) * total_variation(
            canonical_rgb, squared=True
        )
    elif method_id == "tt3d_detector_adaptation":
        components["reconstruction"] = float(
            regularizers["reconstruction_weight"]
        ) * (canonical_rgb - initial_rgb).square().mean()
        components["smooth"] = float(regularizers["smooth_weight"]) * total_variation(
            canonical_rgb, squared=True
        )
    elif method_id == "apa":
        palette = _require(printable_palette, "printable_palette")
        components["nps"] = float(regularizers["nps_weight"]) * apa_non_printability_score(
            canonical_rgb, palette
        )
        tv = apa_total_variation(canonical_rgb)
        floor = tv.new_tensor(float(regularizers["tv_floor"]))
        components["tv"] = float(regularizers["tv_weight"]) * torch.maximum(tv, floor)
    elif method_id == "saam":
        components["tv"] = float(regularizers["tv_weight"]) * saam_squared_sum_tv(
            canonical_rgb
        )
    elif method_id == "3d2fool":
        palette = _require(printable_palette, "printable_palette")
        components["tv"] = float(regularizers["tv_weight"]) * threed2fool_squared_sum_tv(
            canonical_rgb
        )
        components["nps"] = float(regularizers["nps_weight"]) * threed2fool_l1_nps(
            canonical_rgb, palette
        )
    elif method_id == "advrm":
        raw = _require(advrm_losses, "advrm_losses")
        style_lambda = float(regularizers["style_lambda"])
        for name in ("style", "content", "tv"):
            components[name] = (
                style_lambda * float(regularizers[f"{name}_weight"]) * raw[name]
            )
    elif method_id == "bittaa":
        residual = canonical_rgb - initial_rgb
        components["shape_surrogate"] = float(regularizers["shape_weight"]) * _require(
            shape_loss, "shape_loss"
        )
        components["print_surrogate"] = float(regularizers["print_weight"]) * (
            _require(color_budget_loss, "color_budget_loss")
            + float(regularizers["canonical_residual_tv_surrogate_weight"])
            * total_variation(residual)
            + float(regularizers["fourier_high_frequency_surrogate_weight"])
            * fourier_high_frequency_energy(
                residual,
                cutoff_cycles_per_pixel=float(
                    regularizers["fourier_cutoff_cycles_per_pixel"]
                ),
            )
        )
    elif method_id == "ours":
        components["content"] = float(regularizers["content_weight"]) * (
            canonical_rgb - initial_rgb
        ).square().mean()
        components["tv"] = float(regularizers["tv_weight"]) * total_variation(
            canonical_rgb
        )
    else:
        raise ValueError(f"unsupported method: {method_id}")

    total = sum(components.values(), canonical_rgb.sum() * 0.0)
    return LossBreakdown(total, components)
