"""Passive, single-forward SSD evidence and one preregistered C1 objective.

C1 minimizes mean pre-NMS vehicle-group probability on the native SSDMatcher
positives for the known, fixed target-car box. It neither chooses anchors by
road gradients nor forces positives at every feature level. No update, clean
forward, dtype change, model flag change, or post-NMS replacement happens here.
The caller owns frozen-state checks, backward accounting and raw serialization.
"""
from __future__ import annotations

from typing import Any

import torch
from torchvision.models.detection._utils import SSDMatcher
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_iou


VEHICLE_CLASSES = (3, 6, 8)
OBJECTIVE_ID = "c1_native_gt_matched_pre_nms_vehicle_group_probability"


def _float32_finite(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a float32 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def native_matched_vehicle_objective(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    target_box_transformed: torch.Tensor,
    matcher: SSDMatcher,
    level_counts: list[int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Pure C1 formula, exposing matching geometry without model computation.

    `anchors` are native default boxes in transformed-image pixels, never
    decoded predictions. Native SSDMatcher's one forced best match is retained,
    including its own tie handling. Geometry is fixed and must be detached.
    """
    for name, value in (("logits", logits), ("anchors", anchors),
                        ("target_box_transformed", target_box_transformed)):
        _float32_finite(value, name)
    if logits.ndim != 2 or logits.shape[1] <= max(VEHICLE_CLASSES):
        raise ValueError("logits must be [anchors, classes] including classes 3, 6, 8")
    if anchors.shape != (logits.shape[0], 4) or not len(anchors):
        raise ValueError("anchors must be a nonempty [anchors,4] array")
    if target_box_transformed.shape != (1, 4):
        raise ValueError("exactly one transformed target box is required")
    if anchors.device != logits.device or target_box_transformed.device != logits.device:
        raise ValueError("matching geometry and logits must share a device")
    if anchors.requires_grad or target_box_transformed.requires_grad:
        raise ValueError("matching geometry must be fixed, detached inputs")
    if not bool((anchors[:, 2:] > anchors[:, :2]).all()):
        raise ValueError("anchors must have positive area")
    if not bool((target_box_transformed[:, 2:] > target_box_transformed[:, :2]).all()):
        raise ValueError("target box must have positive area")
    if not isinstance(matcher, SSDMatcher):
        raise TypeError("C1 requires the native SSDMatcher")
    if (matcher.high_threshold != 0.5 or matcher.low_threshold != 0.5
            or matcher.allow_low_quality_matches is not False):
        raise ValueError("C1 requires the frozen native SSDMatcher threshold 0.5")
    if (not level_counts or any(type(n) is not int or n <= 0 for n in level_counts)
            or sum(level_counts) != len(anchors)):
        raise ValueError("positive integer level_counts must partition every anchor")
    if not torch.is_grad_enabled() or not logits.requires_grad:
        raise RuntimeError("C1 logits must retain the input attack graph")

    # Selection is independent of every predicted logit, score, regression and
    # input gradient. Do not substitute decoded boxes for the default anchors.
    ious = box_iou(target_box_transformed, anchors)[0]
    matching = matcher(ious.unsqueeze(0))
    selected = matching >= 0
    indices = selected.nonzero(as_tuple=False).flatten()
    if not len(indices):
        raise RuntimeError("native SSDMatcher produced no target positives")
    if not bool((matching[selected] == 0).all()):
        raise RuntimeError("native matcher returned an unexpected GT identity")
    probabilities = logits.softmax(dim=-1)
    selected_probabilities = probabilities[indices][:, list(VEHICLE_CLASSES)]
    loss = selected_probabilities.sum(dim=1).mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise RuntimeError("C1 must remain a finite differentiable scalar")

    level_ids = torch.repeat_interleave(
        torch.arange(len(level_counts), device=logits.device),
        torch.tensor(level_counts, device=logits.device),
    )
    threshold = float(matcher.high_threshold)
    best_index = int(ious.argmax().item())
    best_ties = (ious == ious[best_index]).nonzero(as_tuple=False).flatten()
    forced_below = (selected & (ious < threshold)).nonzero(as_tuple=False).flatten()
    if best_index not in indices.tolist():
        raise RuntimeError("native SSDMatcher did not retain its best anchor")
    per_level = []
    start = 0
    for level, count in enumerate(level_counts):
        matched = selected[start:start + count]
        values = ious[start:start + count][matched]
        per_level.append({
            "level": level, "start": start, "stop": start + count,
            "anchor_count": count, "matched_count": int(matched.sum().item()),
            "matched_iou_min": float(values.min().item()) if len(values) else None,
            "matched_iou_max": float(values.max().item()) if len(values) else None,
        })
        start += count
    evidence = {
        "anchor_ious": ious, "matching": matching,
        "matched_indices": indices, "anchor_level_ids": level_ids,
        "selected_vehicle_probabilities": selected_probabilities,
        "best_anchor_tie_indices": best_ties,
        "forced_below_threshold_indices": forced_below,
    }
    diagnostics = {
        "objective": OBJECTIVE_ID, "direction": "minimize",
        "vehicle_classes": list(VEHICLE_CLASSES),
        "formula": "mean_over_native_GT_matched_anchors(sum_softmax_classes_3_6_8)",
        "matching_rule": "native_ssd_matcher_including_best_anchor_for_gt",
        "matching_iou_threshold": threshold,
        "geometry": "fixed_GT_vs_native_default_anchors_in_transformed_pixels",
        "candidate_count": len(anchors), "matched_prediction_count": len(indices),
        "matched_indices": indices.tolist(),
        "matched_anchor_levels": level_ids[indices].tolist(),
        "matched_anchor_ious": ious[indices].tolist(),
        "best_anchor_index": best_index,
        "best_anchor_tie_indices": best_ties.tolist(),
        "forced_below_threshold_indices": forced_below.tolist(),
        "explicit_best_anchor_assignment_count": 1,
        "forced_below_threshold_count": len(forced_below),
        "loss": float(loss.detach().item()),
        "car3_auxiliary_loss_no_extra_forward_or_backward": float(
            selected_probabilities[:, 0].mean().detach().item()),
        "levels": per_level,
        "selection_uses_predictions_or_gradients": False,
        "cross_level_forced_matches": False,
    }
    return loss, evidence, diagnostics


def capture_ssd_forward(adapter: Any, image: torch.Tensor, target_box: Any):
    """Return (native_prediction_dict, live_evidence, C1_loss, JSON_diagnostics).

    Exactly one original `adapter.predict(image)` is called. Three passive
    forward hooks retain the native transform, head and anchor outputs; they
    return None and are always removed. `live_evidence` retains graph tensors;
    save detached copies before releasing the graph. No pretrained data load.
    """
    model = getattr(adapter, "model", None)
    if getattr(adapter, "model_id", None) != "ssd300" or not isinstance(model, SSD):
        raise TypeError("capture requires the real torchvision SSD adapter")
    if not isinstance(model.anchor_generator, DefaultBoxGenerator):
        raise TypeError("capture requires native SSD default anchors")
    if not isinstance(model.proposal_matcher, SSDMatcher):
        raise TypeError("capture requires native SSDMatcher")
    if (model.proposal_matcher.high_threshold != 0.5
            or model.proposal_matcher.low_threshold != 0.5
            or model.proposal_matcher.allow_low_quality_matches is not False):
        raise ValueError("C1 requires the frozen native SSDMatcher threshold 0.5")
    if any(module.training for module in model.modules()):
        raise RuntimeError("SSD and every submodule must already be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("SSD parameters must already be frozen")
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"SSD tensor {name} must remain float32")
        if value.device != image.device:
            raise ValueError(f"SSD tensor {name} is on a different device")
    _float32_finite(image, "image")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("one image [1,3,H,W] is required")
    if not image.requires_grad or not torch.is_grad_enabled():
        raise RuntimeError("image must retain the differentiable patch graph")
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("image pixels must be in [0,1]")
    if isinstance(target_box, torch.Tensor) and target_box.requires_grad:
        raise ValueError("target box must be fixed and detached")
    target = torch.as_tensor(target_box, dtype=torch.float32, device=image.device)
    if target.shape != (4,):
        raise ValueError("target_box must contain four xyxy coordinates")
    target = target.detach().clone().reshape(1, 4)
    _float32_finite(target, "target_box")
    if (not bool((target[:, 2:] > target[:, :2]).all()) or bool((target < 0).any())
            or float(target[0, 2]) > image.shape[-1] or float(target[0, 3]) > image.shape[-2]):
        raise ValueError("target box must be nondegenerate and inside the image")

    captures: dict[str, list[Any]] = {"transform": [], "head": [], "anchors": []}
    handles = []

    def capture_transform(_module, _args, output):
        captures["transform"].append(output)
        return None

    def capture_head(_module, args, output):
        captures["head"].append((args, output))
        return None

    def capture_anchors(_module, _args, output):
        captures["anchors"].append(output)
        return None

    try:
        handles.append(model.transform.register_forward_hook(capture_transform))
        handles.append(model.head.register_forward_hook(capture_head))
        handles.append(model.anchor_generator.register_forward_hook(capture_anchors))
        prediction = adapter.predict(image)
    finally:
        for handle in handles:
            handle.remove()
    if {name: len(values) for name, values in captures.items()} != {
            "transform": 1, "head": 1, "anchors": 1}:
        raise RuntimeError("native transform, head and anchors must each run exactly once")
    if not isinstance(prediction, list) or len(prediction) != 1:
        raise RuntimeError("SSD adapter must preserve its native one-image prediction list")

    transformed, transformed_targets = captures["transform"][0]
    if transformed_targets is not None:
        raise RuntimeError("native inference unexpectedly received training targets")
    if len(transformed.image_sizes) != 1:
        raise RuntimeError("native transformed batch size changed")
    _float32_finite(transformed.tensors, "transformed_image")
    head_args, outputs = captures["head"][0]
    feature_maps = head_args[0]
    if not isinstance(feature_maps, list) or not feature_maps:
        raise RuntimeError("native SSD head feature layout is unavailable")
    per_location = model.anchor_generator.num_anchors_per_location()
    if len(per_location) != len(feature_maps):
        raise RuntimeError("native anchor levels disagree with head features")
    feature_shapes = [list(feature.shape) for feature in feature_maps]
    if any(len(shape) != 4 or shape[0] != 1 for shape in feature_shapes):
        raise RuntimeError("native SSD feature maps require a one-image batch")
    level_counts = [shape[-2] * shape[-1] * count
                    for shape, count in zip(feature_shapes, per_location)]
    if not isinstance(outputs, dict) or set(outputs) != {"cls_logits", "bbox_regression"}:
        raise RuntimeError("native SSD head outputs changed")
    head_logits, head_regression = outputs["cls_logits"], outputs["bbox_regression"]
    if head_logits.ndim != 3 or head_logits.shape[0] != 1:
        raise RuntimeError("native SSD logits must have one batch dimension")
    if head_regression.shape != (1, head_logits.shape[1], 4):
        raise RuntimeError("native SSD regression layout changed")
    _float32_finite(head_regression, "bbox_regression")
    anchor_batches = captures["anchors"][0]
    if not isinstance(anchor_batches, list) or len(anchor_batches) != 1:
        raise RuntimeError("native SSD anchor batch changed")
    anchors = anchor_batches[0]
    resized_target = resize_boxes(target, image.shape[-2:], transformed.image_sizes[0])
    loss, evidence, diagnostics = native_matched_vehicle_objective(
        head_logits[0], anchors, resized_target, model.proposal_matcher, level_counts)
    evidence.update({
        "cls_logits": head_logits[0], "bbox_regression": head_regression[0],
        "anchors": anchors, "transformed_image": transformed.tensors,
        "target_box_original": target, "target_box_transformed": resized_target,
    })
    diagnostics.update({
        "model_id": "ssd300", "native_adapter_predict_calls": 1,
        "model_forward_equivalents": 1, "backbone_forward_calls": 1,
        "classifier_forward_calls": 1, "extra_clean_forward_calls": 0,
        "autograd_calls": 0,
        "passive_hook_calls": {name: len(values) for name, values in captures.items()},
        "hooks_removed": True, "feature_shapes": feature_shapes,
        "anchors_per_location": list(per_location), "level_counts": level_counts,
        "original_image_size": list(image.shape[-2:]),
        "transformed_image_size": list(transformed.image_sizes[0]),
        "head_layout": "level_concatenation_then_row_column_anchor_class",
        "c0_pre_nms_mapping": {"status": "unsupported", "reason": "no_anchor_index_tracing_implemented"},
        "c1_nonzero_gradient_does_not_establish_attack_success": True,
    })
    return prediction[0], evidence, loss, diagnostics
