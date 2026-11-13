"""Phase-safe Contextual road objective for both frozen core detectors.

Forward mathematics follows frozen parent_v17 contextual_loss: Faster R-CNN
RPN proposals plus fixed GT, or native SSDMatcher default anchors, followed by
mean softmax car class 3 probability. This is not the old C1 vehicle-group sum.
No adapter.predict, second backbone, clean query or parameter update is used.
Only train_pre needs an attack graph; scoring requires the caller's no_grad.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection._utils import SSDMatcher
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.ssd import SSD
from torchvision.ops import box_iou

CAR_LABEL = 3
OBJECTIVE_ID = "contextual_car_class_probability"
PHASES = ("train_pre", "train_post", "calibration")
REFERENCE_PATH = (Path(__file__).resolve().parents[1] /
                  "parent_v17/03_方法/production/road_detection_objectives.py")


@lru_cache(maxsize=1)
def frozen_reference():
    """Load the exact frozen helper rather than an ambiguous external import."""
    spec = importlib.util.spec_from_file_location("v23_frozen_contextual_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_phase(phase: str, require_attack_graph: bool) -> None:
    if phase not in PHASES or type(require_attack_graph) is not bool:
        raise ValueError("known phase and an exact boolean attack-graph flag required")
    if require_attack_graph != (phase == "train_pre"):
        raise ValueError("only train_pre must require the original attack graph")
    if not require_attack_graph and torch.is_grad_enabled():
        raise RuntimeError("calibration and train_post require caller no_grad")


def _validate(adapter, image, require_attack_graph):
    model_id, model = getattr(adapter, "model_id", None), getattr(adapter, "model", None)
    expected = {"fasterrcnn": FasterRCNN, "ssd300": SSD}.get(model_id)
    if expected is None or not isinstance(model, expected):
        raise TypeError("C3 requires the matching real torchvision core detector")
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or image.shape[0] != 1 or image.shape[1] != 3 or image.dtype != torch.float32
            or not bool(torch.isfinite(image).all()) or image.min() < 0 or image.max() > 1):
        raise ValueError("C3 image must be one finite float32 RGB image in [0,1]")
    if require_attack_graph and (not torch.is_grad_enabled() or not image.requires_grad):
        raise RuntimeError("train_pre requires the original differentiable attack image")
    if any(module.training for module in model.modules()):
        raise RuntimeError("C3 requires the frozen eval detector")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("C3 requires frozen detector parameters")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen detector has a pre-existing gradient")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("C3 requires float32 detector parameters")
    matcher = model.proposal_matcher if model_id == "ssd300" else model.roi_heads.proposal_matcher
    if matcher.high_threshold != .5 or matcher.low_threshold != .5:
        raise ValueError("C3 requires the frozen native foreground matching threshold 0.5")
    if model_id == "ssd300" and (not isinstance(matcher, SSDMatcher)
                                 or not isinstance(model.anchor_generator, DefaultBoxGenerator)
                                 or matcher.allow_low_quality_matches is not False):
        raise TypeError("C3 requires native SSDMatcher and native default anchors")
    return model


def capture_contextual_loss(adapter: Any, image: torch.Tensor, target_box: Any,
                            *, require_attack_graph: bool, phase: str):
    """Return (loss, live tensor evidence, JSON diagnostics) with one backbone.

    This detector-loss callback is not a call to the full inference forward.
    A failed callback may already have run some native detector modules.
    """
    validate_phase(phase, require_attack_graph)
    model = _validate(adapter, image, require_attack_graph)
    reference = frozen_reference()
    target = reference._box(target_box, image)
    height, width = image.shape[-2:]
    if target.min() < 0 or target[0, 2] > width or target[0, 3] > height:
        raise ValueError("target_box is outside image bounds")
    labels = [{"boxes": target, "labels": torch.tensor([CAR_LABEL], device=image.device)}]
    with reference.frozen_detector_branch(model):
        transformed, transformed_targets = model.transform([image[0]], labels)
        gt = transformed_targets[0]["boxes"]
        features = reference._features(model, transformed)
        if adapter.model_id == "fasterrcnn":
            proposals, _ = model.rpn(transformed, features)
            candidates = torch.cat([proposals[0], gt], dim=0)
            ious = box_iou(gt, candidates)[0]
            threshold = float(model.roi_heads.proposal_matcher.high_threshold)
            selected = ious >= threshold
            pooled = model.roi_heads.box_roi_pool(features, [candidates[selected]], transformed.image_sizes)
            box_features = model.roi_heads.box_head(pooled)
            logits, _ = model.roi_heads.box_predictor(box_features)
            match_rule = "rpn_proposals_plus_gt_iou_ge_native_roi_fg_threshold"
            forced = 1
        else:
            feature_list = list(features.values())
            outputs = model.head(feature_list)
            candidates = model.anchor_generator(transformed, feature_list)[0]
            ious = box_iou(gt, candidates)[0]
            selected = model.proposal_matcher(ious.unsqueeze(0)) >= 0
            threshold = float(model.proposal_matcher.high_threshold)
            logits = outputs["cls_logits"][0, selected]
            match_rule = "native_ssd_matcher_including_best_anchor_for_gt"
            forced = int((selected & (ious < threshold)).sum().item())
        count = int(selected.sum().item())
        if count == 0:
            raise RuntimeError("no target-matched predictions; empty contextual loss is forbidden")
        if logits.ndim != 2 or logits.shape[1] <= CAR_LABEL:
            raise RuntimeError("detector head lacks the COCO car class index 3")
        probabilities = logits.softmax(dim=-1)[:, CAR_LABEL]
        loss = probabilities.mean()
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError("Contextual objective must be a finite scalar")
        if require_attack_graph and not loss.requires_grad:
            raise RuntimeError("Contextual objective is detached; no surrogate graph is allowed")
        if not require_attack_graph and loss.requires_grad:
            raise RuntimeError("no-grad scoring unexpectedly built an attack graph")
    indices = selected.nonzero(as_tuple=False).flatten()
    evidence = {
        "target_box_original": target, "target_box_transformed": gt,
        "transformed_image": transformed.tensors, "candidates": candidates,
        "candidate_ious": ious, "matched_indices": indices,
        "matched_logits": logits, "matched_probabilities_car3": probabilities,
    }
    diagnostics = {
        "objective": OBJECTIVE_ID, "direction": "minimize", "model_id": adapter.model_id,
        "class_index": CAR_LABEL, "loss": float(loss.detach()),
        "matching_rule": match_rule, "matching_iou_threshold": threshold,
        "matched_prediction_count": count, "candidate_count": int(candidates.shape[0]),
        "explicit_gt_or_forced_match_count": forced,
        "matched_iou_min": float(ious[selected].min().detach()),
        "matched_iou_max": float(ious[selected].max().detach()),
        "model_forward_equivalents": 1, "backbone_forward_calls": 1,
        "classifier_forward_calls": 1, "extra_clean_forward_calls": 0,
        "adaptation": "road_patch_gt_matched_pre_nms_class_suppression",
        "phase": phase, "require_attack_graph": require_attack_graph,
        "loss_requires_grad": loss.requires_grad,
        "formula": "mean_softmax_car_class_3_over_native_target_matches",
        "matched_indices": indices.detach().tolist(),
        "matched_ious": ious[selected].detach().tolist(),
        "matched_car3_probabilities": probabilities.detach().tolist(),
        "autograd_calls": 0, "adapter_predict_calls": 0,
        "reference": "parent_v17/03_方法/production/road_detection_objectives.py:contextual_loss",
    }
    return loss, evidence, diagnostics
