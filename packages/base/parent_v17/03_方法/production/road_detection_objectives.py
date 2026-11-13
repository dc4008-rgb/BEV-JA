"""Detector objectives for the *road adaptations*, never original-paper replicas.

Contextual: mean car-class probability on target-matched pre-NMS predictions.
RobustDPatch-style: maximize native detection training losses against fixed clean
pseudo-labels plus the known inserted-car label. EOT and patch updates live in the
runner; calling this module alone does not constitute an EOT attack.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.ssd import SSD
from torchvision.ops import box_iou

CAR_LABEL = 3
SUPPORTED_MODELS = ("fasterrcnn", "ssd300")
TARGET_PSEUDO_IOU = 0.5


def _validate_adapter(adapter: Any, image: torch.Tensor) -> torch.nn.Module:
    model_id = getattr(adapter, "model_id", None)
    model = getattr(adapter, "model", None)
    if model_id not in SUPPORTED_MODELS:
        raise ValueError("road objectives support only fasterrcnn and ssd300")
    expected = FasterRCNN if model_id == "fasterrcnn" else SSD
    if not isinstance(model, expected):
        raise TypeError("model_id must reference the matching torchvision detector class")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("road objective requires one image [1,3,H,W]")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must have finite floating-point pixels")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("image pixels must be in [0,1]")
    if not image.requires_grad or not torch.is_grad_enabled():
        raise RuntimeError("attack image must retain its differentiable patch graph")
    return model


def _box(target_box: Any, reference: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(target_box, device=reference.device, dtype=reference.dtype)
    if result.shape != (4,) or not torch.isfinite(result).all():
        raise ValueError("target_box must be four finite xyxy coordinates")
    if not torch.all(result[2:] > result[:2]):
        raise ValueError("target_box must have positive width and height")
    return result.detach().reshape(1, 4)


@contextmanager
def frozen_detector_branch(model: torch.nn.Module, native_losses: bool = False) -> Iterator[None]:
    """Use the native loss branch without training weights or mutable submodules.

    All modules, including transform/BN/dropout, first use eval semantics. Only
    the model and Faster R-CNN RPN/ROI containers select their training branch.
    Direct .training restoration preserves originally mixed module states.
    Parameter flags are restored even on failure; no parameter graph is built.
    """
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval()
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        if native_losses:
            model.training = True
            if isinstance(model, FasterRCNN):
                model.rpn.training = True
                model.roi_heads.training = True
        yield
    finally:
        for module, training in modes:
            module.training = training
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)


def _features(model: torch.nn.Module, transformed: Any) -> OrderedDict:
    features = model.backbone(transformed.tensors)
    return OrderedDict([("0", features)]) if isinstance(features, torch.Tensor) else features


def _checked_scalar(loss: torch.Tensor) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("detector objective must be a finite scalar")
    if not loss.requires_grad:
        raise RuntimeError("detector objective is detached; no surrogate fallback is allowed")


def contextual_loss(adapter: Any, image: torch.Tensor, target_box: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return a scalar to MINIMIZE, with a one-backbone-pass accounting record.

    Faster R-CNN: append the known target box to the RPN proposals, then retain
    proposals at native foreground IoU threshold (normally 0.5). This explicit
    GT-proposal adaptation avoids an empty loss once proposals disappear.
    SSD: native SSDMatcher, including its best-anchor-per-GT forced assignment.
    Both retain pre-NMS softmax car-class suppression; neither minimizes a final
    post-NMS confidence or objectness surrogate.
    """
    model = _validate_adapter(adapter, image)
    target = _box(target_box, image)
    height, width = image.shape[-2:]
    if target.min() < 0 or target[0, 2] > width or target[0, 3] > height:
        raise ValueError("target_box is outside image bounds")
    labels = [{"boxes": target, "labels": torch.tensor([CAR_LABEL], device=image.device)}]
    with frozen_detector_branch(model):
        transformed, transformed_targets = model.transform([image[0]], labels)
        gt = transformed_targets[0]["boxes"]
        features = _features(model, transformed)
        if adapter.model_id == "fasterrcnn":
            proposals, _ = model.rpn(transformed, features)
            candidates = torch.cat([proposals[0], gt], dim=0)
            ious = box_iou(gt, candidates)[0]
            threshold = float(model.roi_heads.proposal_matcher.high_threshold)
            selected = ious >= threshold
            # Pool only target-matched proposals; selection and coordinates are
            # discrete, but the features and class logits remain differentiable.
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
        _checked_scalar(loss)
    stats = {
        "objective": "contextual_car_class_probability",
        "direction": "minimize", "model_id": adapter.model_id,
        "class_index": CAR_LABEL, "loss": float(loss.detach()),
        "matching_rule": match_rule, "matching_iou_threshold": threshold,
        "matched_prediction_count": count, "candidate_count": int(candidates.shape[0]),
        "explicit_gt_or_forced_match_count": forced,
        "matched_iou_min": float(ious[selected].min().detach()),
        "matched_iou_max": float(ious[selected].max().detach()),
        "model_forward_equivalents": 1, "backbone_forward_calls": 1,
        "classifier_forward_calls": 1, "extra_clean_forward_calls": 0,
        "adaptation": "road_patch_gt_matched_pre_nms_class_suppression",
    }
    return loss, stats


def make_robust_targets(clean_predictions: Any, target_box: Any, device: Any, score_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
    """Detach fixed clean pseudo-labels and replace target duplicates with car GT.

    Call once per clean scene/model before optimizing. Keep all predicted classes
    at score >= threshold except boxes IoU >= 0.5 with the inserted target. This
    avoids labeling the real background as empty. It differs from ART's wholly
    prediction-based labels and is recorded in the new adaptation contract.
    """
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0,1]")
    if isinstance(clean_predictions, list):
        if len(clean_predictions) != 1:
            raise ValueError("clean prediction batch must contain exactly one scene")
        prediction = clean_predictions[0]
    else:
        prediction = clean_predictions
    boxes = prediction["boxes"].detach().to(device=device, dtype=torch.float32).clone()
    scores = prediction["scores"].detach().to(device=device).clone()
    labels = prediction["labels"].detach().to(device=device).clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
        raise ValueError("clean predictions have inconsistent boxes/scores/labels")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("clean predictions contain non-finite values")
    if torch.any(boxes[:, 2:] <= boxes[:, :2]) or torch.any(scores < 0) or torch.any(scores > 1):
        raise ValueError("clean predictions contain invalid boxes or scores")
    if labels.dtype != torch.int64 or torch.any(labels < 1):
        raise ValueError("clean prediction labels must be int64 foreground class indices")
    target = _box(target_box, boxes)
    keep = (scores >= score_threshold) & (box_iou(target, boxes)[0] < TARGET_PSEUDO_IOU)
    return [{
        "boxes": torch.cat([boxes[keep], target], dim=0).detach(),
        "labels": torch.cat([labels[keep], labels.new_tensor([CAR_LABEL])], dim=0).detach(),
    }]


def robust_loss(adapter: Any, image: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return the positive sum of native training losses to MAXIMIZE.

    Faster R-CNN has RPN objectness/box losses plus ROI classification/box loss;
    SSD has native classification (with hard negatives) and box regression.
    No input/parameter optimization and no EOT happens inside this function.
    """
    model = _validate_adapter(adapter, image)
    if len(targets) != 1 or not len(targets[0]["boxes"]):
        raise ValueError("one nonempty fixed target set is required")
    boxes, labels = targets[0]["boxes"], targets[0]["labels"]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != (len(boxes),):
        raise ValueError("fixed targets have inconsistent boxes/labels")
    if boxes.requires_grad or labels.requires_grad:
        raise ValueError("pseudo-labels must be detached and fixed before attack")
    if boxes.device != image.device or labels.device != image.device:
        raise ValueError("fixed targets must be on the image device")
    if labels.dtype != torch.int64 or torch.any(labels < 1):
        raise ValueError("fixed labels must be int64 foreground indices")
    if not torch.isfinite(boxes).all() or torch.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("fixed targets contain invalid boxes")
    with frozen_detector_branch(model, native_losses=True):
        losses = model([image[0]], targets)
        expected = ({"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
                    if adapter.model_id == "fasterrcnn" else {"bbox_regression", "classification"})
        if not isinstance(losses, dict) or set(losses) != expected:
            raise RuntimeError("native loss keys changed; refusing a partial/surrogate objective")
        for component in losses.values():
            _checked_scalar(component)
            if component.detach() < 0:
                raise RuntimeError("native detector loss is unexpectedly negative")
        loss = sum(losses.values())
        _checked_scalar(loss)
    return loss, {
        "objective": "robust_dpatch_native_detection_loss", "direction": "maximize",
        "model_id": adapter.model_id, "loss": float(loss.detach()),
        "loss_components": {name: float(value.detach()) for name, value in losses.items()},
        "fixed_label_count": int(len(boxes)),
        "model_forward_equivalents": 1, "backbone_forward_calls": 1,
        "classifier_forward_calls": 1, "extra_clean_forward_calls": 0,
        "adaptation": "all_class_clean_pseudo_labels_plus_inserted_car_gt",
    }
