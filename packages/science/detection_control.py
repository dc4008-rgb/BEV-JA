"""Frozen detector matching and detached certified-reference gap; no allocation."""
from dataclasses import dataclass
import torch
from torch import Tensor
@dataclass(frozen=True)
class MatchingConfig:
    theta: float=.3
    chi: float=.5
    epsilon: float=1e-8
    dmax: float=3.0
    vehicle_labels: tuple=(3,6,8)
    uncertified_reference_policy: str="unit_deficit_preserve_scene"
DEFAULT_CONFIG=MatchingConfig()
def _vector(value: Tensor, name: str, *, nonnegative=False, confidence=False) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != 1 or not value.numel():
        raise ValueError(name + " must be a nonempty floating vector")
    result = value.detach().float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(name + " contains nonfinite values")
    if nonnegative and bool((result < 0).any()):
        raise ValueError(name + " contains negative values")
    if confidence and bool(((result < 0) | (result > 1)).any()):
        raise ValueError(name + " must be in [0,1]")
    return result


def target_matched_confidence(output: dict, target_box, config=DEFAULT_CONFIG):
    """Return maximum source-vehicle confidence at IoU >= chi, zero if absent.

    Retain native detector outputs. Matching is detached and does not replace
    the differentiable score-times-IoU LSE training loss. theta is not applied
    here: the clean confidence below theta must remain visible to eligibility.
    Labels use the frozen torchvision COCO IDs car=3,bus=6,truck=8.
    """
    if not isinstance(output, dict) or not all(k in output for k in ("boxes", "scores", "labels")):
        raise ValueError("detector output must contain boxes, scores and labels")
    boxes, scores, labels = (output[k] for k in ("boxes", "scores", "labels"))
    if not isinstance(boxes, Tensor) or not boxes.is_floating_point() or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be floating [N,4]")
    if not isinstance(scores, Tensor) or not scores.is_floating_point() or scores.shape != (len(boxes),):
        raise ValueError("scores must be floating [N]")
    if not isinstance(labels, Tensor) or labels.shape != scores.shape or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be integer [N]")
    if scores.device != boxes.device or labels.device != boxes.device:
        raise ValueError("detector output tensors must share their device")
    with torch.no_grad():
        boxes, scores = boxes.detach().float(), scores.detach().float()
        target = torch.as_tensor(target_box, device=boxes.device, dtype=torch.float32).detach()
        if target.shape != (4,) or not bool(torch.isfinite(target).all()) or not bool((target[2:] > target[:2]).all()):
            raise ValueError("target_box must be a finite nondegenerate xyxy box")
        if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError("detector output contains nonfinite values")
        if bool((boxes[:, 2:] < boxes[:, :2]).any()) or bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError("invalid detector geometry or confidences")
        wh = (torch.minimum(boxes[:, 2:], target[2:]) - torch.maximum(boxes[:, :2], target[:2])).clamp_min(0)
        intersection = wh.prod(dim=1)
        union = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1) + (target[2:] - target[:2]).prod() - intersection
        overlap = intersection / union
        vehicle = torch.zeros_like(labels, dtype=torch.bool)
        for label in config.vehicle_labels:
            vehicle |= labels == label
        matched = vehicle & (overlap >= config.chi)
        confidence = scores[matched].max() if bool(matched.any()) else scores.new_zeros(())
    return confidence.detach(), {
        "native_output_count": len(boxes), "vehicle_output_count": int(vehicle.sum()),
        "matched_output_count": int(matched.sum()), "chi": config.chi,
        "vehicle_labels": list(config.vehicle_labels), "matched_confidence": float(confidence),
        "matching_rule": "vehicle_label_and_iou_greater_equal_chi;maximum_native_score",
        "clean_detectable": bool(confidence > config.theta), "weights_use_identity_view_only": True,
    }


def detection_goal_deficits(adv_scores: Tensor, clean_scores: Tensor, config=DEFAULT_CONFIG):
    """Equation 10 on eligible references; explicitly unit-deficit elsewhere."""
    adv = _vector(adv_scores, "adv_scores", confidence=True)
    clean = _vector(clean_scores, "clean_scores", confidence=True)
    if adv.shape != clean.shape or adv.device != clean.device:
        raise ValueError("paired confidence vectors must share shape and device")
    certified = clean > config.theta
    result = torch.ones_like(adv)
    # Only evaluate equation 10 for its declared domain. No invalid negative
    # denominators are computed and silently clipped to a fabricated success.
    result[certified] = ((adv[certified] - config.theta).clamp_min(0) /
                         (clean[certified] - config.theta + config.epsilon)).clamp(0, config.dmax)
    return result.detach(), {
        "equation": "10_on_certified_references;unit_deficit_on_uncertified",
        "adv_identity_confidences": adv.cpu().tolist(), "clean_identity_confidences": clean.cpu().tolist(),
        "clean_reference_certified": certified.cpu().tolist(),
        "uncertified_reference_count": int((~certified).sum()),
        "uncertified_reference_policy": config.uncertified_reference_policy,
        "uncertified_reference_extension_not_in_pdf": True,
        "all_scenes_retained": True, "uncertified_is_not_success": True,
        "deficits": result.cpu().tolist(), "theta": config.theta,
        "epsilon": config.epsilon, "dmax": config.dmax, "detached": True,
    }

