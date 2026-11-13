"""Loss primitives for normalized production attacks."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


VEHICLE_CLASS_IDS = (3, 6, 8)


def _as_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must have shape [B,1,H,W], got {tuple(depth.shape)}")
    if not torch.isfinite(depth).all() or not depth.gt(0).all():
        raise ValueError("depth must contain finite positive values")
    return depth


def resize_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B,1,H,W], got {tuple(mask.shape)}")
    if mask.shape[0] != reference.shape[0]:
        raise ValueError("mask and reference batch sizes differ")
    return F.interpolate(mask.float(), reference.shape[-2:], mode="nearest").bool()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[resize_mask(mask, values).expand_as(values)]
    if not selected.numel():
        raise ValueError("masked reduction received an empty mask")
    return selected.mean()


def semantic_far_change(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    semantics: str,
) -> torch.Tensor:
    """Stable signed change whose positive direction always means farther away.

    Metric depth increases with distance; relative inverse depth decreases.  The
    asinh coordinate is finite and differentiable at zero and uses only the
    frozen benign prediction as its per-image scale.  It therefore avoids both
    reciprocal and log singularities without choosing an epsilon from results.
    """
    adversarial = adversarial_depth.unsqueeze(1) if adversarial_depth.ndim == 3 else adversarial_depth
    benign = benign_depth.unsqueeze(1) if benign_depth.ndim == 3 else benign_depth
    if adversarial.shape != benign.shape or adversarial.ndim != 4 or adversarial.shape[1] != 1:
        raise ValueError("depth quantities must have matching [B,1,H,W] shapes")
    if not torch.isfinite(adversarial).all() or not torch.isfinite(benign).all():
        raise ValueError("depth quantities must be finite")
    if adversarial.lt(0).any() or benign.lt(0).any():
        raise ValueError("depth quantities must be non-negative")
    if semantics not in {"metric_depth", "relative_depth", "relative_inverse_depth"}:
        raise ValueError(f"unsupported depth semantics: {semantics}")
    scale = benign.detach().abs().mean(dim=(-2, -1), keepdim=True)
    scale = scale.clamp_min(torch.finfo(benign.dtype).eps)
    change = torch.asinh(adversarial / scale) - torch.asinh(benign / scale)
    return change if semantics in {"metric_depth", "relative_depth"} else -change


def semantic_target_far_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    support: torch.Tensor,
    *,
    semantics: str,
    target_shift: float,
) -> torch.Tensor:
    if target_shift <= 0:
        raise ValueError("target_shift must be positive")
    change = semantic_far_change(adversarial_depth, benign_depth, semantics)
    return masked_mean((change - target_shift).square(), support)


def semantic_untargeted_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    support: torch.Tensor,
    *,
    semantics: str,
) -> torch.Tensor:
    change = semantic_far_change(adversarial_depth, benign_depth, semantics)
    return -masked_mean(change.abs(), support)


def semantic_advrm_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    object_mask: torch.Tensor,
    patch_support: torch.Tensor,
    *,
    semantics: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    change = semantic_far_change(adversarial_depth, benign_depth, semantics)
    object_roi = resize_mask(object_mask, change)
    patch_roi = resize_mask(patch_support, change) & ~object_roi
    primary = -change[object_roi].mean()
    hard_values = change[object_roi]
    hard = (-hard_values).clamp_min(0).mean()
    maintain = change.abs()[patch_roi].mean() if patch_roi.any() else change.sum() * 0.0
    components = {"semantic_far_primary": primary, "semantic_wrong_direction": hard, "semantic_maintain": maintain}
    return primary + hard + maintain, components


def fourier_high_frequency_energy(
    residual: torch.Tensor, *, cutoff_cycles_per_pixel: float = 0.25
) -> torch.Tensor:
    """Two-dimensional surrogate for BiTAA's rendered-residual HF penalty.

    The source paper does not disclose a reusable 2D ring mask or gamma value.
    This deterministic radial mask is therefore an explicit project adaptation,
    not Eq.13 reproduced in the original 3D multiview setting.
    """
    if residual.ndim != 4 or residual.shape[1] != 3:
        raise ValueError("Fourier residual must have shape [B,3,H,W]")
    if not 0.0 < cutoff_cycles_per_pixel < 0.5:
        raise ValueError("Fourier cutoff must lie in (0, 0.5)")
    if not torch.isfinite(residual).all():
        raise ValueError("Fourier residual contains non-finite values")
    height, width = residual.shape[-2:]
    fy = torch.fft.fftfreq(height, device=residual.device, dtype=residual.dtype)
    fx = torch.fft.rfftfreq(width, device=residual.device, dtype=residual.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    high = radius >= cutoff_cycles_per_pixel
    if not high.any():
        raise ValueError("Fourier high-frequency mask is empty")
    spectrum = torch.fft.rfft2(residual, norm="ortho")
    return spectrum.abs().square()[..., high].mean()


def box_iou(boxes: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N,4], got {tuple(boxes.shape)}")
    if target_box.shape != (4,):
        raise ValueError(f"target_box must have shape [4], got {tuple(target_box.shape)}")
    if not boxes.numel():
        return boxes.new_empty((0,))
    left_top = torch.maximum(boxes[:, :2], target_box[:2])
    right_bottom = torch.minimum(boxes[:, 2:], target_box[2:])
    extent = (right_bottom - left_top).clamp_min(0.0)
    intersection = extent[:, 0] * extent[:, 1]
    box_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0.0)
    target_area = (target_box[2] - target_box[0]).clamp_min(0.0) * (
        target_box[3] - target_box[1]
    ).clamp_min(0.0)
    return intersection / (box_area + target_area - intersection).clamp_min(1e-12)


def complete_box_iou(boxes: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes, target_box)
    box_width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-12)
    box_height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-12)
    target_width = (target_box[2] - target_box[0]).clamp_min(1e-12)
    target_height = (target_box[3] - target_box[1]).clamp_min(1e-12)
    box_center = (boxes[:, :2] + boxes[:, 2:]) / 2
    target_center = (target_box[:2] + target_box[2:]) / 2
    center_distance = (box_center - target_center).square().sum(dim=1)
    enclosing_top_left = torch.minimum(boxes[:, :2], target_box[:2])
    enclosing_bottom_right = torch.maximum(boxes[:, 2:], target_box[2:])
    enclosing_diagonal = (
        enclosing_bottom_right - enclosing_top_left
    ).square().sum(dim=1).clamp_min(1e-12)
    aspect = (4 / math.pi**2) * (
        torch.atan(target_width / target_height) - torch.atan(box_width / box_height)
    ).square()
    alpha = aspect / (1 - iou + aspect + 1e-12)
    return iou - center_distance / enclosing_diagonal - alpha * aspect


def target_vehicle_candidates(
    output: Mapping[str, torch.Tensor],
    target_box: tuple[int, int, int, int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = output["boxes"]
    scores = output["scores"]
    labels = output["labels"]
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("detector output lengths differ")
    target = torch.as_tensor(target_box, dtype=boxes.dtype, device=boxes.device)
    overlaps = box_iou(boxes, target)
    vehicle_ids = torch.tensor(VEHICLE_CLASS_IDS, device=labels.device)
    keep = torch.isin(labels, vehicle_ids) & overlaps.gt(0.0)
    return scores[keep], overlaps[keep]


def detection_suppression_loss(
    output: Mapping[str, torch.Tensor],
    target_box: tuple[int, int, int, int] | torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    scores, _ = target_vehicle_candidates(output, target_box)
    if not scores.numel():
        return output["scores"].sum() * 0.0
    peak = scores.max().clamp(0.0, 1.0)
    return -torch.log((1.0 - peak).clamp_min(0.0) + epsilon)


def detection_score_iou_loss(
    output: Mapping[str, torch.Tensor],
    target_box: tuple[int, int, int, int] | torch.Tensor,
) -> torch.Tensor:
    scores, overlaps = target_vehicle_candidates(output, target_box)
    if not scores.numel():
        return output["scores"].sum() * 0.0
    return (scores * overlaps).max()


def upc_fasterrcnn_loss(
    internals: Mapping[str, torch.Tensor],
    target_box: tuple[int, int, int, int] | torch.Tensor,
    *,
    confidence_threshold: float = 0.3,
    rpn_weight: float = 100.0,
    rpn_regression_balance: float = 0.005,
    classification_weight: float = 1000.0,
    classification_flip_balance: float = 0.002,
    regression_weight: float = 10.0,
    bbox_delta_target: float = -0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = {
        "anchors",
        "rpn_objectness_logits",
        "rpn_box_regression",
        "proposals",
        "roi_class_logits",
        "roi_box_regression",
    }
    if set(internals) != required:
        raise ValueError(f"UPC Faster R-CNN internals must be exactly {sorted(required)}")
    anchors = internals["anchors"]
    objectness_logits = internals["rpn_objectness_logits"].reshape(-1)
    rpn_box_regression = internals["rpn_box_regression"]
    proposals = internals["proposals"]
    class_logits = internals["roi_class_logits"]
    box_regression = internals["roi_box_regression"]
    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("UPC anchors must have shape [A,4]")
    if objectness_logits.shape != (anchors.shape[0],):
        raise ValueError("UPC RPN objectness count differs from anchors")
    if rpn_box_regression.shape != anchors.shape:
        raise ValueError("UPC RPN box regression must have shape [A,4]")
    if proposals.ndim != 2 or proposals.shape[1] != 4:
        raise ValueError("UPC proposals must have shape [R,4]")
    if class_logits.ndim != 2 or class_logits.shape[0] != proposals.shape[0]:
        raise ValueError("UPC ROI class logits must have shape [R,C]")
    if box_regression.shape != (*class_logits.shape, 4):
        raise ValueError("UPC ROI box regression must have shape [R,C,4]")
    if class_logits.shape[1] <= max(VEHICLE_CLASS_IDS):
        raise ValueError("UPC ROI class logits do not contain COCO vehicle classes")

    target = torch.as_tensor(target_box, dtype=anchors.dtype, device=anchors.device)
    anchor_keep = box_iou(anchors, target).gt(0.0)
    if not anchor_keep.any():
        raise ValueError("UPC target box does not overlap any RPN anchor")
    foreground = objectness_logits[anchor_keep].sigmoid()
    selected_rpn_regression = rpn_box_regression[anchor_keep]
    rpn_classification = rpn_weight * foreground.square().mean()
    rpn_regression = (
        rpn_weight
        * rpn_regression_balance
        * (
            foreground[:, None]
            * (selected_rpn_regression - bbox_delta_target).abs()
        ).mean()
    )

    probabilities = class_logits.softmax(dim=-1)
    vehicle_ids = torch.tensor(VEHICLE_CLASS_IDS, device=class_logits.device)
    vehicle_probabilities, vehicle_indices = probabilities[:, vehicle_ids].max(dim=1)
    proposal_keep = box_iou(proposals, target).gt(0.0) & vehicle_probabilities.gt(
        confidence_threshold
    )
    if proposal_keep.any():
        kept_logits = class_logits[proposal_keep]
        kept_classes = vehicle_ids[vehicle_indices[proposal_keep]]
        kept_probabilities = vehicle_probabilities[proposal_keep]
        class_flip = F.cross_entropy(kept_logits, kept_classes, reduction="sum")
        roi_classification = classification_weight * (
            kept_probabilities.mean() - classification_flip_balance * class_flip
        )
        kept_proposals = proposals[proposal_keep]
        centers = (kept_proposals[:, :2] + kept_proposals[:, 2:]) / 2
        collapsed = torch.cat([centers, centers], dim=1)
        proposal_collapse = torch.linalg.vector_norm(kept_proposals - collapsed)
        kept_regression = box_regression[proposal_keep, kept_classes]
        delta_shift = torch.linalg.vector_norm(kept_regression - bbox_delta_target)
        roi_regression = regression_weight * (proposal_collapse + delta_shift) / 2
    else:
        zero = class_logits.sum() * 0.0 + box_regression.sum() * 0.0
        roi_classification = zero
        roi_regression = zero
    components = {
        "rpn_classification": rpn_classification,
        "rpn_regression": rpn_regression,
        "roi_classification": roi_classification,
        "roi_regression": roi_regression,
    }
    return sum(components.values(), objectness_logits.sum() * 0.0), components


def fca_detector_head_loss(
    internals: Mapping[str, torch.Tensor],
    target_box: tuple[int, int, int, int] | torch.Tensor,
    *,
    box_weight: float = 0.05,
    objectness_weight: float = 1.0,
    classification_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = {
        "objectness_boxes",
        "predicted_boxes",
        "objectness_logits",
        "classification_boxes",
        "class_logits",
    }
    if set(internals) != required:
        raise ValueError(f"FCA detector internals must be exactly {sorted(required)}")
    objectness_boxes = internals["objectness_boxes"]
    predicted_boxes = internals["predicted_boxes"]
    objectness_logits = internals["objectness_logits"].reshape(-1)
    classification_boxes = internals["classification_boxes"]
    class_logits = internals["class_logits"]
    if objectness_boxes.ndim != 2 or objectness_boxes.shape[1] != 4:
        raise ValueError("FCA objectness boxes must have shape [A,4]")
    if predicted_boxes.shape != objectness_boxes.shape:
        raise ValueError("FCA predicted boxes must have shape [A,4]")
    if objectness_logits.shape != (objectness_boxes.shape[0],):
        raise ValueError("FCA objectness logit count differs from boxes")
    if classification_boxes.ndim != 2 or classification_boxes.shape[1] != 4:
        raise ValueError("FCA classification boxes must have shape [R,4]")
    if class_logits.ndim != 2 or class_logits.shape[0] != classification_boxes.shape[0]:
        raise ValueError("FCA class logits must have shape [R,C]")
    if class_logits.shape[1] <= max(VEHICLE_CLASS_IDS):
        raise ValueError("FCA class logits do not contain COCO vehicle classes")

    target = torch.as_tensor(
        target_box, dtype=objectness_boxes.dtype, device=objectness_boxes.device
    )
    objectness_overlaps = box_iou(objectness_boxes, target)
    if not objectness_overlaps.max().gt(0.0):
        raise ValueError("FCA target box does not overlap any objectness candidate")
    objectness_matches = objectness_overlaps.eq(objectness_overlaps.max())
    box_term = box_weight * complete_box_iou(
        predicted_boxes[objectness_matches], target
    ).mean()
    objectness_targets = objectness_matches.to(objectness_logits.dtype)
    objectness_term = objectness_weight * F.binary_cross_entropy_with_logits(
        objectness_logits, objectness_targets
    )

    classification_overlaps = box_iou(classification_boxes, target)
    if not classification_overlaps.max().gt(0.0):
        raise ValueError("FCA target box does not overlap any classification candidate")
    classification_matches = classification_overlaps.eq(classification_overlaps.max())
    vehicle_ids = torch.tensor(VEHICLE_CLASS_IDS, device=class_logits.device)
    vehicle_class_means = class_logits[classification_matches][:, vehicle_ids].mean(dim=0)
    class_term = classification_weight * vehicle_class_means.max().clamp_min(0.0)
    components = {
        "box_iou": box_term,
        "objectness_bce": objectness_term,
        "vehicle_class_logit": class_term,
    }
    return sum(components.values(), objectness_logits.sum() * 0.0), components


def detached_gradient_attention_loss(
    detector_loss: torch.Tensor,
    image: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    if not detector_loss.requires_grad:
        return image.sum() * 0.0
    gradient = torch.autograd.grad(
        detector_loss,
        image,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        return image.sum() * 0.0
    attention = (gradient.detach() * image).abs().mean(dim=1, keepdim=True)
    return masked_mean(attention, object_mask)


def das_connected_cam_loss(cam: torch.Tensor) -> torch.Tensor:
    if cam.shape != (7, 7):
        raise ValueError(f"DAS CAM must have shape [7,7], got {tuple(cam.shape)}")
    transformed = torch.tanh(cam)
    positive = transformed.detach().gt(0).cpu()
    visited = torch.zeros_like(positive)
    component_losses = []
    for row in range(7):
        for column in range(7):
            if not positive[row, column] or visited[row, column]:
                continue
            pending = [(row, column)]
            visited[row, column] = True
            points = []
            while pending:
                point_row, point_column = pending.pop()
                points.append((point_row, point_column))
                for neighbor_row, neighbor_column in (
                    (point_row - 1, point_column),
                    (point_row + 1, point_column),
                    (point_row, point_column - 1),
                    (point_row, point_column + 1),
                ):
                    if (
                        0 <= neighbor_row < 7
                        and 0 <= neighbor_column < 7
                        and positive[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        pending.append((neighbor_row, neighbor_column))
            values = torch.stack([transformed[index] for index in points])
            component_losses.append(values.sum() / (50 - len(points)))
    if not component_losses:
        return cam.sum() * 0.0
    return torch.stack(component_losses).mean()


def total_variation(rgb: torch.Tensor, *, squared: bool = False) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
    horizontal = rgb[..., :, 1:] - rgb[..., :, :-1]
    vertical = rgb[..., 1:, :] - rgb[..., :-1, :]
    if squared:
        horizontal = horizontal.square()
        vertical = vertical.square()
    else:
        horizontal = horizontal.abs()
        vertical = vertical.abs()
    return horizontal.mean() + vertical.mean()


def advrm_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(
            f"features must have shape [B,C,H,W], got {tuple(features.shape)}"
        )
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch * channels, height * width)
    return flattened @ flattened.T / (batch * channels * height * width)


def advrm_feature_losses(
    current: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    style_layers: tuple[str, ...],
    content_layer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = set(style_layers) | {content_layer}
    missing = sorted(required - current.keys() | required - reference.keys())
    if missing:
        raise ValueError(f"missing AdvRM VGG features: {missing}")
    style = sum(
        torch.nn.functional.mse_loss(
            advrm_gram_matrix(current[name]),
            advrm_gram_matrix(reference[name]).detach(),
        )
        for name in style_layers
    )
    content = torch.nn.functional.mse_loss(
        current[content_layer], reference[content_layer].detach()
    )
    return style, content


def advrm_residual_tv(rgb: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if rgb.shape != reference.shape or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("AdvRM RGB and reference must have matching [B,3,H,W] shapes")
    residual = rgb - reference
    horizontal = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).square().sum()
    vertical = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).square().sum()
    return (horizontal + vertical) / residual.numel()


def printability_loss(rgb: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must have shape [K,3], got {tuple(palette.shape)}")
    pixels = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    distances = torch.linalg.vector_norm(
        pixels[:, None, :] - palette.to(device=rgb.device, dtype=rgb.dtype)[None],
        dim=-1,
    )
    return distances.min(dim=1).values.mean()


def apa_non_printability_score(
    rgb: torch.Tensor, palette: torch.Tensor, *, epsilon: float = 1e-6
) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must have shape [K,3], got {tuple(palette.shape)}")
    colors = palette.to(device=rgb.device, dtype=rgb.dtype)
    distances = (
        (rgb[:, None] - colors[None, :, :, None, None] + epsilon)
        .square()
        .sum(dim=2)
        .add(epsilon)
        .sqrt()
    )
    return distances.min(dim=1).values.sum() / rgb.numel()


def apa_total_variation(rgb: torch.Tensor, *, epsilon: float = 1e-6) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
    horizontal = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1] + epsilon).abs().sum()
    vertical = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :] + epsilon).abs().sum()
    return (horizontal + vertical) / rgb.numel()


def threed2fool_squared_sum_tv(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
    batch, channels, height, width = rgb.shape
    vertical = (rgb[:, :, 1:, :-1] - rgb[:, :, :-1, :-1]).square()
    horizontal = (rgb[:, :, :-1, 1:] - rgb[:, :, :-1, :-1]).square()
    return (vertical + horizontal).sum().square() / (
        batch * channels * height * width
    )


def threed2fool_l1_nps(rgb: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must have shape [K,3], got {tuple(palette.shape)}")
    pixels = rgb.permute(0, 2, 3, 1)
    colors = palette.to(device=rgb.device, dtype=rgb.dtype)
    distances = (pixels[:, None] - colors[None, :, None, None]).abs().sum(dim=-1) / 255.0
    return distances.min(dim=1).values.sum() / (rgb.shape[-2] * rgb.shape[-1])


def linf_excess(
    rgb: torch.Tensor, reference: torch.Tensor, radius: float
) -> torch.Tensor:
    if rgb.shape != reference.shape:
        raise ValueError("rgb and reference shapes differ")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return (rgb.sub(reference).abs() - radius).clamp_min(0.0).square().mean()


def edge_weighted_content_loss(
    rgb: torch.Tensor,
    reference: torch.Tensor,
    edge_weight: float,
) -> torch.Tensor:
    if rgb.shape != reference.shape:
        raise ValueError("rgb and reference shapes differ")
    if edge_weight < 1.0:
        raise ValueError("edge_weight must be at least one")
    grayscale = reference.mean(dim=1, keepdim=True)
    horizontal = F.pad((grayscale[..., 1:] - grayscale[..., :-1]).abs(), (0, 1))
    vertical = F.pad((grayscale[..., 1:, :] - grayscale[..., :-1, :]).abs(), (0, 0, 0, 1))
    edges = (horizontal + vertical).detach()
    edges = edges / edges.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    weights = 1.0 + (edge_weight - 1.0) * edges
    return ((rgb - reference).square() * weights).mean()


def apa_target_depth_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    patch_support: torch.Tensor,
    *,
    target_depth: float = 150.0,
) -> torch.Tensor:
    depth = _as_depth(adversarial_depth)
    benign = _as_depth(benign_depth)
    if depth.shape != benign.shape:
        raise ValueError("adversarial and benign depth shapes differ")
    if target_depth <= 0:
        raise ValueError("target_depth must be positive")
    if patch_support.ndim != 4 or patch_support.shape[1] != 1:
        raise ValueError("APA patch support must have shape [B,1,H,W]")
    support = F.interpolate(
        patch_support.to(depth), depth.shape[-2:], mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0).expand_as(depth)
    if not support.gt(0).any():
        raise ValueError("APA patch support is empty")
    target = (1.0 - support) * benign.detach() + support * target_depth
    return torch.nn.functional.l1_loss(depth * support, target * support)


def saam_untargeted_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    patch_support: torch.Tensor,
) -> torch.Tensor:
    adversarial = _as_depth(adversarial_depth)
    benign = _as_depth(benign_depth)
    if adversarial.shape != benign.shape:
        raise ValueError("adversarial and benign depth shapes differ")
    if patch_support.ndim != 4 or patch_support.shape[1] != 1:
        raise ValueError("SAAM patch support must have shape [B,1,H,W]")
    support = F.interpolate(
        patch_support.to(adversarial),
        adversarial.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).expand_as(adversarial)
    support_sum = support.sum()
    if not support_sum.gt(0):
        raise ValueError("SAAM patch support is empty")
    return -((adversarial - benign).abs() * support).sum() / support_sum


def saam_squared_sum_tv(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W], got {tuple(rgb.shape)}")
    horizontal = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).square().sum()
    vertical = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).square().sum()
    return horizontal + vertical


def inverse_depth_squared_loss(
    adversarial_depth: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    depth = _as_depth(adversarial_depth)
    return masked_mean(depth.clamp_min(epsilon).reciprocal().square(), object_mask)


def advrm_primary_hard_maintain_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    object_mask: torch.Tensor,
    patch_support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    adversarial = _as_depth(adversarial_depth)
    benign = _as_depth(benign_depth)
    if adversarial.shape != benign.shape:
        raise ValueError("adversarial and benign depth shapes differ")
    object_roi = resize_mask(object_mask, adversarial)
    patch_roi = resize_mask(patch_support, adversarial) & ~object_roi
    primary = -adversarial[object_roi].mean()
    hard_roi = object_roi & adversarial.lt(benign)
    hard = -adversarial[hard_roi].mean() if hard_roi.any() else adversarial.sum() * 0.0
    maintain = (
        (adversarial - benign).abs()[patch_roi].mean()
        if patch_roi.any()
        else adversarial.sum() * 0.0
    )
    components = {"primary": primary, "hard": hard, "maintain": maintain}
    return primary + hard + maintain, components


def signed_log_depth_loss(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    beta: float = 0.1,
    direction: str = "far",
    epsilon: float = 1e-6,
) -> torch.Tensor:
    adversarial = _as_depth(adversarial_depth)
    benign = _as_depth(benign_depth)
    if adversarial.shape != benign.shape:
        raise ValueError("adversarial and benign depth shapes differ")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if direction not in {"far", "near"}:
        raise ValueError(f"unsupported depth direction: {direction}")
    sign = 1.0 if direction == "far" else -1.0
    residual = torch.log(adversarial.clamp_min(epsilon)) - torch.log(
        benign.clamp_min(epsilon)
    )
    return masked_mean((residual - sign * beta).square(), object_mask)
