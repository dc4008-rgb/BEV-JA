"""Frozen model adapters for the normalized reproduction protocol."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    ssd300_vgg16,
)
from torchvision.models.detection.rpn import concat_box_prediction_layers


CANVAS_HEIGHT = 320
CANVAS_WIDTH = 1024
DETECTION_MODEL_IDS = ("fasterrcnn", "ssd300", "maskrcnn", "fcos")
DEPTH_MODEL_IDS = ("monodepth2", "dpt_large", "depth_anything_v1", "zoedepth")
CORE_MODEL_IDS = ("fasterrcnn", "ssd300", "monodepth2", "dpt_large")
TRANSFER_MODEL_IDS = ("maskrcnn", "fcos", "depth_anything_v1", "zoedepth")

# transformers 4.46.3 instantiates this branch, but
# DPTFeatureFusionStage.forward calls layers[0] without a residual. These four
# tensors are therefore unreachable for the first fusion layer. The frozen
# Intel/dpt-large snapshot omits exactly these keys. Any other load discrepancy
# is an identity failure.
DPT_LARGE_ALLOWED_DEAD_MISSING_KEYS = frozenset(
    {
        "neck.fusion_stage.layers.0.residual_layer1.convolution1.bias",
        "neck.fusion_stage.layers.0.residual_layer1.convolution1.weight",
        "neck.fusion_stage.layers.0.residual_layer1.convolution2.bias",
        "neck.fusion_stage.layers.0.residual_layer1.convolution2.weight",
    }
)
ZOE_D_K_COMPATIBLE_NONPERSISTENT_KEYS = frozenset(
    f"core.core.pretrained.model.blocks.{index}.attn.relative_position_index"
    for index in range(24)
)


def validate_dpt_large_loading_info(loading_info: dict[str, object]) -> None:
    missing = set(loading_info.get("missing_keys", []))
    unexpected = set(loading_info.get("unexpected_keys", []))
    mismatched = loading_info.get("mismatched_keys", [])
    if missing != DPT_LARGE_ALLOWED_DEAD_MISSING_KEYS:
        raise RuntimeError(
            "DPT-Large checkpoint identity failure: unexpected missing-key set "
            f"{sorted(missing)}"
        )
    if unexpected or mismatched:
        raise RuntimeError(
            "DPT-Large checkpoint identity failure: "
            f"unexpected={sorted(unexpected)} mismatched={mismatched}"
        )


def resolve_processor_input_size(size: object) -> tuple[int, int]:
    """Normalize old dict and newer Transformers ``SizeDict`` contracts."""

    if isinstance(size, int):
        return size, size
    if isinstance(size, Mapping):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    height = getattr(size, "height", None)
    width = getattr(size, "width", None)
    if height is not None and width is not None:
        return int(height), int(width)
    shortest_edge = getattr(size, "shortest_edge", None)
    if shortest_edge is not None:
        edge = int(shortest_edge)
        return edge, edge
    raise ValueError(f"unsupported DPT-Large processor size contract: {size}")


def load_zoedepth_k_checkpoint_strict(
    model: torch.nn.Module, checkpoint: Path
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError("ZoeD_M12_K checkpoint has no state dictionary")
    state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }
    model_keys = set(model.state_dict())
    state_keys = set(state)
    missing = model_keys - state_keys
    compatible = state_keys - model_keys
    if missing:
        raise RuntimeError(f"ZoeD_K checkpoint is missing keys: {sorted(missing)}")
    if compatible != ZOE_D_K_COMPATIBLE_NONPERSISTENT_KEYS:
        raise RuntimeError(
            "ZoeD_K checkpoint has unexpected compatibility keys: "
            f"{sorted(compatible)}"
        )
    comparisons = []
    for name in sorted(compatible):
        runtime_value = model
        for part in name.split("."):
            runtime_value = (
                runtime_value[int(part)] if part.isdigit() else getattr(runtime_value, part)
            )
        checkpoint_value = state[name]
        exact_equal = bool(torch.equal(checkpoint_value, runtime_value))
        if not exact_equal:
            raise RuntimeError(f"ZoeD_K runtime buffer differs from checkpoint: {name}")
        comparisons.append(
            {"key": name, "shape": list(checkpoint_value.shape), "exact_equal": True}
        )
    filtered = {name: value for name, value in state.items() if name in model_keys}
    model.load_state_dict(filtered, strict=True)
    return {
        "strict_parameter_and_persistent_buffer_load": True,
        "missing_keys": [],
        "nonpersistent_compatibility_keys": comparisons,
    }


def apply_zoedepth_timm_eval_compatibility(model: torch.nn.Module) -> dict[str, object]:
    blocks = model.core.core.pretrained.model.blocks
    if len(blocks) != 24:
        raise RuntimeError(f"ZoeD_K expected 24 BEiT blocks, found {len(blocks)}")
    records = []
    for index, block in enumerate(blocks):
        if hasattr(block, "drop_path"):
            raise RuntimeError("ZoeD_K runtime unexpectedly already has legacy drop_path")
        first = getattr(block, "drop_path1", None)
        second = getattr(block, "drop_path2", None)
        if not isinstance(first, torch.nn.Identity) or not isinstance(second, torch.nn.Identity):
            raise RuntimeError(
                f"ZoeD_K timm compatibility is only valid for zero-rate Identity drop paths: {index}"
            )
        # The clean official MiDaS forward calls the legacy name twice. In this
        # frozen inference construction both modern branches are Identity, so
        # aliasing the first is exactly equivalent and introduces no state.
        block.drop_path = first
        records.append(
            {"block": index, "drop_path1": "Identity", "drop_path2": "Identity", "drop_prob": 0.0}
        )
    return {
        "reason": "official MiDaS legacy drop_path name with timm 1.0.26",
        "evaluation_equivalence": "Identity(x) for both modern branches and legacy alias",
        "blocks": records,
    }


class Predictor(Protocol):
    model_id: str
    task: str
    role: str

    def predict(self, image: torch.Tensor) -> object: ...


def validate_image(image: torch.Tensor) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
    if image.shape[-2:] != (CANVAS_HEIGHT, CANVAS_WIDTH):
        raise ValueError(f"image must use the frozen canvas, got {tuple(image.shape[-2:])}")
    if not torch.isfinite(image).all():
        raise ValueError("image contains non-finite pixels")


@dataclass
class DetectionAdapter:
    model_id: str
    role: str
    model: torch.nn.Module
    task: str = "detection"

    def predict(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        validate_image(image)
        return self.model([sample for sample in image])

    def _fasterrcnn_raw(self, image: torch.Tensor) -> dict[str, Any]:
        model: Any = self.model
        original_height, original_width = image.shape[-2:]
        transformed, _ = model.transform([image[0]], None)
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        feature_list = list(features.values())
        objectness_by_level, regression_by_level = model.rpn.head(feature_list)
        anchors = model.rpn.anchor_generator(transformed, feature_list)
        anchors_per_level = [value[0].numel() for value in objectness_by_level]
        objectness, rpn_regression = concat_box_prediction_layers(
            objectness_by_level, regression_by_level
        )
        decoded = model.rpn.box_coder.decode(rpn_regression.detach(), anchors)
        decoded = decoded.view(len(anchors), -1, 4)
        differentiable_boxes = model.rpn.box_coder.decode(rpn_regression, anchors)
        differentiable_boxes = differentiable_boxes.view(len(anchors), -1, 4)
        proposals, _ = model.rpn.filter_proposals(
            decoded, objectness, transformed.image_sizes, anchors_per_level
        )
        pooled = model.roi_heads.box_roi_pool(
            features, proposals, transformed.image_sizes
        )
        box_features = model.roi_heads.box_head(pooled)
        class_logits, roi_regression = model.roi_heads.box_predictor(box_features)

        transformed_height, transformed_width = transformed.image_sizes[0]
        scale = image.new_tensor(
            [
                original_width / transformed_width,
                original_height / transformed_height,
                original_width / transformed_width,
                original_height / transformed_height,
            ]
        )
        return {
            "feature_maps": tuple(features.values()),
            "anchors": anchors[0] * scale,
            "decoded_rpn_boxes": differentiable_boxes[0] * scale,
            "rpn_objectness_logits": objectness.reshape(-1),
            "rpn_box_regression": rpn_regression.reshape(-1, 4),
            "proposals": proposals[0] * scale,
            "roi_class_logits": class_logits,
            "roi_box_regression": roi_regression.reshape(
                class_logits.shape[0], class_logits.shape[1], 4
            ),
        }

    def predict_upc_fasterrcnn(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_image(image)
        if self.model_id != "fasterrcnn":
            raise ValueError(
                "UPC internal objective is only defined for fasterrcnn; "
                "SSD requires a separately specified adaptation"
            )
        if image.shape[0] != 1:
            raise ValueError("UPC Faster R-CNN adaptation requires batch size one")
        raw = self._fasterrcnn_raw(image)
        return {
            name: value
            for name, value in raw.items()
            if name not in {"feature_maps", "decoded_rpn_boxes"}
        }

    def _ssd_raw(self, image: torch.Tensor) -> dict[str, Any]:
        model: Any = self.model
        original_height, original_width = image.shape[-2:]
        transformed, _ = model.transform([image[0]], None)
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        feature_maps = tuple(features.values())
        head_outputs = model.head(list(feature_maps))
        anchors = model.anchor_generator(transformed, list(feature_maps))[0]
        predicted_boxes = model.box_coder.decode_single(
            head_outputs["bbox_regression"][0], anchors
        )
        transformed_height, transformed_width = transformed.image_sizes[0]
        scale = image.new_tensor(
            [
                original_width / transformed_width,
                original_height / transformed_height,
                original_width / transformed_width,
                original_height / transformed_height,
            ]
        )
        return {
            "feature_maps": feature_maps,
            "candidate_boxes": anchors * scale,
            "predicted_boxes": predicted_boxes * scale,
            "class_logits": head_outputs["cls_logits"][0],
        }

    def predict_fca_internals(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_image(image)
        if image.shape[0] != 1:
            raise ValueError("FCA detector-head adaptation requires batch size one")
        if self.model_id == "fasterrcnn":
            raw = self._fasterrcnn_raw(image)
            return {
                "objectness_boxes": raw["anchors"],
                "predicted_boxes": raw["decoded_rpn_boxes"],
                "objectness_logits": raw["rpn_objectness_logits"],
                "classification_boxes": raw["proposals"],
                "class_logits": raw["roi_class_logits"],
            }
        if self.model_id == "ssd300":
            raw = self._ssd_raw(image)
            class_logits = raw["class_logits"]
            objectness_logits = torch.logsumexp(class_logits[:, 1:], dim=1) - class_logits[:, 0]
            return {
                "objectness_boxes": raw["candidate_boxes"],
                "predicted_boxes": raw["predicted_boxes"],
                "objectness_logits": objectness_logits,
                "classification_boxes": raw["candidate_boxes"],
                "class_logits": class_logits,
            }
        raise ValueError(f"FCA detector-head adaptation is not defined for {self.model_id}")

    def predict_das_cam(
        self,
        image: torch.Tensor,
        target_box: tuple[int, int, int, int],
    ) -> torch.Tensor:
        validate_image(image)
        if image.shape[0] != 1:
            raise ValueError("DAS detector-CAM requires batch size one")
        if self.model_id == "fasterrcnn":
            raw = self._fasterrcnn_raw(image)
            candidate_boxes = raw["proposals"]
            class_logits = raw["roi_class_logits"]
        elif self.model_id == "ssd300":
            raw = self._ssd_raw(image)
            candidate_boxes = raw["candidate_boxes"]
            class_logits = raw["class_logits"]
        else:
            raise ValueError(f"DAS detector-CAM is not defined for {self.model_id}")
        if class_logits.shape[1] <= 8:
            raise ValueError("DAS detector logits do not contain COCO vehicle classes")
        target = image.new_tensor(target_box)
        top_left = torch.maximum(candidate_boxes[:, :2], target[:2])
        bottom_right = torch.minimum(candidate_boxes[:, 2:], target[2:])
        intersects_target = (bottom_right - top_left).clamp_min(0).prod(dim=1).gt(0)
        if not intersects_target.any():
            raise ValueError(f"DAS found no target-overlapping {self.model_id} candidates")
        vehicle_ids = torch.tensor((3, 6, 8), device=image.device)
        vehicle_logits = class_logits[:, vehicle_ids].max(dim=1).values
        target_score = vehicle_logits[intersects_target].mean()
        feature_maps = raw["feature_maps"]
        gradients = torch.autograd.grad(
            target_score,
            feature_maps,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        cams = []
        for feature_map, gradient in zip(feature_maps, gradients):
            if gradient is None or not gradient.detach().abs().any():
                continue
            weights = gradient.detach().mean(dim=(-2, -1), keepdim=True)
            cam = (weights * feature_map).sum(dim=1, keepdim=True)
            cams.append(F.adaptive_avg_pool2d(cam, (7, 7))[0, 0])
        if not cams:
            raise RuntimeError(f"DAS target score has no active {self.model_id} feature gradient")
        return torch.stack(cams).mean(dim=0)


@dataclass
class DepthTrace:
    stages: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def record(
        self,
        *,
        native_output: torch.Tensor,
        resized_output: torch.Tensor,
        clamped_output: torch.Tensor,
        final_depth: torch.Tensor,
        native_semantics: str,
        resized_semantics: str,
        interpolation: str,
        lower_bound: float,
        metric_depth_in_meters: bool,
        upper_bound: float | None = None,
        reciprocal_after_clamp: bool = False,
        evaluation_output: torch.Tensor | None = None,
        evaluation_semantics: str | None = None,
        evaluation_interpolation: str | None = None,
        extra_stages: dict[str, torch.Tensor] | None = None,
    ) -> None:
        values = {
            "native_output": native_output,
            "resized_output": resized_output,
            "clamped_output": clamped_output,
            "final_depth": final_depth,
            "evaluation_output": final_depth if evaluation_output is None else evaluation_output,
            "lower_clamp_mask": resized_output < lower_bound,
            "upper_clamp_mask": resized_output > upper_bound if upper_bound is not None else torch.zeros_like(resized_output, dtype=torch.bool),
            "nonfinite_resized_mask": ~torch.isfinite(resized_output),
            **(extra_stages or {}),
        }
        self.stages = {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}
        self.metadata = {
            "native_semantics": native_semantics,
            "resized_semantics": resized_semantics,
            "interpolation": interpolation,
            "align_corners": False if interpolation != "none" else None,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "reciprocal_after_clamp": reciprocal_after_clamp,
            "metric_depth_in_meters": metric_depth_in_meters,
            "evaluation_semantics": evaluation_semantics or resized_semantics,
            "evaluation_interpolation": evaluation_interpolation or interpolation,
            "evaluation_policy": "native_quantity_without_reciprocal_singularity",
            "capture": "detached_cpu_copy_without_numeric_policy_change",
        }


@dataclass(frozen=True)
class DepthPrediction:
    depth: torch.Tensor
    stages: dict[str, torch.Tensor]
    metadata: dict[str, object]


class DepthAdapter:
    training_output_semantics = "metric_depth"

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        return self._predict(image, None)

    def predict_for_training(self, image: torch.Tensor) -> torch.Tensor:
        """Return the differentiable quantity named by training_output_semantics."""
        return self.predict(image)

    def predict_with_diagnostics(self, image: torch.Tensor) -> DepthPrediction:
        trace = DepthTrace()
        depth = self._predict(image, trace)
        return DepthPrediction(depth, trace.stages, trace.metadata)

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        raise NotImplementedError


class MiDaSAdapter(DepthAdapter):
    model_id = "midas"
    task = "depth"
    role = "source"
    training_output_semantics = "relative_inverse_depth"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def _native_inverse_depth(self, image: torch.Tensor) -> torch.Tensor:
        validate_image(image)
        resized = F.interpolate(
            image,
            size=(160, 512),
            mode="bicubic",
            align_corners=False,
        )
        normalized = (resized - 0.5) / 0.5
        inverse_depth = self.model(normalized)
        return inverse_depth.unsqueeze(1) if inverse_depth.ndim == 3 else inverse_depth

    def predict_for_training(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            self._native_inverse_depth(image),
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        # MiDaS minimal resize for 1024x320 -> 512x160, constrained to /32.
        inverse_depth = self._native_inverse_depth(image)
        native_output = inverse_depth
        inverse_depth = F.interpolate(
            inverse_depth,
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bicubic",
            align_corners=False,
        )
        clamped = inverse_depth.clamp_min(1e-6)
        depth = clamped.reciprocal()
        evaluation_output = F.interpolate(
            native_output.unsqueeze(1) if native_output.ndim == 3 else native_output,
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        if trace is not None:
            trace.record(
                native_output=native_output, resized_output=inverse_depth,
                clamped_output=clamped, final_depth=depth,
                native_semantics="relative_inverse_depth", resized_semantics="relative_inverse_depth",
                interpolation="bicubic", lower_bound=1e-6,
                metric_depth_in_meters=False, reciprocal_after_clamp=True,
                evaluation_output=evaluation_output,
                evaluation_semantics="relative_inverse_depth",
                evaluation_interpolation="bilinear",
            )
        return depth


class AdaBinsAdapter(DepthAdapter):
    model_id = "adabins"
    task = "depth"
    role = "source"
    training_output_semantics = "metric_depth"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        validate_image(image)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (image - mean) / std
        model_input = F.interpolate(
            normalized,
            size=(352, 1216),
            mode="bilinear",
            align_corners=False,
        )
        with sdpa_kernel(SDPBackend.MATH):
            _, depth = self.model(model_input)
        native_output = depth
        depth = F.interpolate(
            depth,
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bicubic",
            align_corners=False,
        )
        clamped = depth.clamp(1e-3, 80.0)
        if trace is not None:
            trace.record(
                native_output=native_output, resized_output=depth,
                clamped_output=clamped, final_depth=clamped,
                native_semantics="metric_depth", resized_semantics="metric_depth",
                interpolation="bicubic", lower_bound=1e-3, upper_bound=80.0,
                metric_depth_in_meters=True,
                evaluation_output=clamped,
                evaluation_semantics="metric_depth",
            )
        return clamped


class Monodepth2Adapter(DepthAdapter):
    model_id = "monodepth2"
    task = "depth"
    role = "source"
    training_output_semantics = "relative_depth"

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        disp_to_depth: object,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.disp_to_depth = disp_to_depth

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        validate_image(image)
        features = self.encoder(image)
        outputs = self.decoder(features)
        disparity = outputs[("disp", 0)]
        scaled_disparity, depth = self.disp_to_depth(disparity, 0.1, 100.0)
        native_depth = depth
        interpolation = "none"
        if depth.shape[-2:] != (CANVAS_HEIGHT, CANVAS_WIDTH):
            interpolation = "bilinear"
            depth = F.interpolate(
                depth,
                size=(CANVAS_HEIGHT, CANVAS_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
        clamped = depth.clamp_min(1e-6)
        if trace is not None:
            trace.record(
                native_output=disparity, resized_output=depth,
                clamped_output=clamped, final_depth=clamped,
                native_semantics="sigmoid_disparity", resized_semantics="mono_scale_ambiguous_depth",
                interpolation=interpolation, lower_bound=1e-6,
                metric_depth_in_meters=False,
                evaluation_output=clamped,
                evaluation_semantics="mono_scale_ambiguous_depth",
                extra_stages={"scaled_disparity": scaled_disparity, "converted_native_depth": native_depth},
            )
            trace.metadata["disp_to_depth_bounds"] = [0.1, 100.0]
        return clamped


class DepthAnythingAdapter(DepthAdapter):
    model_id = "depth_anything_v1"
    task = "depth"
    role = "transfer"
    training_output_semantics = "relative_inverse_depth"

    def __init__(self, model: torch.nn.Module, processor: object, device: torch.device) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        validate_image(image)
        # Held-out only: the official processor need not preserve input gradients.
        batch = self.processor(
            images=[sample.detach().cpu() for sample in image],
            return_tensors="pt",
            do_rescale=False,
        )
        pixel_values = batch.pixel_values.to(self.device)
        inverse_depth = self.model(pixel_values=pixel_values).predicted_depth
        native_output = inverse_depth
        if inverse_depth.ndim == 3:
            inverse_depth = inverse_depth.unsqueeze(1)
        inverse_depth = F.interpolate(
            inverse_depth,
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bicubic",
            align_corners=False,
        )
        clamped = inverse_depth.clamp_min(1e-6)
        depth = clamped.reciprocal()
        evaluation_output = F.interpolate(
            native_output.unsqueeze(1) if native_output.ndim == 3 else native_output,
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        if trace is not None:
            trace.record(
                native_output=native_output, resized_output=inverse_depth,
                clamped_output=clamped, final_depth=depth,
                native_semantics="relative_inverse_depth", resized_semantics="relative_inverse_depth",
                interpolation="bicubic", lower_bound=1e-6,
                metric_depth_in_meters=False, reciprocal_after_clamp=True,
                evaluation_output=evaluation_output,
                evaluation_semantics="relative_inverse_depth",
                evaluation_interpolation="bilinear",
            )
        return depth


class DPTLargeAdapter(DepthAdapter):
    """Differentiable tensor-only adapter for the frozen DPT-Large core model."""

    model_id = "dpt_large"
    task = "depth"
    role = "source"
    training_output_semantics = "relative_inverse_depth"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        input_size: tuple[int, int],
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
    ) -> None:
        self.model = model
        self.input_size = input_size
        self.image_mean = image_mean
        self.image_std = image_std

    def _native_inverse_depth(self, image: torch.Tensor) -> torch.Tensor:
        validate_image(image)
        resized = F.interpolate(
            image, size=self.input_size, mode="bicubic", align_corners=False
        )
        mean = image.new_tensor(self.image_mean).view(1, 3, 1, 1)
        std = image.new_tensor(self.image_std).view(1, 3, 1, 1)
        output = self.model(pixel_values=(resized - mean) / std)
        inverse_depth = (
            output.predicted_depth
            if hasattr(output, "predicted_depth")
            else output["predicted_depth"]
        )
        if inverse_depth.ndim == 3:
            inverse_depth = inverse_depth.unsqueeze(1)
        return inverse_depth

    def predict_for_training(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            self._native_inverse_depth(image),
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        native = self._native_inverse_depth(image)
        evaluated = F.interpolate(
            native,
            size=(CANVAS_HEIGHT, CANVAS_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        if not torch.isfinite(evaluated).all() or evaluated.lt(0).any():
            raise ValueError("DPT-Large produced negative or non-finite relative inverse depth")
        if trace is not None:
            trace.record(
                native_output=native,
                resized_output=evaluated,
                clamped_output=evaluated,
                final_depth=evaluated,
                native_semantics="relative_inverse_depth",
                resized_semantics="relative_inverse_depth",
                interpolation="bilinear",
                lower_bound=0.0,
                metric_depth_in_meters=False,
                evaluation_output=evaluated,
                evaluation_semantics="relative_inverse_depth",
                evaluation_interpolation="bilinear",
            )
        return evaluated


class ZoeDepthAdapter(DepthAdapter):
    """Held-out ZoeD_K metric-depth adapter; never used for attack training."""

    model_id = "zoedepth"
    task = "depth"
    role = "transfer"
    training_output_semantics = "metric_depth"

    def __init__(
        self,
        model: torch.nn.Module,
        processor: object | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def _native_depth(self, image: torch.Tensor) -> torch.Tensor:
        validate_image(image)
        if self.processor is not None:
            if self.device is None:
                raise ValueError("ZoeDepth processor binding requires an explicit device")
            # ZoeDepth is transfer-only.  Use the frozen Hugging Face image
            # processor exactly; the input is already RGB float in [0,1], so
            # rescaling by 1/255 must be disabled.
            batch = self.processor(
                images=[sample.detach().cpu() for sample in image],
                return_tensors="pt",
                do_rescale=False,
            )
            output = self.model(pixel_values=batch.pixel_values.to(self.device))
            depth = (
                output.predicted_depth
                if hasattr(output, "predicted_depth")
                else output["predicted_depth"]
            )
        elif hasattr(self.model, "infer"):
            depth = self.model.infer(image)
        else:
            output = self.model(image)
            depth = output.get("metric_depth") if isinstance(output, dict) else output
        if depth is None:
            raise ValueError("ZoeD_K output does not contain metric depth")
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        return depth

    def _predict(self, image: torch.Tensor, trace: DepthTrace | None) -> torch.Tensor:
        native = self._native_depth(image)
        interpolation = "none"
        evaluated = native
        if native.shape[-2:] != (CANVAS_HEIGHT, CANVAS_WIDTH):
            interpolation = "bilinear"
            evaluated = F.interpolate(
                native,
                size=(CANVAS_HEIGHT, CANVAS_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
        if not torch.isfinite(evaluated).all() or evaluated.lt(0).any():
            raise ValueError("ZoeD_K produced negative or non-finite metric depth")
        if trace is not None:
            trace.record(
                native_output=native,
                resized_output=evaluated,
                clamped_output=evaluated,
                final_depth=evaluated,
                native_semantics="metric_depth",
                resized_semantics="metric_depth",
                interpolation=interpolation,
                lower_bound=0.0,
                metric_depth_in_meters=True,
                evaluation_output=evaluated,
                evaluation_semantics="metric_depth",
                evaluation_interpolation=interpolation,
            )
        return evaluated


def build_detection_adapter(model_id: str, device: torch.device) -> DetectionAdapter:
    if model_id == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            min_size=320,
            max_size=1024,
            box_score_thresh=1e-4,
            box_detections_per_img=1000,
        )
        role = "source"
    elif model_id == "ssd300":
        model = ssd300_vgg16(
            weights=SSD300_VGG16_Weights.COCO_V1,
            score_thresh=1e-4,
            topk_candidates=2000,
            detections_per_img=1000,
        )
        role = "source"
    elif model_id == "maskrcnn":
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            min_size=320,
            max_size=1024,
            box_score_thresh=1e-4,
            box_detections_per_img=1000,
        )
        role = "transfer"
    elif model_id == "fcos":
        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.COCO_V1,
            min_size=320,
            max_size=1024,
            score_thresh=1e-4,
            topk_candidates=2000,
            detections_per_img=1000,
        )
        role = "transfer"
    else:
        raise ValueError(f"unsupported detection model: {model_id}")
    model = model.to(device).eval().requires_grad_(False)
    return DetectionAdapter(model_id=model_id, role=role, model=model)


def build_midas_adapter(device: torch.device, midas_root: Path) -> MiDaSAdapter:
    root_text = str(midas_root.resolve())
    sys.path.insert(0, root_text)
    try:
        loader = importlib.import_module("midas.model_loader")
    finally:
        sys.path.remove(root_text)
    model, _, _, _ = loader.load_model(
        device,
        str(midas_root / "weights/dpt_beit_large_512.pt"),
        "dpt_beit_large_512",
        False,
        512,
        False,
    )
    return MiDaSAdapter(model.to(device).eval().requires_grad_(False))


def build_adabins_adapter(device: torch.device, adabins_root: Path) -> AdaBinsAdapter:
    root_text = str(adabins_root.resolve())
    sys.path.insert(0, root_text)
    try:
        models = importlib.import_module("models")
        model_io = importlib.import_module("model_io")
    finally:
        sys.path.remove(root_text)
    model = models.UnetAdaptiveBins.build(
        n_bins=256,
        min_val=1e-3,
        max_val=80.0,
    )
    checkpoint = adabins_root / "pretrained/AdaBins_kitti.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "epoch" in payload:
        model, _, _ = model_io.load_checkpoint(str(checkpoint), model)
    else:
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        remapped = {}
        for name, value in state.items():
            name = name.replace("module.", "")
            if name.startswith("adaptive_bins_layer.embedding_conv."):
                name = name.replace(
                    "adaptive_bins_layer.embedding_conv.",
                    "adaptive_bins_layer.conv3x3.",
                )
            elif name.startswith(
                "adaptive_bins_layer.patch_transformer.embedding_encoder"
            ):
                name = name.replace(
                    "adaptive_bins_layer.patch_transformer.embedding_encoder",
                    "adaptive_bins_layer.patch_transformer.embedding_convPxP",
                )
            remapped[name] = value
        model.load_state_dict(remapped)
    return AdaBinsAdapter(model.to(device).eval().requires_grad_(False))


def build_monodepth2_adapter(
    device: torch.device, repository: Path
) -> Monodepth2Adapter:
    repository_text = str(repository.resolve())
    sys.path.insert(0, repository_text)
    try:
        networks = importlib.import_module("networks")
        layers = importlib.import_module("layers")
    finally:
        sys.path.remove(repository_text)
    weights = repository / "models/mono_1024x320"
    encoder_payload = torch.load(
        weights / "encoder.pth", map_location="cpu", weights_only=True
    )
    encoder = networks.ResnetEncoder(18, False)
    encoder_state = encoder.state_dict()
    encoder.load_state_dict(
        {name: value for name, value in encoder_payload.items() if name in encoder_state}
    )
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(
        torch.load(weights / "depth.pth", map_location="cpu", weights_only=True)
    )
    encoder = encoder.to(device).eval().requires_grad_(False)
    decoder = decoder.to(device).eval().requires_grad_(False)
    return Monodepth2Adapter(encoder, decoder, layers.disp_to_depth)


def build_depth_anything_adapter(
    device: torch.device, snapshot: Path
) -> DepthAnythingAdapter:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot, local_files_only=True
    )
    model = model.to(device).eval().requires_grad_(False)
    return DepthAnythingAdapter(model, processor, device)


def build_dpt_large_adapter(
    device: torch.device, snapshot: Path
) -> DPTLargeAdapter:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model, loading_info = AutoModelForDepthEstimation.from_pretrained(
        snapshot, local_files_only=True, output_loading_info=True
    )
    validate_dpt_large_loading_info(loading_info)
    input_size = resolve_processor_input_size(getattr(processor, "size", None))
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("DPT-Large processor mean/std must contain three channels")
    model = model.to(device).eval().requires_grad_(False)
    return DPTLargeAdapter(
        model,
        input_size=input_size,
        image_mean=mean,
        image_std=std,
    )


def build_zoedepth_adapter(
    device: torch.device,
    repository: Path,
    checkpoint: Path,
    midas_repository: Path,
) -> ZoeDepthAdapter:
    if not (repository / "zoedepth").is_dir():
        raise FileNotFoundError("bound official ZoeDepth repository is unavailable")
    if not (midas_repository / "hubconf.py").is_file():
        raise FileNotFoundError("bound official MiDaS repository is unavailable")
    if not checkpoint.is_file():
        raise FileNotFoundError("bound official ZoeD_M12_K checkpoint is unavailable")

    repository_text = str(repository.resolve())
    original_hub_load = torch.hub.load

    def local_midas_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "intel-isl/MiDaS":
            return original_hub_load(repo_or_dir, model, *args, **kwargs)
        kwargs.pop("source", None)
        return original_hub_load(
            str(midas_repository.resolve()), model, *args, source="local", **kwargs
        )

    sys.path.insert(0, repository_text)
    torch.hub.load = local_midas_load
    try:
        config_module = importlib.import_module("zoedepth.utils.config")
        builder_module = importlib.import_module("zoedepth.models.builder")
        config = config_module.get_config(
            "zoedepth",
            "infer",
            dataset="kitti",
            config_version="kitti",
            pretrained_resource=None,
        )
        if config.get("max_depth") != 80 or config.get("dataset") != "kitti":
            raise RuntimeError("official ZoeD_K configuration is not the KITTI/80m contract")
        model = builder_module.build_model(config)
        timm_compatibility = apply_zoedepth_timm_eval_compatibility(model)
        compatibility = load_zoedepth_k_checkpoint_strict(model, checkpoint.resolve())
        compatibility["timm_eval_compatibility"] = timm_compatibility
        model.normalized_v9_checkpoint_compatibility = compatibility
    finally:
        torch.hub.load = original_hub_load
        sys.path.remove(repository_text)
    model = model.to(device).eval().requires_grad_(False)
    return ZoeDepthAdapter(model)


def build_depth_adapter(
    model_id: str,
    device: torch.device,
    *,
    monodepth2_root: Path = Path(
        "/root/autodl-tmp/normalized_reproduction_20260904/02_模型/monodepth2"
    ),
    dpt_large_snapshot: Path = Path("/root/autodl-tmp/model_bindings/dpt-large"),
    depth_anything_snapshot: Path = Path(
        "/root/autodl-tmp/hf_cache/hub/models--LiheYoung--depth-anything-large-hf/"
        "snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6"
    ),
    zoedepth_repository: Path = Path("/root/autodl-tmp/model_bindings/ZoeDepth"),
    zoedepth_checkpoint: Path = Path("/root/autodl-tmp/model_bindings/ZoeD_M12_K.pt"),
    zoedepth_midas_repository: Path = Path("/root/autodl-tmp/model_bindings/MiDaS"),
    model_manifest: Path | None = None,
) -> Predictor:
    if model_manifest is not None:
        payload = json.loads(Path(model_manifest).read_text(encoding="utf-8"))
        matches = [item for item in payload["models"] if item["id"] == model_id]
        if len(matches) != 1:
            raise ValueError(f"model manifest must identify exactly one {model_id}")
        definition = matches[0]
        artifacts = [Path(item["path"]) for item in definition.get("artifacts", [])]
        if model_id in {"dpt_large", "zoedepth"} and not artifacts:
            raise ValueError(f"model artifacts remain unbound: {model_id}")
        if model_id == "monodepth2":
            monodepth2_root = Path(definition["repository"]["path"])
        elif model_id == "dpt_large":
            configs = [path for path in artifacts if path.name == "config.json"]
            if len(configs) != 1:
                raise ValueError("DPT-Large binding must contain one config.json")
            dpt_large_snapshot = configs[0].parent
        elif model_id == "depth_anything_v1":
            configs = [path for path in artifacts if path.name == "config.json"]
            if len(configs) != 1:
                raise ValueError("Depth Anything binding must contain one config.json")
            depth_anything_snapshot = configs[0].parent
        elif model_id == "zoedepth":
            repository = definition.get("repository", {})
            zoedepth_repository = Path(repository["path"])
            zoedepth_midas_repository = Path(repository["midas_path"])
            checkpoints = [path for path in artifacts if path.name == "ZoeD_M12_K.pt"]
            if len(checkpoints) != 1:
                raise ValueError("ZoeDepth-KITTI binding must contain ZoeD_M12_K.pt")
            zoedepth_checkpoint = checkpoints[0]
    if model_id == "monodepth2":
        return build_monodepth2_adapter(device, monodepth2_root)
    if model_id == "dpt_large":
        return build_dpt_large_adapter(device, dpt_large_snapshot)
    if model_id == "depth_anything_v1":
        return build_depth_anything_adapter(device, depth_anything_snapshot)
    if model_id == "zoedepth":
        return build_zoedepth_adapter(
            device,
            zoedepth_repository,
            zoedepth_checkpoint,
            zoedepth_midas_repository,
        )
    raise ValueError(f"unsupported depth model: {model_id}")


def release_adapter(adapter: Predictor) -> None:
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
