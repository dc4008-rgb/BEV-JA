"""v23 objective dispatch and fixed, pre-optimization train-only calibration.

The calibration scales are reporting/numerical normalization, not an attack
strength mechanism. Detector gradients are subsequently normalized per model.
No optimized v20 capture, validation output, or gradient selects a scale.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from training.attack_objectives import capture_contextual_loss, validate_phase


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FLOOR = 1e-6


def _parent():
    from training.inputs import setup_parent_paths
    setup_parent_paths()
    import trainer
    return trainer


def _objective(value: str) -> None:
    if value not in ("C0", "C3"):
        raise ValueError("attack objective must be exactly C0 or C3")


@dataclass(frozen=True)
class CapturedLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    live_evidence: dict[str, torch.Tensor] = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.components.items()}


class CalibrationFailure(RuntimeError):
    """Preserve calibration's actual callback prefix; never infer failed forwards."""

    def __init__(self, error: Exception, progress_evidence: dict[str, Any]):
        self.progress_evidence = copy.deepcopy(progress_evidence)
        self.progress_evidence.update(
            status="calibration_failed_retained",
            failure={"type": type(error).__name__, "message": str(error)})
        super().__init__(f"calibration failed: {type(error).__name__}: {error}")


def model_loss(config, contract, model_id, adapter, context, image, support,
               cache, sample_id, *, attack_objective: str, phase: str):
    """C0 delegates frozen Ours; C3 uses Contextual math on both core detectors.

    train_pre must receive its real differentiable image. The two scoring
    phases may execute under no_grad and never manufacture a surrogate graph.
    C3 live_evidence is ephemeral; a caller retaining it must detach and clone.
    """
    _objective(attack_objective)
    requires_graph = phase == "train_pre"
    validate_phase(phase, requires_graph)
    if requires_graph and (not torch.is_grad_enabled() or not image.requires_grad):
        raise RuntimeError("train_pre requires the original differentiable image")
    if attack_objective == "C0":
        return _parent().model_loss(config, contract, model_id, adapter, context,
                                    image, support, cache, sample_id)
    if config.method_id != "ours":
        raise ValueError("v23 C3 is bound to the ours training objective")
    if model_id not in ("fasterrcnn", "ssd300") or getattr(adapter, "model_id", None) != model_id:
        raise ValueError("C3 is bound to the matching two core detector adapters")
    counted = getattr(adapter, "contextual_forward", None)
    if callable(counted):
        loss, evidence, diagnostics = counted(
            capture_contextual_loss, image, context.object_box,
            require_attack_graph=requires_graph, phase=phase)
    else:
        loss, evidence, diagnostics = capture_contextual_loss(
            adapter, image, context.object_box,
            require_attack_graph=requires_graph, phase=phase)
    return CapturedLoss(loss, {"contextual_car_class_probability": loss},
                        diagnostics, evidence)


def _fixed_calibrators():
    parent = _parent()
    from data import load_training_manifest
    path = ROOT / "parent_v17/01_公共数据集/runtime_v1/train_validation_manifest.csv"
    return parent.calibration_records(load_training_manifest(path))


def _record_identity(record) -> dict[str, str]:
    # Local source path remapping is allowed; source contents and geometry are not.
    return {name: getattr(record, name) for name in
            ("sample_id", "dataset", "split", "source_sha256", "road_quad_norm")}


def _tensor_sha(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous()
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def calibrate_normalizers(config, contract, carrier, adapters, records, car,
                          car_mask, device, seed, verified, benign_depth_cache,
                          *, attack_objective: str):
    """Four fixed train scenes, unchanged initial crop, no EOT or backward.

    Scale per model = max(mean(abs(raw loss)), 1e-6), exactly v17's rule.
    Evidence explicitly retains every raw value and whether the floor applied.
    The caller must save evidence before optimization and bind its file hash.
    """
    _objective(attack_objective)
    parent = _parent()
    from road_patch_contract import RoadRGBView
    records = tuple(records)
    expected = _fixed_calibrators()
    if (len(records) != 4 or len({r.sample_id for r in records}) != 4
            or Counter((r.dataset, r.split) for r in records)
            != Counter({("kitti", "train"): 2, ("nuscenes", "train"): 2})
            or [_record_identity(r) for r in records]
            != [_record_identity(r) for r in expected]):
        raise ValueError("calibration requires the original fixed four train rows in order")
    if tuple(config.train_models) != ("fasterrcnn", "ssd300"):
        raise ValueError("v23 calibration is frozen to the two det-only core models")
    if config.method_id != "ours":
        raise ValueError("v23 calibration is bound to ours")
    if not isinstance(carrier, RoadRGBView) or carrier.rgb.requires_grad:
        raise ValueError("calibration requires the detached unoptimized RoadRGBView")
    original_crop = carrier.rgb.detach().clone()
    losses = {model_id: [] for model_id in config.train_models}
    sample_evidence = []
    progress = {
        "schema": "v23_fixed_train_loss_calibration_prefix_v1",
        "attack_objective": attack_objective, "samples": sample_evidence,
        "model_loss_attempts": 0, "model_loss_completed": 0,
        "autograd_calls": 0, "current": None,
        "count_semantics": "model_loss callbacks, not actual forward counts; a failed callback may already have executed a model forward",
    }
    try:
        with torch.no_grad():
            for record_index, record in enumerate(records):
                progress["current"] = {
                    "record_index": record_index, "sample_id": record.sample_id,
                    "model_id": None, "stage": "build_context"}
                context = parent.build_context(record, car, car_mask, device, verified)
                if record_index == 0:
                    progress["current"]["stage"] = "verify_unoptimized_crop"
                    expected_crop = parent.extract_road_patch(context.background, context.road_quad)
                    if not torch.equal(carrier.rgb, expected_crop):
                        raise ValueError("calibration crop is not the fixed first train road crop")
                progress["current"]["stage"] = "render_without_eot"
                rendered = carrier.render(
                    context, eot=None,
                    generator=parent.make_generator(device, seed + record_index))
                row = {**_record_identity(record), "eot": None, "models": {}}
                sample_evidence.append(row)
                for model_id in config.train_models:
                    progress["current"].update(model_id=model_id, stage="model_loss")
                    progress["model_loss_attempts"] += 1
                    result = model_loss(
                        config, contract, model_id, adapters[model_id], context,
                        rendered.image, rendered.support, benign_depth_cache,
                        record.sample_id, attack_objective=attack_objective, phase="calibration")
                    progress["model_loss_completed"] += 1
                    progress["current"]["stage"] = "validate_returned_loss"
                    if result.total.ndim != 0 or result.total.requires_grad:
                        raise RuntimeError("calibration must produce a no-grad scalar")
                    raw = float(result.total)
                    if not math.isfinite(raw):
                        row["models"][model_id] = {"raw_loss_nonfinite": repr(raw)}
                        raise RuntimeError(f"non-finite calibration loss: {model_id}")
                    losses[model_id].append(abs(raw))
                    row["models"][model_id] = {
                        "raw_loss": raw, "absolute_loss": abs(raw),
                        "components": result.scalars(),
                        "diagnostics": getattr(result, "diagnostics", {}),
                    }
                    del result
        progress["current"] = {"stage": "verify_crop_unchanged"}
        if not torch.equal(original_crop, carrier.rgb):
            raise RuntimeError("calibration changed the initial crop")
    except Exception as error:
        raise CalibrationFailure(error, progress) from error
    means = {name: sum(values) / len(values) for name, values in losses.items()}
    normalizers = {name: max(value, CALIBRATION_FLOOR) for name, value in means.items()}
    if any(not math.isfinite(value) or value <= 0 for value in normalizers.values()):
        raise RuntimeError("invalid objective normalizer")
    model_evidence = {
        name: {"raw_mean_absolute": means[name], "normalizer": value,
               "floor": CALIBRATION_FLOOR,
               "floor_applied": means[name] < CALIBRATION_FLOOR,
               "raw_sample_absolute_losses": losses[name]}
        for name, value in normalizers.items()
    }
    evidence = {
        "schema": "v23_fixed_train_loss_calibration_v1",
        "attack_objective": attack_objective,
        "formula": "max(mean(abs(raw_loss)), 1e-6)",
        "sample_ids": [r.sample_id for r in records], "samples": sample_evidence,
        "models": model_evidence, "normalizers": normalizers,
        "initial_crop_tensor_sha256": _tensor_sha(original_crop),
        "initial_crop_shape": list(original_crop.shape),
        "initial_crop_dtype": str(original_crop.dtype),
        "source": "fixed_first_train_road_crop_before_vae_noise_or_optimization",
        "eot": None, "victim_forward_calls": 8, "autograd_calls": 0,
        "model_loss_attempts": progress["model_loss_attempts"],
        "model_loss_completed": progress["model_loss_completed"],
        "updated_parameters": 0, "test_data_opened": False,
        "validation_data_opened": False, "optimized_v20_states_used": False,
        "calibration_is_not_attack_strength_evidence": True,
    }
    return normalizers, evidence
