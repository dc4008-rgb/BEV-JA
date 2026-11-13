#!/usr/bin/env python3
"""Unified evaluator for frozen normalized attack artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from advrm_insertion import insert_car, load_car, load_scene, parse_road_quad
from artifacts import implementation_fingerprint, load_artifact, sha256
from carriers import FrozenRgbCarrier, SceneContext
from data import SceneRecord, load_test_manifest, load_training_manifest, verify_source
from model_adapters import (
    DEPTH_MODEL_IDS,
    DETECTION_MODEL_IDS,
    build_depth_adapter,
    build_detection_adapter,
)
from raw_evidence import RawEvidenceStore, SCHEMA_VERSION as RAW_SCHEMA_VERSION, model_binding

CANDIDATE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CANDIDATE_ROOT / "00_冻结协议"))
from protocol_runtime import load_protocol, require_run_allowed


VEHICLE_CLASS_IDS = (3, 6, 8)
CAR_CLASS_IDS = (3,)
DOMAINS = ("kitti", "nuscenes", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--car", type=Path, required=True)
    parser.add_argument("--car-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-evidence-root", type=Path, help="Shared content-addressed raw store; default OUTPUT/raw")
    parser.add_argument(
        "--raw-evidence-policy",
        choices=("full", "metrics-only"),
        default="full",
        help="metrics-only retains scalar diagnostics but no replay tensors; formal use must be protocol-authorized.",
    )
    parser.add_argument("--model-manifest", type=Path, default=Path(__file__).resolve().parent.parent / "02_模型" / "model_manifest.json")
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--purpose", choices=("validation", "diagnostic"), default="validation")
    parser.add_argument(
        "--models",
        default=",".join((*DETECTION_MODEL_IDS, *DEPTH_MODEL_IDS)),
    )
    parser.add_argument("--mode", choices=("pilot", "production"), required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_models(raw: str) -> tuple[str, ...]:
    models = tuple(value.strip() for value in raw.split(",") if value.strip())
    allowed = set((*DETECTION_MODEL_IDS, *DEPTH_MODEL_IDS))
    if not models or len(models) != len(set(models)) or set(models) - allowed:
        raise ValueError(f"invalid unique model list: {models}")
    return models


def evaluation_records(path: Path, mode: str, limit: int) -> tuple[SceneRecord, ...]:
    if mode == "production":
        if limit:
            raise ValueError("production evaluation cannot limit the frozen test set")
        return tuple(load_test_manifest(path))
    if limit < 1 or limit > 20:
        raise ValueError("pilot requires limit-per-dataset in [1,20]")
    records = load_training_manifest(path)
    selected = tuple(
        record
        for dataset in ("kitti", "nuscenes")
        for record in sorted(
            (
                value
                for value in records
                if value.dataset == dataset and value.split == "validation"
            ),
            key=lambda value: value.sample_id,
        )[:limit]
    )
    if Counter(record.dataset for record in selected) != {
        "kitti": limit,
        "nuscenes": limit,
    }:
        raise ValueError("pilot selection is missing a dataset")
    return selected


def box_iou(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not boxes.numel():
        return boxes.new_empty((0,))
    left_top = torch.maximum(boxes[:, :2], target[:2])
    right_bottom = torch.minimum(boxes[:, 2:], target[2:])
    extent = (right_bottom - left_top).clamp_min(0.0)
    intersection = extent[:, 0] * extent[:, 1]
    box_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0.0)
    target_area = (target[2] - target[0]).clamp_min(0.0) * (
        target[3] - target[1]
    ).clamp_min(0.0)
    return intersection / (box_area + target_area - intersection).clamp_min(1e-12)


def detection_candidates(
    output: dict[str, torch.Tensor],
    target_box: tuple[int, int, int, int],
    class_ids: Sequence[int],
) -> list[dict[str, float | int]]:
    target = output["boxes"].new_tensor(target_box)
    allowed = torch.tensor(class_ids, device=output["labels"].device)
    keep = torch.isin(output["labels"], allowed)
    boxes = output["boxes"][keep]
    scores = output["scores"][keep]
    labels = output["labels"][keep]
    overlaps = box_iou(boxes, target)
    intersecting = overlaps.gt(0.0)
    return [
        {"score": float(score), "iou": float(overlap), "label": int(label)}
        for score, overlap, label in zip(
            scores[intersecting], overlaps[intersecting], labels[intersecting]
        )
    ]


def average_precision(
    scene_candidates: Iterable[tuple[str, Sequence[dict[str, float | int]]]],
    ground_truth_count: int,
) -> float:
    predictions = sorted(
        (
            (float(candidate["score"]), sample_id, float(candidate["iou"]))
            for sample_id, candidates in scene_candidates
            for candidate in candidates
        ),
        key=lambda value: value[0],
        reverse=True,
    )
    if ground_truth_count < 1:
        raise ValueError("ground_truth_count must be positive")
    if not predictions:
        return 0.0
    matched: set[str] = set()
    true_positive = []
    false_positive = []
    for _, sample_id, overlap in predictions:
        is_true = overlap >= 0.5 and sample_id not in matched
        if is_true:
            matched.add(sample_id)
        true_positive.append(float(is_true))
        false_positive.append(float(not is_true))
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = np.concatenate(([0.0], tp / ground_truth_count, [1.0]))
    precision = np.concatenate(([0.0], tp / np.maximum(tp + fp, 1e-12), [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def tnr_detection(clean_ap: float, adversarial_ap: float) -> float:
    if clean_ap <= 0:
        raise ValueError("clean AP must be positive for TNR-det")
    return (clean_ap - adversarial_ap) / clean_ap


def depth_metrics(
    adversarial_depth: torch.Tensor,
    benign_depth: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    semantics: str = "unspecified_nonnegative_depth_quantity",
    log_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Compute the frozen clean-reference metrics and bounded diagnostics.

    Primary metrics use the same static object ROI and the validity mask
    ``clean > 0``.  This mask is determined only from the clean reference, never
    from the adversarial error.  Invalid, negative, or non-finite adversarial
    values are not clipped or silently removed.  SRE remains a diagnostic and
    is never substituted for AbsRel or RMSE.
    """
    if not math.isfinite(log_epsilon) or log_epsilon <= 0:
        raise ValueError("log_epsilon must be finite and positive")
    if adversarial_depth.ndim == 3:
        adversarial_depth = adversarial_depth.unsqueeze(1)
    if benign_depth.ndim == 3:
        benign_depth = benign_depth.unsqueeze(1)
    if adversarial_depth.shape != benign_depth.shape:
        raise ValueError("adversarial and benign depth shapes differ")
    roi = F.interpolate(
        object_mask.to(device=adversarial_depth.device, dtype=torch.float32),
        adversarial_depth.shape[-2:],
        mode="nearest",
    ).bool()
    adversarial = adversarial_depth[roi].double()
    benign = benign_depth[roi].double()
    if not adversarial.numel():
        raise ValueError("depth ROI is empty")
    if not torch.isfinite(adversarial).all() or not torch.isfinite(benign).all():
        raise ValueError("depth ROI contains non-finite values")
    if (adversarial < 0).any() or (benign < 0).any():
        raise ValueError("depth evaluation quantity contains negative values")
    difference = adversarial - benign
    valid = benign > 0
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise ValueError("clean-reference depth ROI has no positive valid pixels")
    valid_adversarial = adversarial[valid]
    valid_benign = benign[valid]
    valid_difference = valid_adversarial - valid_benign
    abs_rel = valid_difference.abs() / valid_benign
    log_difference = torch.log(valid_adversarial + log_epsilon) - torch.log(
        valid_benign + log_epsilon
    )
    rmse = valid_difference.square().mean().sqrt()
    rmse_log = log_difference.square().mean().sqrt()
    delta_sigma = log_difference.mean()
    pair_scale = adversarial.abs() + benign.abs()
    symmetric = torch.where(
        pair_scale > 0,
        2.0 * difference.abs() / pair_scale,
        torch.zeros_like(pair_scale),
    )
    signed_symmetric = torch.where(
        pair_scale > 0,
        2.0 * difference / pair_scale,
        torch.zeros_like(pair_scale),
    )
    rms_difference = difference.square().mean().sqrt()
    rms_scale = adversarial.square().mean().sqrt() + benign.square().mean().sqrt()
    symmetric_rmse = 0.0 if float(rms_scale) == 0.0 else float(2.0 * rms_difference / rms_scale)
    return {
        "depth_representation": semantics,
        "roi_pixels": int(adversarial.numel()),
        "valid_pixels": valid_pixels,
        "valid_fraction": valid_pixels / int(adversarial.numel()),
        "validity_policy": "static_object_roi_and_clean_reference_strictly_positive",
        "log_epsilon": log_epsilon,
        "abs_rel_clean_ref": float(abs_rel.mean()),
        "rmse_clean_ref": float(rmse),
        "rmse_log_clean_ref": float(rmse_log),
        "delta_sigma": float(delta_sigma),
        "symmetric_rel_clean_ref": float(symmetric.mean()),
        "symmetric_rmse_clean_ref": symmetric_rmse,
        "signed_symmetric_rel": float(signed_symmetric.mean()),
        "zero_pair_pixels": int(((adversarial == 0) & (benign == 0)).sum()),
        "adversarial_zero_pixels": int((adversarial == 0).sum()),
        "benign_zero_pixels": int((benign == 0).sum()),
        "negative_pixels": 0,
        "nonfinite_pixels": 0,
        "sum_symmetric_rel": float(symmetric.sum()),
        "sum_signed_symmetric_rel": float(signed_symmetric.sum()),
        "sum_squared_error": float(difference.square().sum()),
        "sum_abs_rel_valid": float(abs_rel.sum()),
        "sum_squared_error_valid": float(valid_difference.square().sum()),
        "sum_squared_log_error_valid": float(log_difference.square().sum()),
        "sum_log_difference_valid": float(log_difference.sum()),
        "rms_adversarial": float(adversarial.square().mean().sqrt()),
        "rms_benign": float(benign.square().mean().sqrt()),
    }


def percentile_interval(values: Sequence[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def detection_summary(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    *,
    allow_zero_reference: bool = False,
) -> dict[str, Any]:
    clean = average_precision(
        ((record["sample_id"], record["clean_candidates"]) for record in records),
        len(records),
    )
    adversarial = average_precision(
        ((record["sample_id"], record["adversarial_candidates"]) for record in records),
        len(records),
    )
    car_clean = average_precision(
        ((record["sample_id"], record["clean_car_candidates"]) for record in records),
        len(records),
    )
    car_adversarial = average_precision(
        ((record["sample_id"], record["adversarial_car_candidates"]) for record in records),
        len(records),
    )
    if clean <= 0 and not allow_zero_reference:
        raise RuntimeError("clean target-vehicle AP is zero")
    generator = np.random.default_rng(seed)
    boot_clean = []
    boot_adversarial = []
    boot_tnr = []
    for _ in range(bootstrap_samples):
        indices = generator.integers(0, len(records), len(records))
        clean_scenes = [
            (f"bootstrap_{position}", records[index]["clean_candidates"])
            for position, index in enumerate(indices)
        ]
        adversarial_scenes = [
            (f"bootstrap_{position}", records[index]["adversarial_candidates"])
            for position, index in enumerate(indices)
        ]
        clean_value = average_precision(clean_scenes, len(indices))
        adversarial_value = average_precision(adversarial_scenes, len(indices))
        boot_clean.append(clean_value)
        boot_adversarial.append(adversarial_value)
        if clean_value > 0:
            boot_tnr.append(tnr_detection(clean_value, adversarial_value))
    return {
        "scene_count": len(records),
        "target_vehicle_ap50_clean": clean,
        "target_vehicle_ap50_adversarial": adversarial,
        "tnr_det": tnr_detection(clean, adversarial) if clean > 0 else None,
        "tnr_defined": clean > 0,
        "target_vehicle_ap50_drop": clean - adversarial,
        "car_only_ap50_clean": car_clean,
        "car_only_ap50_adversarial": car_adversarial,
        "bootstrap_95_ci": {
            "target_vehicle_ap50_clean": percentile_interval(boot_clean),
            "target_vehicle_ap50_adversarial": percentile_interval(boot_adversarial),
            "tnr_det": percentile_interval(boot_tnr) if boot_tnr else None,
        },
    }


def depth_summary(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metric_names = (
        "abs_rel_clean_ref",
        "rmse_clean_ref",
        "rmse_log_clean_ref",
        "delta_sigma",
        "symmetric_rel_clean_ref",
        "symmetric_rmse_clean_ref",
        "signed_symmetric_rel",
    )
    macro = {
        name: {
            "mean": statistics.fmean(float(record[name]) for record in records),
            "std": statistics.pstdev(float(record[name]) for record in records),
        }
        for name in metric_names
    }
    pixel_count = sum(int(record["roi_pixels"]) for record in records)
    valid_count = sum(int(record["valid_pixels"]) for record in records)
    if valid_count <= 0:
        raise ValueError("depth summary has no valid clean-reference pixels")
    pooled = {
        "abs_rel_clean_ref": sum(float(record["sum_abs_rel_valid"]) for record in records) / valid_count,
        "rmse_clean_ref": math.sqrt(sum(float(record["sum_squared_error_valid"]) for record in records) / valid_count),
        "rmse_log_clean_ref": math.sqrt(sum(float(record["sum_squared_log_error_valid"]) for record in records) / valid_count),
        "delta_sigma": sum(float(record["sum_log_difference_valid"]) for record in records) / valid_count,
        "symmetric_rel_clean_ref": sum(float(record["sum_symmetric_rel"]) for record in records) / pixel_count,
        "signed_symmetric_rel": sum(float(record["sum_signed_symmetric_rel"]) for record in records) / pixel_count,
    }
    generator = np.random.default_rng(seed)
    bootstrap = {name: [] for name in metric_names}
    for _ in range(bootstrap_samples):
        indices = generator.integers(0, len(records), len(records))
        for name in metric_names:
            bootstrap[name].append(
                statistics.fmean(float(records[index][name]) for index in indices)
            )
    for name in metric_names:
        macro[name]["bootstrap_95_ci"] = percentile_interval(bootstrap[name])
    return {
        "scene_count": len(records),
        "roi_pixels": pixel_count,
        "valid_pixels": valid_count,
        "valid_fraction": valid_count / pixel_count,
        "reference": "same_model_prediction_on_benign_scene_with_inserted_car",
        "metric_policy": "paper_named_clean_reference_metrics_plus_bounded_sre_diagnostic",
        "macro": macro,
        "pooled_pixel": pooled,
    }


def zero_step_comparisons(
    task: str,
    context: SceneContext,
    clean_output: Any,
    zero_output: Any,
    final_output: Any,
    depth_semantics: str = "unspecified_nonnegative_depth_quantity",
) -> dict[str, dict[str, Any]]:
    comparisons = {}
    for name, reference, evaluated in (
        ("original_to_zero_step", clean_output, zero_output),
        ("zero_step_to_final", zero_output, final_output),
    ):
        if task == "detection":
            comparisons[name] = {
                "clean_candidates": detection_candidates(reference, context.object_box, VEHICLE_CLASS_IDS),
                "adversarial_candidates": detection_candidates(evaluated, context.object_box, VEHICLE_CLASS_IDS),
                "clean_car_candidates": detection_candidates(reference, context.object_box, CAR_CLASS_IDS),
                "adversarial_car_candidates": detection_candidates(evaluated, context.object_box, CAR_CLASS_IDS),
            }
        elif task == "depth":
            comparisons[name] = depth_metrics(
                evaluated, reference, context.object_mask, semantics=depth_semantics
            )
        else:
            raise ValueError(f"unsupported control task: {task}")
    return comparisons


def zero_step_summary(
    records: list[dict[str, Any]],
    task: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for comparison in ("original_to_zero_step", "zero_step_to_final"):
        selected = [
            {"sample_id": record["sample_id"], **record["zero_step_control"][comparison]}
            for record in records
        ]
        summary = (
            detection_summary(selected, bootstrap_samples, seed, allow_zero_reference=True)
            if task == "detection" else depth_summary(selected, bootstrap_samples, seed)
        )
        summary["reference"] = "original_car" if comparison == "original_to_zero_step" else "zero_step_same_carrier"
        summary["evaluated"] = "zero_step_same_carrier" if comparison == "original_to_zero_step" else "final_carrier"
        result[comparison] = summary
    return result


def build_context(
    record: SceneRecord,
    car: torch.Tensor,
    car_mask: torch.Tensor,
    device: torch.device,
    verified: set[str],
) -> SceneContext:
    if record.sample_id not in verified:
        verify_source(record)
        verified.add(record.sample_id)
    background = load_scene(record.source_path, device)
    road_quad = parse_road_quad(record.road_quad_norm)
    benign = insert_car(background, car, car_mask, road_quad)
    return SceneContext(
        background=background,
        benign=benign.image,
        car=car,
        car_mask=car_mask,
        object_mask=benign.mask,
        object_box=benign.geometry.box_xyxy,
        road_quad=road_quad,
    )


def depth_predictions_with_raw(
    adapter: Any, images: dict[str, torch.Tensor], supports: dict[str, torch.Tensor],
    object_mask: torch.Tensor, store: RawEvidenceStore | None, bindings: dict[str, Any],
    parameters: dict[str, Any], *, require_complete: bool, raw_policy: str = "full",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Capture a single inference per state in the historical inference order.

    Toy adapters can exercise metrics without diagnostics only outside formal
    evaluation, and their records explicitly declare incomplete raw evidence.
    """
    diagnostics = getattr(adapter, "predict_with_diagnostics", None)
    diagnostics_available = callable(diagnostics)
    if require_complete and not diagnostics_available:
        raise ValueError("formal depth evaluation requires predict_with_diagnostics")
    predictions = {}
    states = {}
    evaluation_semantics: str | None = None
    for state in ("original", "final", "zero_step"):
        if diagnostics_available:
            prediction = diagnostics(images[state])
            if "evaluation_output" not in prediction.stages:
                raise ValueError("depth diagnostics omit evaluation_output")
            state_semantics = str(prediction.metadata.get("evaluation_semantics", ""))
            if not state_semantics:
                raise ValueError("depth diagnostics omit evaluation semantics")
            if evaluation_semantics is not None and state_semantics != evaluation_semantics:
                raise ValueError("depth evaluation semantics changed across states")
            evaluation_semantics = state_semantics
            predictions[state] = prediction.stages["evaluation_output"]
            if raw_policy == "full":
                if store is None:
                    raise ValueError("full raw policy requires a raw evidence store")
                states[state] = store.put_state(
                    state, prediction, images[state], object_mask, supports[state], bindings, parameters,
                )
        else:
            predictions[state] = adapter.predict(images[state])
    return predictions, {
        "schema_version": RAW_SCHEMA_VERSION,
        "available": diagnostics_available and raw_policy == "full",
        "diagnostics_available": diagnostics_available,
        "storage_policy": raw_policy,
        "states": states,
        "evaluation_semantics": evaluation_semantics,
        "metric_input_stage": "evaluation_output" if diagnostics_available else "adapter_predict_toy_only",
        "missing_reasons": (
            [] if diagnostics_available and raw_policy == "full"
            else ["raw_tensors_disabled_for_full_budget_validation"] if diagnostics_available
            else ["adapter_has_no_predict_with_diagnostics; numeric-only toy/pilot evidence"]
        ),
    }


def guard_evaluation_args(args: argparse.Namespace, artifact: Any) -> Any:
    """Bind every evaluation entry point to the candidate's allowed purpose."""
    if getattr(args, "protocol_dir", None) is None:
        raise ValueError("candidate evaluation requires explicit --protocol-dir")
    ctx = load_protocol(CANDIDATE_ROOT, args.protocol_dir)
    purpose = "formal" if args.mode == "production" else getattr(args, "purpose", "validation")
    config_id = artifact.metadata["config_id"]
    require_run_allowed(ctx, purpose=purpose, config_id=config_id)
    row = ctx.row(config_id)
    for key, expected_value in {
        "protocol_id": ctx.protocol["protocol_id"], "run_id": ctx.run_id,
        "protocol_sha256": sha256(ctx.protocol_path),
        "candidate_fingerprint": ctx.protocol["candidate_fingerprint"],
        "run_matrix_sha256": ctx.protocol["run_matrix_sha256"],
        "experiment_purpose": purpose,
    }.items():
        if artifact.metadata.get(key) != expected_value:
            raise ValueError(f"artifact provenance differs from candidate: {key}")
    expected = {
        "manifest": ctx.path(ctx.protocol["test_manifest_path" if args.mode == "production" else "training_manifest_path"]),
        "car": ctx.path("01_公共数据集/car_00.jpg"),
        "car_mask": ctx.path("01_公共数据集/car_00_mask.jpg"),
        "model_manifest": ctx.path(ctx.protocol["model_manifest_path"]),
        "artifact": ctx.path(row["artifact_output"]),
        "output": ctx.path(row["evaluation_output"]),
    }
    for name, path in expected.items():
        actual = getattr(args, name, None)
        if actual is None or Path(actual).resolve() != path:
            raise ValueError(f"evaluation {name} differs from the selected candidate protocol")
    if set(parse_models(args.models)) != set(ctx.evaluation_model_ids):
        raise ValueError("candidate evaluation must include exactly the frozen eight models")
    if args.bootstrap_samples != ctx.protocol["metrics"]["bootstrap_samples"] or args.bootstrap_seed != ctx.protocol["budget"]["seed"]:
        raise ValueError("evaluation bootstrap policy differs from the candidate")
    if args.mode == "production":
        expected_policy = ctx.protocol.get("storage_policy", {}).get(
            "raw_depth_tensors", "full"
        )
        if getattr(args, "raw_evidence_policy", "full") != expected_policy:
            raise ValueError("production raw-evidence policy differs from the sealed protocol")
    raw_root = Path(getattr(args, "raw_evidence_root", None) or (args.output / "raw")).resolve()
    if raw_root not in {ctx.path(f"05_运行/{ctx.run_id}/raw_evidence"), (expected["output"] / "raw").resolve()}:
        raise ValueError("raw evidence root must be the candidate shared run store or OUTPUT/raw")
    return ctx


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    if args.bootstrap_samples != 1000:
        raise ValueError("frozen protocol requires exactly 1000 bootstrap samples")
    models = parse_models(args.models)
    artifact = load_artifact(args.artifact)
    ctx = guard_evaluation_args(args, artifact)
    records = evaluation_records(args.manifest, args.mode, args.limit_per_dataset)
    if artifact.zero_step is None:
        raise ValueError("current evaluation requires a saved zero-step control; use the historical evaluator for legacy artifacts")
    expected_mode = "production" if args.mode == "production" else "pilot"
    if artifact.metadata.get("run_mode") != expected_mode:
        raise ValueError(f"{args.mode} evaluation requires a {expected_mode} artifact")
    current_implementation, _ = implementation_fingerprint(
        Path(__file__).resolve().parent.parent / "03_方法" / "production"
    )
    if artifact.metadata["implementation_sha256"] != current_implementation:
        raise ValueError("artifact was produced by a different implementation")
    device = torch.device(args.device)
    car, car_mask = load_car(args.car, args.car_mask, device)
    carrier = FrozenRgbCarrier(
        artifact.canonical_rgb.to(device), artifact.metadata["carrier_kind"]
    ).to(device)
    zero_carrier = FrozenRgbCarrier(
        artifact.zero_step["canonical_rgb"].to(device), artifact.metadata["carrier_kind"]
    ).to(device)
    verified: set[str] = set()
    per_sample: list[dict[str, Any]] = []
    summaries = []
    started = time.monotonic()
    raw_root = getattr(args, "raw_evidence_root", None) or (args.output / "raw")
    raw_policy = getattr(args, "raw_evidence_policy", "full")
    raw_store = RawEvidenceStore(raw_root) if raw_policy == "full" else None
    model_manifest_path = getattr(args, "model_manifest", None) or (Path(__file__).resolve().parent.parent / "02_模型" / "model_manifest.json")
    protocol_dir = getattr(args, "protocol_dir", None)
    protocol_sha256 = sha256(ctx.protocol_path) if ctx is not None else (sha256(Path(protocol_dir) / "protocol.json") if protocol_dir else None)
    common_bindings = {
        "config_id": artifact.metadata["config_id"],
        "manifest_sha256": sha256(args.manifest),
        "artifact_sha256": artifact.metadata["artifact_sha256"],
        "artifact_metadata_sha256": sha256(args.artifact / "metadata.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "model_adapters_sha256": sha256(Path(__file__).resolve().parent.parent / "02_模型" / "model_adapters.py"),
        "raw_evidence_sha256": sha256(Path(__file__).resolve().parent / "raw_evidence.py"),
        "car_sha256": sha256(args.car), "car_mask_sha256": sha256(args.car_mask),
        "protocol_sha256": protocol_sha256,
    }

    for model_id in models:
        task = "detection" if model_id in DETECTION_MODEL_IDS else "depth"
        adapter = (
            build_detection_adapter(model_id, device)
            if task == "detection"
            else build_depth_adapter(model_id, device, model_manifest=model_manifest_path)
        )
        depth_model_binding = (
            model_binding(model_manifest_path, model_id)
            if task == "depth" and callable(getattr(adapter, "predict_with_diagnostics", None)) else {}
        )
        model_records = []
        model_started = time.monotonic()
        with torch.no_grad():
            for index, record in enumerate(records, start=1):
                context = build_context(record, car, car_mask, device, verified)
                final_render = carrier.render(
                    context,
                    eot=None,
                    generator=torch.Generator(
                        device=device.type if device.type == "cuda" else "cpu"
                    ).manual_seed(0),
                )
                adversarial = final_render.image
                zero_render = zero_carrier.render(
                    context, eot=None,
                    generator=torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(0),
                )
                zero_image = zero_render.image
                if task == "detection":
                    clean_output = adapter.predict(context.benign)[0]
                    adversarial_output = adapter.predict(adversarial)[0]
                    zero_output = adapter.predict(zero_image)[0]
                    result = {
                        "sample_id": record.sample_id,
                        "dataset": record.dataset,
                        "split": record.split,
                        "model": model_id,
                        "task": task,
                        "target_box": list(context.object_box),
                        "clean_candidates": detection_candidates(
                            clean_output, context.object_box, VEHICLE_CLASS_IDS
                        ),
                        "adversarial_candidates": detection_candidates(
                            adversarial_output, context.object_box, VEHICLE_CLASS_IDS
                        ),
                        "clean_car_candidates": detection_candidates(
                            clean_output, context.object_box, CAR_CLASS_IDS
                        ),
                        "adversarial_car_candidates": detection_candidates(
                            adversarial_output, context.object_box, CAR_CLASS_IDS
                        ),
                        "zero_step_control": zero_step_comparisons(
                            task, context, clean_output, zero_output, adversarial_output,
                        ),
                    }
                else:
                    depth_outputs, raw_evidence = depth_predictions_with_raw(
                        adapter,
                        {"original": context.benign, "final": adversarial, "zero_step": zero_image},
                        {"original": torch.zeros_like(context.object_mask), "final": final_render.support, "zero_step": zero_render.support},
                        context.object_mask, raw_store,
                        {**common_bindings, **depth_model_binding, "sample_id": record.sample_id,
                         "dataset": record.dataset, "split": record.split, "model": model_id,
                         "source_sha256": record.source_sha256, "source_path": str(record.source_path)},
                        {"target_box": list(context.object_box), "road_quad": context.road_quad,
                         "carrier_kind": artifact.metadata["carrier_kind"], "eot": None,
                         "render_seed": 0,
                         "metric_policy": "bounded_symmetric_relative_no_clipping_no_pixel_filtering",
                         "depth_reference": "same_model_prediction_on_benign_scene_with_inserted_car",
                         "original_carrier_support": "zero_no_adversarial_carrier"},
                        require_complete=args.mode == "production",
                        raw_policy=raw_policy,
                    )
                    benign_depth = depth_outputs["original"]
                    adversarial_depth = depth_outputs["final"]
                    zero_depth = depth_outputs["zero_step"]
                    depth_semantics = str(raw_evidence["evaluation_semantics"])
                    result = {
                        "sample_id": record.sample_id,
                        "dataset": record.dataset,
                        "split": record.split,
                        "model": model_id,
                        "task": task,
                        "target_box": list(context.object_box),
                        **depth_metrics(
                            adversarial_depth, benign_depth, context.object_mask,
                            semantics=depth_semantics,
                        ),
                        "zero_step_control": zero_step_comparisons(
                            task, context, benign_depth, zero_depth, adversarial_depth,
                            depth_semantics,
                        ),
                        "raw_depth_evidence": raw_evidence,
                    }
                result["schema_version"] = 2
                result["config_id"] = artifact.metadata["config_id"]
                result["source_sha256"] = record.source_sha256
                model_records.append(result)
                per_sample.append(result)
                print(
                    f"EVAL {model_id} {index}/{len(records)} {record.sample_id}",
                    flush=True,
                )
        domain_summaries = {}
        for domain_index, domain in enumerate(DOMAINS):
            selected = (
                model_records
                if domain == "combined"
                else [record for record in model_records if record["dataset"] == domain]
            )
            domain_summaries[domain] = (
                detection_summary(
                    selected, args.bootstrap_samples, args.bootstrap_seed + domain_index
                )
                if task == "detection"
                else depth_summary(
                    selected, args.bootstrap_samples, args.bootstrap_seed + domain_index
                )
            )
            domain_summaries[domain]["zero_step_control"] = zero_step_summary(
                selected, task, args.bootstrap_samples, args.bootstrap_seed + domain_index,
            )
        summaries.append(
            {
                "model": model_id,
                "task": task,
                "elapsed_seconds": time.monotonic() - model_started,
                "domains": domain_summaries,
            }
        )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "model_sample_records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in per_sample:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    expected_records = len(records) * len(models)
    if len(per_sample) != expected_records:
        raise RuntimeError(
            f"model-sample record count mismatch: {len(per_sample)} != {expected_records}"
        )
    report = {
        "schema_version": 2,
        "status": "complete",
        "mode": args.mode,
        "config_id": artifact.metadata["config_id"],
        "protocol_sha256": protocol_sha256,
        "purpose": "formal" if args.mode == "production" else getattr(args, "purpose", "validation"),
        "model_manifest_sha256": sha256(model_manifest_path),
        "artifact_metadata_sha256": sha256(args.artifact / "metadata.json"),
        "artifact_sha256": artifact.metadata["artifact_sha256"],
        "manifest_sha256": sha256(args.manifest),
        "models": list(models),
        "scene_count": len(records),
        "model_count": len(models),
        "model_sample_record_count": len(per_sample),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "records_sha256": sha256(records_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "raw_evidence_sha256": sha256(Path(__file__).resolve().parent / "raw_evidence.py"),
        "zero_step_control": {
            "available": True,
            "attack_updates": 0,
            "initialization": artifact.zero_step["initialization"],
            "comparisons": ["original_to_zero_step", "zero_step_to_final"],
            "selection_used": False,
        },
        "raw_depth_evidence": {
            "schema_version": RAW_SCHEMA_VERSION,
            "storage_policy": raw_policy,
            "root": os.path.relpath(Path(raw_root).resolve(), args.output.resolve()),
            "complete": all(record["raw_depth_evidence"]["available"] for record in per_sample if record["task"] == "depth"),
            "expected_states": 3 * sum(record["task"] == "depth" for record in per_sample),
            "missing_records": sum(not record["raw_depth_evidence"]["available"] for record in per_sample if record["task"] == "depth"),
            "tensor_codec": "zlib_lossless_contiguous_cpu_bytes",
            "deduplication": "content_address_includes_dtype_shape_and_exact_bytes",
            **(
                raw_store.statistics()
                if raw_store is not None
                else {"stored_states": 0, "created_objects": 0, "reused_objects": 0, "created_bytes": 0, "logical_tensor_bytes": 0}
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "summaries": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
