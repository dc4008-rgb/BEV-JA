"""Read-only depth conversion evidence; never changes evaluation values."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from artifacts import sha256
from evaluator import depth_metrics
from model_adapters import DepthAdapter, DepthPrediction


def roi_for(value: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("depth evidence requires [B,1,H,W] or [B,H,W]")
    return F.interpolate(object_mask.float(), value.shape[-2:], mode="nearest").bool().expand_as(value)


def tensor_statistics(value: torch.Tensor) -> dict[str, Any]:
    flattened = value.detach().reshape(-1).double()
    finite_mask = torch.isfinite(flattened)
    finite = flattened[finite_mask]
    return {
        "count": flattened.numel(), "finite_count": int(finite_mask.sum()),
        "nan_count": int(torch.isnan(flattened).sum()),
        "positive_inf_count": int(torch.isposinf(flattened).sum()),
        "negative_inf_count": int(torch.isneginf(flattened).sum()),
        "nonpositive_finite_count": int((finite <= 0).sum()),
        "finite_min": float(finite.min()) if finite.numel() else None,
        "finite_max": float(finite.max()) if finite.numel() else None,
        "finite_quantiles": dict(zip(
            ("p01", "p50", "p99"),
            torch.quantile(finite, finite.new_tensor([0.01, 0.5, 0.99])).tolist(),
        )) if finite.numel() else None,
        "finite_only_statistics_do_not_filter_saved_tensors_or_metrics": True,
    }


def summarize_prediction(prediction: DepthPrediction, object_mask: torch.Tensor) -> dict[str, Any]:
    result = {"conversion": prediction.metadata, "stages": {}}
    for name, stage in prediction.stages.items():
        shaped = stage.unsqueeze(1) if stage.ndim == 3 else stage
        roi = roi_for(shaped, object_mask)
        result["stages"][name] = {
            "shape": list(stage.shape), "dtype": str(stage.dtype),
            "full": tensor_statistics(stage), "vehicle_roi": tensor_statistics(shaped[roi]),
        }
        if stage.dtype == torch.bool:
            result["stages"][name]["true_pixels"] = int(stage.sum())
            result["stages"][name]["vehicle_roi_true_pixels"] = int(shaped[roi].sum())
    result["native_roi_mapping"] = "nearest_mask_resize_for_localization_only"
    return result


def compare_predictions(
    reference: DepthPrediction,
    evaluated: DepthPrediction,
    object_mask: torch.Tensor,
) -> dict[str, Any]:
    clean = reference.stages["evaluation_output"]
    final = evaluated.stages["evaluation_output"]
    if clean.shape != final.shape:
        raise ValueError("diagnostic comparison depth shapes differ")
    roi = roi_for(final, object_mask)
    if not roi.any():
        raise ValueError("diagnostic vehicle ROI is empty")
    result: dict[str, Any] = {}
    try:
        reference_semantics = str(reference.metadata["evaluation_semantics"])
        evaluated_semantics = str(evaluated.metadata["evaluation_semantics"])
        if reference_semantics != evaluated_semantics:
            raise ValueError("diagnostic comparison semantics differ")
        metrics = depth_metrics(
            final, clean, object_mask, semantics=reference_semantics
        )
        invalid = [
            name for name, value in metrics.items()
            if isinstance(value, float) and not math.isfinite(value)
        ]
        result["candidate_metrics"] = {name: value if not isinstance(value, float) or math.isfinite(value) else None for name, value in metrics.items()}
        result["candidate_metrics_status"] = "nonfinite_metric" if invalid else "finite"
        result["nonfinite_metric_fields"] = invalid
    except ValueError as error:
        result["candidate_metrics"] = None
        result["candidate_metrics_status"] = str(error)
        result["nonfinite_metric_fields"] = []
    invalid_pixels = ~(torch.isfinite(final) & torch.isfinite(clean))
    result["nonfinite_roi_pairs"] = int((invalid_pixels & roi).sum())
    if (invalid_pixels & roi).any():
        result["clamp_partition"] = None
        result["clamp_partition_status"] = "unavailable_no_nonfinite_pixel_filtering"
        return result
    legacy_clean = reference.stages["final_depth"]
    legacy_final = evaluated.stages["final_depth"]
    absolute_relative = (legacy_final.double().clamp_min(1e-6) - legacy_clean.double().clamp_min(1e-6)).abs() / legacy_clean.double().clamp_min(1e-6)
    total = float(absolute_relative[roi].sum())
    reference_clamp = reference.stages["lower_clamp_mask"] | reference.stages["upper_clamp_mask"]
    evaluated_clamp = evaluated.stages["lower_clamp_mask"] | evaluated.stages["upper_clamp_mask"]
    partitions = {}
    for name, mask in (
        ("neither_clamped", ~reference_clamp & ~evaluated_clamp),
        ("reference_only_clamped", reference_clamp & ~evaluated_clamp),
        ("evaluated_only_clamped", ~reference_clamp & evaluated_clamp),
        ("both_clamped", reference_clamp & evaluated_clamp),
    ):
        selected = mask & roi
        subtotal = float(absolute_relative[selected].sum())
        partitions[name] = {"roi_pixels": int(selected.sum()), "sum_abs_rel": subtotal,
                            "abs_rel_contribution_fraction": subtotal / total if total > 0 else None}
    result.update({"clamp_partition": partitions, "clamp_partition_status": "complete",
                   "partition_accumulation_dtype": "float64_diagnostic_only",
                   "partition_sum_abs_rel": total, "partition_roi_pixels": int(roi.sum()),
                   "roi_nonpositive_final_count": int(((final <= 0) & roi).sum()),
                   "roi_nonpositive_reference_count": int(((clean <= 0) & roi).sum())})
    return result


def capture_depth_case(
    adapter: DepthAdapter,
    images: Mapping[str, torch.Tensor],
    object_mask: torch.Tensor,
    supports: Mapping[str, torch.Tensor],
    output: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    names = set(images)
    if names not in ({"original"}, {"original", "zero_step", "final"}) or set(supports) != names:
        raise ValueError("capture requires original alone or all three image states with supports")
    if object_mask.ndim != 4 or object_mask.shape[:2] != (1, 1) or not torch.isfinite(object_mask).all() or not object_mask.bool().any():
        raise ValueError("capture requires a nonempty finite object mask [1,1,H,W]")
    json.dumps(provenance, allow_nan=False)
    output = output.resolve()
    if "formal_v3_delivery" in output.parts:
        raise ValueError("cannot write into historical delivery")
    output.mkdir(parents=True, exist_ok=False)
    cpu_mask = object_mask.detach().cpu().clone()
    observations = {}
    predictions = {}
    with torch.no_grad():
        for name in ("original", "zero_step", "final"):
            if name not in images:
                continue
            prediction = adapter.predict_with_diagnostics(images[name])
            predictions[name] = prediction
            payload = {
                "schema_version": 1, "model_id": adapter.model_id, "state": name,
                "input_rgb": images[name].detach().cpu().clone(), "object_mask": cpu_mask,
                "carrier_support": supports[name].detach().cpu().clone(),
                "stages": prediction.stages, "conversion": prediction.metadata,
                "provenance": provenance,
            }
            path = output / f"{name}.pt"
            torch.save(payload, path)
            observations[name] = {"tensor_file": path.name, "tensor_sha256": sha256(path),
                                  "tensor_bytes": path.stat().st_size, **summarize_prediction(prediction, cpu_mask)}
    comparisons = {}
    if "final" in predictions:
        for key, reference, evaluated in (
            ("original_to_zero_step", "original", "zero_step"),
            ("original_to_final", "original", "final"),
            ("zero_step_to_final", "zero_step", "final"),
        ):
            comparisons[key] = {"reference": reference, "evaluated": evaluated,
                                **compare_predictions(predictions[reference], predictions[evaluated], cpu_mask)}
    report = {
        "schema_version": 1, "status": "captured_not_effectiveness_validation",
        "model_id": adapter.model_id, "provenance": provenance,
        "numeric_policy_changed": True, "metric_clipping_added": False,
        "pixels_filtered": False,
        "numeric_policy": "bilinear_native_relative_inverse_plus_bounded_symmetric_metric",
        "observations": observations, "comparisons": comparisons,
        "diagnostic_writer_sha256": sha256(Path(__file__)),
        "metric_implementation_sha256": sha256(Path(__file__).with_name("evaluator.py")),
        "adapter_implementation_sha256": sha256(Path(__file__).parents[1] / "02_模型/model_adapters.py"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
