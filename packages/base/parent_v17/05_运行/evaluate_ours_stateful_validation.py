#!/usr/bin/env python3
"""Evaluate a completed ours candidate on the frozen 20+20 validation rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "03_方法" / "production",
    ROOT / "04_统一评估",
    ROOT / "02_模型",
    ROOT / "01_公共数据集",
):
    sys.path.insert(0, str(path))

from advrm_insertion import load_car  # noqa: E402
from road_patch_contract import RoadRGBView, load_patch
from road_runtime import training_records, preflight as road_preflight, build_offline_detector
from ours_road_v16 import CANDIDATE_PREFIX, model_order, depth_asset_checks, require_candidate_run_path, verify_patch_bindings  # noqa: E402
from evaluator import (  # noqa: E402
    CAR_CLASS_IDS,
    VEHICLE_CLASS_IDS,
    build_context,
    depth_metrics,
    depth_summary,
    detection_candidates,
    detection_summary,
    zero_step_comparisons,
    zero_step_summary,
)
from model_adapters import (  # noqa: E402
    DETECTION_MODEL_IDS,
    build_depth_adapter,
)
from ours_stateful import reject_test_input, texture_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "validation"), default="preflight")
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=ROOT / "01_公共数据集" / "runtime_v1" / "train_validation_manifest.csv",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=ROOT / "02_模型" / "model_manifest.json",
    )
    parser.add_argument(
        "--car", type=Path, default=ROOT / "01_公共数据集" / "car_00.jpg"
    )
    parser.add_argument(
        "--car-mask",
        type=Path,
        default=ROOT / "01_公共数据集" / "car_00_mask.jpg",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args()
    args.output = require_candidate_run_path(args.output, ROOT)
    args.run_output = require_candidate_run_path(args.run_output, ROOT)
    return args


def validation_records(records) -> tuple[Any, ...]:
    selected = tuple(record for record in records if record.split == "validation")
    if Counter(record.dataset for record in selected) != {"kitti": 20, "nuscenes": 20}:
        raise ValueError("frozen validation selection must contain 20+20 rows")
    return selected


def depth_prediction(adapter: Any, image: torch.Tensor) -> tuple[torch.Tensor, str]:
    diagnostic = adapter.predict_with_diagnostics(image)
    if "evaluation_output" not in diagnostic.stages:
        raise ValueError("depth adapter omits evaluation_output")
    semantics = str(diagnostic.metadata.get("evaluation_semantics", ""))
    if not semantics:
        raise ValueError("depth adapter omits evaluation semantics")
    return diagnostic.stages["evaluation_output"], semantics


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    summary_path = args.run_output / "summary.json"
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    allowed_statuses = {"train_optimization_complete"} if args.mode == "validation" else {"train_optimization_complete", "smoke_complete"}
    if run_summary.get("status") not in allowed_statuses:
        raise ValueError("evaluation mode is incompatible with the completed optimization run")
    variant = run_summary.get("task_variant")
    selected_models = model_order(variant)
    if run_summary.get("candidate_id") != CANDIDATE_PREFIX + variant:
        raise ValueError("evaluation refuses an inherited or foreign run identity")
    if tuple(run_summary.get("training_models", [])) != selected_models:
        raise ValueError("run training model order does not match task variant")
    if run_summary.get("test_data_opened") is not False:
        raise ValueError("run summary does not prove test-data isolation")

    verify_patch_bindings(args.run_output, run_summary)
    from data import sha256
    if sha256(args.training_manifest) != run_summary.get("training_manifest_sha256"):
        raise ValueError("evaluation manifest differs from training run")
    if sha256(args.model_manifest) != run_summary.get("model_manifest_sha256"):
        raise ValueError("evaluation victim manifest differs from training run")
    records_all = training_records(args.training_manifest, args.source_prefix, args.source_root)
    reject_test_input(args.training_manifest, records_all)
    records = validation_records(records_all)
    if args.mode == "smoke":
        records = tuple(next(r for r in records if r.dataset == dataset) for dataset in ("kitti", "nuscenes"))
    preflight = road_preflight(records, selected_models[:2], args.model_manifest,
                               args.car, args.car_mask, args.weights_root)
    preflight["depth_assets"] = depth_asset_checks(selected_models, args.model_manifest)
    preflight.update(candidate_id=run_summary["candidate_id"], task_variant=variant,
                     evaluation_models=list(selected_models), transfer_models_used=False,
                     test_data_opened=False, formal_test_allowed=False)
    if args.mode == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if not preflight["inputs_ready"] or not preflight["depth_assets"]["inputs_ready"]:
        raise RuntimeError("validation inputs failed source/car/weight verification")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("candidate validation requires CUDA")
    zero_patch = load_patch(args.run_output / "zero_step_patch.pt").to(device)
    final_patch = load_patch(args.run_output / "final_patch.pt").to(device)
    zero_carrier = RoadRGBView(zero_patch).to(device)
    final_carrier = RoadRGBView(final_patch).to(device)
    car, car_mask = load_car(args.car, args.car_mask, device)
    evaluation_models = selected_models
    verified: set[str] = set()
    summaries = []
    per_sample = []
    started = time.monotonic()

    for model_index, model_id in enumerate(evaluation_models):
        task = "detection" if model_id in DETECTION_MODEL_IDS else "depth"
        adapter = (
            build_offline_detector(model_id, device, args.model_manifest, args.weights_root)
            if task == "detection"
            else build_depth_adapter(model_id, device, model_manifest=args.model_manifest)
        )
        model_rows = []
        model_started = time.monotonic()
        with torch.no_grad():
            for index, record in enumerate(records, start=1):
                context = build_context(record, car, car_mask, device, verified)
                generator_device = device.type if device.type == "cuda" else "cpu"
                zero_render = zero_carrier.render(
                    context,
                    eot=None,
                    generator=torch.Generator(device=generator_device).manual_seed(0),
                )
                final_render = final_carrier.render(
                    context,
                    eot=None,
                    generator=torch.Generator(device=generator_device).manual_seed(0),
                )
                if task == "detection":
                    clean = adapter.predict(context.benign)[0]
                    zero = adapter.predict(zero_render.image)[0]
                    adversarial = adapter.predict(final_render.image)[0]
                    row = {
                        "sample_id": record.sample_id,
                        "dataset": record.dataset,
                        "split": record.split,
                        "model": model_id,
                        "task": task,
                        "clean_candidates": detection_candidates(
                            clean, context.object_box, VEHICLE_CLASS_IDS
                        ),
                        "adversarial_candidates": detection_candidates(
                            adversarial, context.object_box, VEHICLE_CLASS_IDS
                        ),
                        "clean_car_candidates": detection_candidates(
                            clean, context.object_box, CAR_CLASS_IDS
                        ),
                        "adversarial_car_candidates": detection_candidates(
                            adversarial, context.object_box, CAR_CLASS_IDS
                        ),
                        "zero_step_control": zero_step_comparisons(
                            task, context, clean, zero, adversarial
                        ),
                    }
                else:
                    clean, clean_semantics = depth_prediction(adapter, context.benign)
                    zero, zero_semantics = depth_prediction(adapter, zero_render.image)
                    adversarial, adversarial_semantics = depth_prediction(
                        adapter, final_render.image
                    )
                    if len({clean_semantics, zero_semantics, adversarial_semantics}) != 1:
                        raise ValueError("depth semantics changed between clean/final")
                    row = {
                        "sample_id": record.sample_id,
                        "dataset": record.dataset,
                        "split": record.split,
                        "model": model_id,
                        "task": task,
                        **depth_metrics(
                            adversarial,
                            clean,
                            context.object_mask,
                            semantics=clean_semantics,
                        ),
                        "zero_step_control": zero_step_comparisons(
                            task,
                            context,
                            clean,
                            zero,
                            adversarial,
                            clean_semantics,
                        ),
                    }
                model_rows.append(row)
                per_sample.append(row)
                print(f"VALIDATE {model_id} {index}/{len(records)} {record.sample_id}", flush=True)

        domains = {}
        for domain_index, domain in enumerate(("kitti", "nuscenes", "combined")):
            selected = model_rows if domain == "combined" else [
                row for row in model_rows if row["dataset"] == domain
            ]
            domains[domain] = (
                detection_summary(
                    selected,
                    args.bootstrap_samples,
                    args.bootstrap_seed + 10 * model_index + domain_index,
                    allow_zero_reference=True,
                )
                if task == "detection"
                else depth_summary(
                    selected,
                    args.bootstrap_samples,
                    args.bootstrap_seed + 10 * model_index + domain_index,
                )
            )
            domains[domain]["zero_step_control"] = zero_step_summary(
                selected,
                task,
                args.bootstrap_samples,
                args.bootstrap_seed + 10 * model_index + domain_index,
            )
        summaries.append(
            {
                "model": model_id,
                "task": task,
                "elapsed_seconds": time.monotonic() - model_started,
                "domains": domains,
            }
        )
        del adapter
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "model_sample_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_sample:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "status": "complete",
        "identity": "v16 ours core-model validation only; transfer held out; not formal test",
        "test_data_opened": False,
        "scene_count": len(records),
        "candidate_id": run_summary["candidate_id"],
        "task_variant": variant,
        "evaluation_mode": args.mode,
        "transfer_models_used": False,
        "canonical_patch_hw": [64, 128],
        "model_count": len(evaluation_models),
        "model_sample_record_count": len(per_sample),
        "models": list(evaluation_models),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "zero_step_texture": texture_diagnostics(zero_patch),
        "final_texture": texture_diagnostics(final_patch),
        "elapsed_seconds": time.monotonic() - started,
        "summaries": summaries,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
