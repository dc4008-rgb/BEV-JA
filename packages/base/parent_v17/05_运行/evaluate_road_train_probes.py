#!/usr/bin/env python3
"""Fixed train-only observations of completed v17 100-update runs.

These four calibration scenes describe a training trajectory, not generalization
or convergence. All predeclared checkpoints are retained; final100 is always the
validation candidate. The CLI only loads the two core detectors, never optimizes,
and never opens a validation/test image. Candidate arrays use evaluator.py's
existing AP50/TNR semantics; matched car confidence is a separate statistic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in ("03_方法/production", "02_模型", "01_公共数据集", "04_统一评估"):
    sys.path.insert(0, str(ROOT / directory))

from advrm_insertion import load_car
from data import calibration_records, sha256
from evaluator import (CAR_CLASS_IDS, VEHICLE_CLASS_IDS, detection_candidates,
                       detection_summary)
from ours_road_v16 import require_candidate_run_path
from road_patch_contract import RoadRGBView, load_patch
from road_runtime import (CORE, TRAIN_MANIFEST_SHA256, MODEL_MANIFEST_SHA256,
                          build_context, build_offline_detector, preflight,
                          training_records)

CANDIDATE_ID = "normalized-v17-roadpatch-20260920"
CHECKPOINT_STEPS = (25, 50, 75, 100)
STATES = (("zero", "zero_step_patch.pt"),) + tuple(
    ("checkpoint_%04d" % step, "checkpoint_%04d.pt" % step)
    for step in CHECKPOINT_STEPS
)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def fixed_train_probes(records, bound_sample_ids):
    """Selection uses frozen manifest metadata only, never observed model scores."""
    if any(record.split not in ("train", "validation") for record in records):
        raise ValueError("test/unknown split is forbidden in probe manifest metadata")
    selected = calibration_records(records)
    ids = [record.sample_id for record in selected]
    if ids != list(bound_sample_ids or ()):
        raise ValueError("calibration_sample_ids changed from fixed train calibration")
    require_train_probes(selected, ids)
    return selected


def require_train_probes(records, expected_ids):
    if len(records) != 4 or any(record.split != "train" for record in records):
        raise ValueError("probe images must be exactly four train rows; validation/test forbidden")
    if Counter(record.dataset for record in records) != {"kitti": 2, "nuscenes": 2}:
        raise ValueError("fixed train probes require two rows from each domain")
    ids = [record.sample_id for record in records]
    if len(set(ids)) != 4 or ids != list(expected_ids):
        raise ValueError("probe calibration order/identity drift")


def load_bound_run(run_output, training_manifest, model_manifest):
    """Verify frozen identities and every checkpoint before tensor deserialization."""
    run_output = Path(run_output)
    summary_path = run_output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = summary.get("candidate_id")
    if identity == CANDIDATE_ID:
        if (summary.get("status") != "development_optimization_complete"
                or summary.get("method") not in ("contextual", "robust_dpatch")
                or summary.get("mode") != "validation"
                or summary.get("updates") != 100
                or tuple(summary.get("train_models", ())) != CORE):
            raise ValueError("baseline must be a completed v17 100-update run")
        patch_bindings = summary.get("patch_sha256", {})
    elif identity in (CANDIDATE_ID + "-ours-det-only", CANDIDATE_ID + "-ours-dual"):
        variant = summary.get("task_variant")
        expected_models = CORE if variant == "det-only" else (*CORE, "monodepth2", "dpt_large")
        if (variant not in ("det-only", "dual") or identity != CANDIDATE_ID + "-ours-" + variant
                or summary.get("status") != "train_optimization_complete"
                or summary.get("executed_updates") != 100
                or tuple(summary.get("training_models", ())) != expected_models):
            raise ValueError("ours must be a completed v17 100-update run")
        patch_bindings = summary.get("patch_bindings", {})
    else:
        raise ValueError("foreign/historical candidate identity")
    if summary.get("test_data_opened") is not False:
        raise ValueError("completed run must prove test-data isolation")
    for path, field, frozen in (
        (training_manifest, "training_manifest_sha256", TRAIN_MANIFEST_SHA256),
        (model_manifest, "model_manifest_sha256", MODEL_MANIFEST_SHA256),
    ):
        actual = sha256(Path(path))
        if actual != frozen or summary.get(field) != actual:
            raise ValueError("run input binding failed: " + field)
    checkpoint_bindings = summary.get("checkpoint_bindings", {})
    expected_names = {"checkpoint_%04d.pt" % step for step in CHECKPOINT_STEPS}
    if set(checkpoint_bindings) != expected_names:
        raise ValueError("checkpoint_bindings must contain exactly 25/50/75/100")
    bindings = {name: patch_bindings.get(name)
                for name in ("zero_step_patch.pt", "final_patch.pt")}
    bindings.update(checkpoint_bindings)
    for name, expected in bindings.items():
        path = run_output / name
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError("patch/checkpoint hash binding failed: " + name)
    patches = {state: load_patch(run_output / filename) for state, filename in STATES}
    if not torch.equal(patches["checkpoint_0100"], load_patch(run_output / "final_patch.pt")):
        raise ValueError("final_patch must equal the fixed checkpoint_0100 canonical RGB")
    for step in CHECKPOINT_STEPS:
        payload = torch.load(run_output / ("checkpoint_%04d.pt" % step),
                             map_location="cpu", weights_only=True)
        if payload.get("metadata", {}).get("updates") != step:
            raise ValueError("checkpoint update metadata differs from its declared step")
    return summary, patches, {"summary_sha256": sha256(summary_path),
                              "patch_bindings": bindings}


def raw_predictions(output):
    """Retain all post-adapter detections, including non-car and unmatched boxes."""
    required = ("boxes", "labels", "scores")
    if any(key not in output for key in required):
        raise ValueError("detector prediction lacks boxes/labels/scores")
    boxes, labels, scores = (output[key].detach().cpu() for key in required)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != scores.shape or len(boxes) != len(scores):
        raise ValueError("detector prediction shapes differ")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("nonfinite detector output; retain failed probe")
    return {"boxes": boxes.tolist(), "labels": labels.tolist(), "scores": scores.tolist()}


def matched_car_confidence(car_candidates):
    return max((float(item["score"]) for item in car_candidates
                if int(item["label"]) in CAR_CLASS_IDS and float(item["iou"]) >= .5), default=0.0)


def evaluate_probes(records, patches, expected_ids, device, adapter_factory,
                    context_factory, emit_row=None):
    """No-gradient evaluation core; injectable factories serve synthetic tests only."""
    require_train_probes(records, expected_ids)
    if set(patches) != {state for state, _ in STATES}:
        raise ValueError("all five predeclared patch states must be evaluated")
    rows = []
    queries = {"clean_model_forwards": 0, "patch_model_forwards": 0,
               "optimization_updates": 0, "backward_calls": 0}
    # No optimization, parameters, loss calls, best-state selection, or EOT here.
    with torch.inference_mode():
        views = {state: RoadRGBView(patches[state].to(device)) for state, _ in STATES}
        for model_id in CORE:
            adapter = adapter_factory(model_id)
            for record in records:
                context = context_factory(record)
                clean = adapter.predict(context.benign)[0]
                queries["clean_model_forwards"] += 1
                clean_raw = raw_predictions(clean)
                clean_vehicle = detection_candidates(clean, context.object_box, VEHICLE_CLASS_IDS)
                clean_car = detection_candidates(clean, context.object_box, CAR_CLASS_IDS)
                for state, filename in STATES:
                    rendered = views[state].render(context)
                    adversarial = adapter.predict(rendered.image)[0]
                    queries["patch_model_forwards"] += 1
                    adversarial_raw = raw_predictions(adversarial)
                    adversarial_car = detection_candidates(adversarial, context.object_box, CAR_CLASS_IDS)
                    row = {
                        "sample_id": record.sample_id, "dataset": record.dataset,
                        "split": "train", "model": model_id, "state": state,
                        "checkpoint_file": filename, "task": "detection",
                        "target_box": list(context.object_box),
                        "source_sha256": record.source_sha256,
                        "clean_candidates": clean_vehicle,
                        "adversarial_candidates": detection_candidates(adversarial, context.object_box, VEHICLE_CLASS_IDS),
                        "clean_car_candidates": clean_car,
                        "adversarial_car_candidates": adversarial_car,
                        "clean_raw_detections": clean_raw,
                        "adversarial_raw_detections": adversarial_raw,
                        "matched_car_confidence_clean": matched_car_confidence(clean_car),
                        "matched_car_confidence_patch": matched_car_confidence(adversarial_car),
                    }
                    rows.append(row)
                    if emit_row is not None:
                        emit_row(row)
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()
    queries["total_model_forwards"] = queries["clean_model_forwards"] + queries["patch_model_forwards"]
    if len(rows) != 40 or queries["total_model_forwards"] != 48:
        raise RuntimeError("fixed probe query budget differs from 8 clean + 40 patch")
    return rows, queries


def aggregate_probes(rows):
    summaries = []
    for model in CORE:
        for state, _ in STATES:
            for domain in ("kitti", "nuscenes", "combined"):
                selected = [r for r in rows if r["model"] == model and r["state"] == state
                            and (domain == "combined" or r["dataset"] == domain)]
                metrics = detection_summary(selected, 1000, 20260919, allow_zero_reference=True)
                summaries.append({"model": model, "state": state, "domain": domain, **metrics,
                                  "mean_matched_car_confidence_clean": sum(r["matched_car_confidence_clean"] for r in selected) / len(selected),
                                  "mean_matched_car_confidence_patch": sum(r["matched_car_confidence_patch"] for r in selected) / len(selected)})
    return summaries


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "02_模型/model_manifest.json")
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--car", type=Path, default=ROOT / "01_公共数据集/car_00.jpg")
    parser.add_argument("--car-mask", type=Path, default=ROOT / "01_公共数据集/car_00_mask.jpg")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output = require_candidate_run_path(args.output, ROOT)
    args.run_output = require_candidate_run_path(args.run_output, ROOT)
    if args.output == args.run_output or args.run_output in args.output.parents or args.output in args.run_output.parents:
        raise ValueError("probe output must be separate from the completed training run")
    return args


def main(argv=None):
    args = arguments(argv)
    # Existing even-empty output is evidence. Reserve exactly one new run name.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "invocation.json", {key: str(value) if isinstance(value, Path) else value
                                                    for key, value in vars(args).items()})
        summary, patches, bindings = load_bound_run(args.run_output, args.training_manifest, args.model_manifest)
        records_all = training_records(args.training_manifest, args.source_prefix, args.source_root)
        records = fixed_train_probes(records_all, summary.get("calibration_sample_ids"))
        checks = preflight(records, CORE, args.model_manifest, args.car, args.car_mask, args.weights_root)
        write_json(args.output / "input_preflight.json", checks)
        if not checks["inputs_ready"]:
            raise RuntimeError("fixed train source/car/model offline hash preflight failed")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real pretrained train probes require CUDA")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        car, car_mask = load_car(args.car, args.car_mask, device)
        with (args.output / "model_sample_records.jsonl").open("x", encoding="utf-8") as stream:
            def emit_row(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print("TRAIN_PROBE", row["model"], row["state"], row["sample_id"], flush=True)

            rows, queries = evaluate_probes(
                records, patches, summary["calibration_sample_ids"], device,
                lambda model: build_offline_detector(model, device, args.model_manifest, args.weights_root),
                lambda record: build_context(record, car, car_mask, device), emit_row,
            )
        torch.cuda.synchronize(device)
        report = {
            "schema_version": 1, "status": "fixed_train_probes_complete",
            "candidate_id": summary["candidate_id"], "method": summary.get("method"),
            "task_variant": summary.get("task_variant"), "executed_training_updates": 100,
            "diagnostic_scope": "four fixed training/calibration scenes; not validation, test, or evidence of generalization/convergence",
            "checkpoint_selection": "none; all 0/25/50/75/100 retained, final100 always goes to validation",
            "test_data_opened": False, "validation_images_opened": False,
            "transfer_models_used": False, "optimization_performed": False,
            "models": list(CORE), "calibration_sample_ids": summary["calibration_sample_ids"],
            "scene_count": 4, "record_count": len(rows), "query_accounting": queries,
            "matched_car_confidence_definition": "maximum car class 3 score with target box IoU >= 0.5; no match = 0; no extra score threshold",
            "ap_tnr_definition": "evaluator.py local intersecting vehicle (3,6,8) candidates; AP@0.5 across scenes; (clean AP - patch AP) / clean AP, null if clean AP = 0",
            "input_bindings": bindings, "training_manifest_sha256": sha256(args.training_manifest),
            "model_manifest_sha256": sha256(args.model_manifest),
            "records_sha256": sha256(args.output / "model_sample_records.jsonl"),
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "summaries": aggregate_probes(rows),
        }
        write_json(args.output / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        write_json(args.output / "failure.json", {
            "status": "failed_retained", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "optimization_performed": False,
            "validation_images_opened": False, "test_data_opened": False,
        })
        raise


if __name__ == "__main__":
    main()
