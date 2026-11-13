#!/usr/bin/env python3
"""One train/validation-only detector evaluation for all four v17 final100 runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ("03_方法/production", "02_模型", "01_公共数据集", "04_统一评估"):
    sys.path.insert(0, str(ROOT / directory))

import torch
from PIL import Image, ImageDraw
from advrm_insertion import load_car
from carriers import extract_road_patch
from data import calibration_records, seeded_round_robin, sha256
from road_patch_contract import (CANONICAL_HW, RoadRGBView, canonicalize_rgb,
                                 load_patch, save_patch, _tensor_digest)
from road_runtime import (CORE, build_context, build_offline_detector, preflight,
                          training_records)
from evaluator import (CAR_CLASS_IDS, VEHICLE_CLASS_IDS, detection_candidates,
                       detection_summary)

SEED = 20260919
CANDIDATE_ID = "normalized-v17-roadpatch-20260920"
CHECKPOINTS = {"checkpoint_%04d.pt" % step: step for step in (25, 50, 75, 100)}
STATES = ("zero", "random", "final")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def candidate_run_path(path, root=ROOT):
    path, allowed = Path(path).resolve(), (Path(root) / "runs").resolve()
    try:
        relative = path.relative_to(allowed)
    except ValueError as error:
        raise ValueError("run/output must be inside this candidate runs directory") from error
    if not relative.parts:
        raise ValueError("a named run directory is required")
    return path


def common_random_patch():
    # CPU generator matches the baseline's predeclared random control exactly.
    return torch.rand((1, 3, *CANONICAL_HW), dtype=torch.float32,
                      generator=torch.Generator().manual_seed(SEED))


def validation_rows(records):
    if any(record.split not in ("train", "validation") for record in records):
        raise ValueError("only train/validation records are permitted")
    chosen = sorted((r for r in records if r.split == "validation"),
                    key=lambda r: (r.dataset, r.sample_id))
    if Counter(r.dataset for r in chosen) != {"kitti": 20, "nuscenes": 20}:
        raise ValueError("exactly 20+20 validation scenes are required")
    if len({r.sample_id for r in chosen}) != 40:
        raise ValueError("duplicate validation sample ID")
    if {r.sample_id for r in chosen} & {r.sample_id for r in records if r.split == "train"}:
        raise ValueError("train/validation sample IDs overlap")
    return chosen


def checked_bindings(run, bindings, required, exact=False):
    if not isinstance(bindings, dict) or not set(required).issubset(bindings):
        raise ValueError("missing strict patch/checkpoint bindings")
    if exact and set(bindings) != set(required):
        raise ValueError("checkpoint bindings must contain exactly 25/50/75/100")
    for name, digest in bindings.items():
        if Path(name).name != name or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid patch binding")
        path = run / name
        if path.resolve().parent != run or not path.is_file() or sha256(path) != digest:
            raise ValueError("patch/checkpoint hash mismatch: " + name)


def validate_completed_run(run, records, training_manifest, model_manifest, root=ROOT):
    """Fail before inference if completion, identity, split or artifact evidence differs."""
    run = candidate_run_path(run, root)
    validation_rows(records)
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("test_data_opened") is not False:
        raise ValueError("completed run does not prove test isolation")
    method = summary.get("method")
    if method in ("contextual", "robust_dpatch"):
        if (summary.get("candidate_id") != CANDIDATE_ID or
                summary.get("status") != "development_optimization_complete" or
                summary.get("mode") != "validation" or summary.get("updates") != 100 or
                tuple(summary.get("train_models", [])) != CORE):
            raise ValueError("baseline must be a completed v17 validation100 run")
        bindings = summary.get("patch_sha256")
        checked_bindings(run, bindings, ("zero_step_patch.pt", "random_control_patch.pt", "final_patch.pt"))
        if summary.get("final_patch_sha256") != bindings["final_patch.pt"]:
            raise ValueError("baseline final patch bindings disagree")
        trace_name, step_field = "trace.jsonl", "step"
        runtime_path = run / "runtime.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
    else:
        variant = summary.get("task_variant")
        if variant not in ("det-only", "dual"):
            raise ValueError("unsupported comparison method")
        expected_models = CORE if variant == "det-only" else (*CORE, "monodepth2", "dpt_large")
        if (summary.get("candidate_id") != CANDIDATE_ID + "-ours-" + variant or
                summary.get("status") != "train_optimization_complete" or
                summary.get("executed_updates") != 100 or
                tuple(summary.get("training_models", [])) != expected_models):
            raise ValueError("ours must be a completed v17 validation100 run")
        method = "ours_" + variant
        bindings = summary.get("patch_bindings")
        checked_bindings(run, bindings, ("zero_step_patch.pt", "final_patch.pt"))
        trace_name, step_field = "attack_trace.jsonl", "global_step"
        runtime = {}
    for field, path in (("training_manifest_sha256", training_manifest),
                        ("model_manifest_sha256", model_manifest)):
        if summary.get(field, runtime.get(field)) != sha256(path):
            raise ValueError("completed run manifest differs: " + field)
    checkpoints = summary.get("checkpoint_bindings")
    checked_bindings(run, checkpoints, CHECKPOINTS, exact=True)
    for name, step in CHECKPOINTS.items():
        payload = torch.load(run / name, map_location="cpu", weights_only=True)
        if payload.get("metadata", {}).get("updates") != step:
            raise ValueError("checkpoint step metadata differs")
        load_patch(run / name)
    final, zero = load_patch(run / "final_patch.pt"), load_patch(run / "zero_step_patch.pt")
    if not torch.equal(final, load_patch(run / "checkpoint_0100.pt")):
        raise ValueError("final canonical patch differs from fixed checkpoint100")
    random = common_random_patch()
    if "random_control_patch.pt" in bindings and not torch.equal(random, load_patch(run / "random_control_patch.pt")):
        raise ValueError("run random control differs from common seed20260919 control")
    trace = [json.loads(line) for line in (run / trace_name).read_text().splitlines() if line.strip()]
    expected = seeded_round_robin(records, 100, SEED)
    if len(trace) != 100:
        raise ValueError("actual update trace must contain exactly100 rows")
    for index, (row, record) in enumerate(zip(trace, expected), 1):
        if (row.get(step_field) != index or row.get("sample_id") != record.sample_id or
                row.get("dataset") != record.dataset or row.get("split", "train") != "train" or
                record.split != "train"):
            raise ValueError("trace differs from the frozen100 train-only schedule")
    return {"method": method, "summary": summary, "run": run,
            "patches": {"zero": zero, "random": random, "final": final},
            "trace_sha256": sha256(run / trace_name),
            "summary_sha256": sha256(run / "summary.json"),
            "patch_bindings": bindings, "checkpoint_bindings": checkpoints}


def invariant_record(context, render, record, model, state):
    support = render.support > 0
    protected = context.object_mask.ne(0).clone()
    left, top, right, bottom = map(int, context.object_box)
    protected[..., top:bottom, left:right] = True
    image, clean = render.image, context.benign
    checks = {
        "support_nonempty": bool(support.any()),
        "protected_support_disjoint": not bool((support & protected).any()),
        "protected_vehicle_pixels_exact": torch.equal(image[protected.expand_as(image)], clean[protected.expand_as(clean)]),
        "outside_support_pixels_exact": torch.equal(image[(~support).expand_as(image)], clean[(~support).expand_as(clean)]),
        "finite_input": bool(torch.isfinite(image).all()),
    }
    if not all(checks.values()):
        raise RuntimeError("geometry/vehicle invariant failed: " + record.sample_id)
    return {"sample_id": record.sample_id, "dataset": record.dataset, "split": record.split,
            "model": model, "state": state, "target_box": list(context.object_box),
            "road_quad": torch.as_tensor(context.road_quad).cpu().tolist(),
            "support_pixels": int(support.sum()), "protected_pixels": int(protected.sum()),
            "support_sha256": _tensor_digest(support), "checks": checks}


def save_preview(path, image, predictions, box, label):
    array = (image[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    canvas = Image.fromarray(array)
    draw = ImageDraw.Draw(canvas)
    for predicted, category, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if int(category) in VEHICLE_CLASS_IDS and float(score) >= .3:
            xyxy = [float(x) for x in predicted]
            draw.rectangle(xyxy, outline="orange", width=1)
            draw.text((xyxy[0], xyxy[1]), "%s:%.3f" % (int(category), float(score)), fill="orange")
    draw.rectangle(box, outline="cyan", width=2)
    draw.rectangle((0, 0, canvas.width, 16), fill="black")
    draw.text((3, 2), label, fill="white")
    canvas.save(path)


def cpu_predictions(predictions):
    return {key: value.detach().cpu() for key, value in predictions.items()}


def evaluate_comparison(args, records, checked, device):
    selected = validation_rows(records)
    car, mask = load_car(args.car, args.car_mask, device)
    calibration = calibration_records(records)
    initial_context = build_context(calibration[0], car, mask, device)
    expected_zero = canonicalize_rgb(extract_road_patch(initial_context.background, initial_context.road_quad))
    if not torch.equal(checked["patches"]["zero"].to(device), expected_zero):
        raise ValueError("zero patch differs from common frozen initialization")
    # All methods receive this separately bound control, never mutate the training run.
    random_path = args.output / "random_control_patch.pt"
    save_patch(random_path, checked["patches"]["random"], {"state": "unoptimized_random_control", "seed": SEED})
    patches = {key: value.to(device) for key, value in checked["patches"].items()}
    rows, summaries, geometry = [], [], []
    prediction_counts = {model: 0 for model in CORE}
    first = {d: next(r.sample_id for r in selected if r.dataset == d) for d in ("kitti", "nuscenes")}
    for model_index, model in enumerate(CORE):
        adapter = build_offline_detector(model, device, args.model_manifest, args.weights_root)
        by_state = {state: [] for state in STATES}
        with torch.no_grad():
            for record in selected:
                context = build_context(record, car, mask, device)
                clean = adapter.predict(context.benign)[0]
                prediction_counts[model] += 1
                visual = record.sample_id == first[record.dataset]
                prefix = model + "_" + record.dataset
                raw_predictions = {"sample_id": record.sample_id, "split": "validation", "model": model,
                                   "target_box": context.object_box, "display_threshold_only": .3,
                                   "clean": cpu_predictions(clean)} if visual else None
                raw_inputs = {"sample_id": record.sample_id, "split": "validation", "target_box": context.object_box,
                              "clean": context.benign.detach().cpu(), "object_mask": context.object_mask.detach().cpu(),
                              "road_quad": torch.as_tensor(context.road_quad).cpu()} if visual and model_index == 0 else None
                if visual:
                    save_preview(args.output / (prefix + "_clean.png"), context.benign, clean, context.object_box,
                                 model + " | " + record.sample_id + " | clean")
                for state in STATES:
                    render = RoadRGBView(patches[state]).render(context)
                    geometry.append(invariant_record(context, render, record, model, state))
                    output = adapter.predict(render.image)[0]
                    prediction_counts[model] += 1
                    row = {"method": checked["method"], "model": model, "sample_id": record.sample_id,
                           "dataset": record.dataset, "split": "validation", "state": state,
                           "clean_candidates": detection_candidates(clean, context.object_box, VEHICLE_CLASS_IDS),
                           "adversarial_candidates": detection_candidates(output, context.object_box, VEHICLE_CLASS_IDS),
                           "clean_car_candidates": detection_candidates(clean, context.object_box, CAR_CLASS_IDS),
                           "adversarial_car_candidates": detection_candidates(output, context.object_box, CAR_CLASS_IDS)}
                    by_state[state].append(row)
                    rows.append(row)
                    if visual:
                        raw_predictions[state] = cpu_predictions(output)
                        save_preview(args.output / (prefix + "_" + state + ".png"), render.image, output,
                                     context.object_box, model + " | " + record.sample_id + " | " + state)
                    if raw_inputs is not None:
                        raw_inputs[state] = render.image.detach().cpu()
                        raw_inputs[state + "_support"] = render.support.detach().cpu()
                if visual:
                    torch.save(raw_predictions, args.output / (prefix + "_predictions.pt"))
                if raw_inputs is not None:
                    torch.save(raw_inputs, args.output / (record.dataset + "_inputs.pt"))
            for state in STATES:
                for domain in ("kitti", "nuscenes", "combined"):
                    subset = [r for r in by_state[state] if domain == "combined" or r["dataset"] == domain]
                    summaries.append(dict(detection_summary(subset, 1000, SEED, allow_zero_reference=True),
                                          model=model, state=state, dataset=domain))
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(rows) != 240 or len(geometry) != 240:
        raise RuntimeError("incomplete40 scene x2model x3state coverage")
    write_json(args.output / "per_sample.json", rows)
    write_json(args.output / "geometry_audit.json", {"status": "pass", "unique_scenes": 40,
               "checked_model_scene_states": len(geometry), "checks": geometry})
    return {"status": "development_validation_complete", "candidate_id": CANDIDATE_ID,
            "method": checked["method"], "updates": 100, "records": len(rows), "summaries": summaries,
            "evaluation_models": list(CORE), "transfer_models_used": False,
            "evaluation_model_scene_forwards": prediction_counts,
            "evaluation_forward_total": sum(prediction_counts.values()),
            "test_data_opened": False, "formal_test_allowed": False,
            "metric": "tnr_det=(target_vehicle_AP50_clean-target_vehicle_AP50_adv)/target_vehicle_AP50_clean",
            "control_tensor_sha256": {k: _tensor_digest(v) for k, v in checked["patches"].items()},
            "random_control_patch_sha256": sha256(random_path),
            "run_summary_sha256": checked["summary_sha256"], "run_trace_sha256": checked["trace_sha256"],
            "run_patch_bindings": checked["patch_bindings"], "run_checkpoint_bindings": checked["checkpoint_bindings"],
            "geometry_audit_sha256": sha256(args.output / "geometry_audit.json"),
            "note": "Fixed final100 only; no outcome filtering, best checkpoint selection or training."}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "02_模型/model_manifest.json")
    parser.add_argument("--car", type=Path, default=ROOT / "01_公共数据集/car_00.jpg")
    parser.add_argument("--car-mask", type=Path, default=ROOT / "01_公共数据集/car_00_mask.jpg")
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output = candidate_run_path(args.output)
    args.run_output = candidate_run_path(args.run_output)
    if args.output == args.run_output or args.run_output in args.output.parents:
        raise ValueError("evaluation output must not mutate a completed run")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite evaluation output")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        records = training_records(args.training_manifest, args.source_prefix, args.source_root)
        checked = validate_completed_run(args.run_output, records, args.training_manifest, args.model_manifest)
        checks = preflight(records, CORE, args.model_manifest, args.car, args.car_mask, args.weights_root)
        write_json(args.output / "input_preflight.json", checks)
        write_json(args.output / "invocation.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        import torchvision
        runtime = {"candidate_id": CANDIDATE_ID, "python": platform.python_version(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "cuda_version": torch.version.cuda,
                   "model_manifest_sha256": sha256(args.model_manifest), "training_manifest_sha256": sha256(args.training_manifest),
                   "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.rglob("*.py")
                                     if "runs" not in p.relative_to(ROOT).parts}}
        write_json(args.output / "runtime.json", runtime)
        if not checks["inputs_ready"]:
            raise RuntimeError("missing/altered offline model, vehicle or240 train/validation source assets")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real comparison evaluation requires CUDA; no synthetic fallback")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        result = evaluate_comparison(args, records, checked, device)
        torch.cuda.synchronize(device)
        result.update(seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                      runtime_sha256=sha256(args.output / "runtime.json"))
        write_json(args.output / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed_retained", "type": type(error).__name__,
                   "error": str(error), "test_data_opened": False, "formal_test_allowed": False})
        raise


if __name__ == "__main__":
    main()
