#!/usr/bin/env python3
"""Train/validation-only v17 road baselines. No formal/test entry point."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
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
from road_runtime import (CORE, TRANSFER, build_context, build_offline_detector,
                          patch_brightness, patch_update, preflight, training_records)
from road_patch_contract import RoadRGBView, canonicalize_rgb, save_patch, load_patch
from road_detection_objectives import contextual_loss, robust_loss, make_robust_targets

SEED = 20260919
EOT_SAMPLES = 2
CANDIDATE_ID = "normalized-v17-roadpatch-20260920"
UPDATES = {"smoke": 2, "pilot": 20, "validation": 100}
CHECKPOINT_STEPS = (25, 50, 75, 100)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_rgb(path, tensor):
    data = (tensor.detach()[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(data).save(path)


def save_detection_preview(path, image, predictions, box):
    """Fixed display threshold only; AP metrics retain all evaluated candidates."""
    pixels = (image[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    for predicted_box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if int(label) in (3, 6, 8) and float(score) >= .3:
            coordinates = [float(x) for x in predicted_box]
            draw.rectangle(coordinates, outline="orange", width=1)
            draw.text((coordinates[0], coordinates[1]), f"{int(label)}:{float(score):.3f}", fill="orange")
    draw.rectangle(box, outline="cyan", width=2)
    canvas.save(path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "smoke", "pilot", "validation", "evaluate"), default="preflight")
    p.add_argument("--method", choices=("contextual", "robust_dpatch"), required=True)
    p.add_argument("--training-manifest", type=Path, default=ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv")
    p.add_argument("--model-manifest", type=Path, default=ROOT / "02_模型/model_manifest.json")
    p.add_argument("--weights-root", type=Path)
    p.add_argument("--source-prefix", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-output", type=Path, help="Completed smoke/pilot/validation training to evaluate on validation only")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def reserve_output(path):
    path = path.resolve()
    # Keep all local development output under a new candidate, never old deliveries.
    if ROOT.resolve() not in path.parents or "runs" not in path.relative_to(ROOT).parts:
        raise ValueError("output must be inside this candidate's runs/ directory")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("refusing to overwrite a previous run: " + str(path))


def objective(method, adapter, image, box, targets):
    if method == "contextual":
        return contextual_loss(adapter, image, box)
    return robust_loss(adapter, image, targets)


def optimize(args, records, device):
    started = time.monotonic()
    if any(record.split not in ("train", "validation") for record in records):
        raise ValueError("only train/validation records are allowed")
    if args.mode not in UPDATES:
        raise ValueError("unsupported optimization mode")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # CUDA reset is invalid before device initialization on the real server.
    # CPU is only exercised by explicitly mocked synthetic integration tests.
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    adapters = {m: build_offline_detector(m, device, args.model_manifest, args.weights_root) for m in CORE}
    car, mask = load_car(ROOT / "01_公共数据集/car_00.jpg", ROOT / "01_公共数据集/car_00_mask.jpg", device)
    calibration = calibration_records(records)
    context0 = build_context(calibration[0], car, mask, device)
    initial = canonicalize_rgb(extract_road_patch(context0.background, context0.road_quad)).detach()
    patch = torch.nn.Parameter(initial.clone())
    random_patch = torch.rand(initial.shape, generator=torch.Generator().manual_seed(SEED)).to(device)
    save_patch(args.output / "zero_step_patch.pt", initial, {"state": "zero", "initializer": calibration[0].sample_id})
    save_patch(args.output / "random_control_patch.pt", random_patch, {"state": "unoptimized_random_control", "seed": SEED})
    learning_rate = .01 if args.method == "contextual" else 2 / 255
    optimizer = torch.optim.Adam([patch], lr=learning_rate) if args.method == "contextual" else None
    counters = {"loss_forward": {m: 0 for m in CORE}, "input_backward": {m: 0 for m in CORE},
                "clean_pseudo_label_forward": {m: 0 for m in CORE}, "patch_updates": 0,
                "scene_exposures": 0, "calibration_unique_scenes": 4,
                "calibration_model_scene_forwards": 4 * len(CORE), "initialization_scene_reads": 1,
                "generator_forward": 0}
    targets_cache = {}

    def labels_for(record, context, model_id):
        key = (record.sample_id, model_id)
        if args.method != "robust_dpatch":
            return None
        if key not in targets_cache:
            with torch.no_grad():
                predicted = adapters[model_id].predict(context.benign)[0]
            targets_cache[key] = make_robust_targets(predicted, context.object_box, device)
            counters["clean_pseudo_label_forward"][model_id] += 1
        return targets_cache[key]

    normalizers = {}
    # Fixed four TRAIN images only. No score-based selection or validation fitting.
    for model_id, adapter in adapters.items():
        values = []
        for record in calibration:
            context = build_context(record, car, mask, device)
            labels = labels_for(record, context, model_id)
            # Adapter enforces differentiability even for a forward-only scale
            # measurement. This leaf is not optimized and receives no backward.
            image = RoadRGBView(initial).render(context).image.detach().requires_grad_(True)
            loss, stats = objective(args.method, adapter, image, context.object_box, labels)
            counters["loss_forward"][model_id] += 1
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite calibration loss")
            values.append(abs(float(loss.detach())))
            del loss, image
        normalizers[model_id] = max(sum(values) / len(values), 1e-6)
    write_json(args.output / "calibration.json", {"sample_ids": [r.sample_id for r in calibration],
               "normalizers": normalizers, "rule": "mean absolute zero-patch native loss, floor1e-6; equal model mean"})
    schedule = seeded_round_robin(records, UPDATES[args.mode], SEED)
    write_json(args.output / "schedule.json", [{"sample_id": r.sample_id, "dataset": r.dataset, "split": r.split} for r in schedule])
    schedule_sha256 = sha256(args.output / "schedule.json")
    checkpoint_bindings = {}
    gradient_diagnostics = {
        "model_eot_evaluations": {m: {str(draw): 0 for draw in range(EOT_SAMPLES)} for m in CORE},
        "zero_model_eot_gradients": {m: {str(draw): 0 for draw in range(EOT_SAMPLES)} for m in CORE},
        "zero_model_update_counts": {m: 0 for m in CORE},
        "zero_total_gradient_steps": [],
        "rule": "exact zero; retain each model/EOT observation; smoke rejects a model with all draws zero; every mode rejects a zero aggregate gradient",
    }
    trace_path = args.output / "trace.jsonl"
    for step, record in enumerate(schedule, 1):
        step_started = time.monotonic()
        context = build_context(record, car, mask, device)
        total_gradient = torch.zeros_like(patch)
        rows = []
        before = patch.detach().clone()
        for model_id, adapter in adapters.items():
            labels = labels_for(record, context, model_id)
            for draw in range(EOT_SAMPLES):
                generator = torch.Generator(device=device).manual_seed(SEED + 1000 * step + draw)
                transformed, brightness = patch_brightness(patch, generator)
                render = RoadRGBView(transformed).render(context)
                torch.manual_seed(SEED + 1000 * step + draw)
                loss, stats = objective(args.method, adapter, render.image, context.object_box, labels)
                counters["loss_forward"][model_id] += 1
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite attack loss")
                gradient = torch.autograd.grad(loss / normalizers[model_id], patch)[0]
                counters["input_backward"][model_id] += 1
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("nonfinite model gradient")
                gradient_l2 = float(gradient.norm())
                gradient_diagnostics["model_eot_evaluations"][model_id][str(draw)] += 1
                gradient_diagnostics["zero_model_eot_gradients"][model_id][str(draw)] += int(gradient_l2 == 0)
                total_gradient += gradient.detach() / (len(CORE) * EOT_SAMPLES)
                rows.append({"model": model_id, "draw": draw, "loss": float(loss.detach()),
                             "gradient_l2": gradient_l2, "zero_gradient": gradient_l2 == 0,
                             "brightness": brightness, "details": stats})
        zero_models = [m for m in CORE if not any(r["gradient_l2"] > 0 for r in rows if r["model"] == m)]
        for model_id in zero_models:
            gradient_diagnostics["zero_model_update_counts"][model_id] += 1
        if not total_gradient.detach().abs().any():
            gradient_diagnostics["zero_total_gradient_steps"].append(step)
        write_json(args.output / "gradient_diagnostics.json", gradient_diagnostics)
        if zero_models or not total_gradient.detach().abs().any():
            with (args.output / "zero_gradient_events.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "sample_id": record.sample_id,
                    "zero_models": zero_models, "total_gradient_l2": float(total_gradient.norm()),
                    "model_draws": rows}, allow_nan=False) + "\n")
        if zero_models and args.mode == "smoke":
            write_json(args.output / "zero_gradient_diagnostic.json", {
                "step": step, "sample_id": record.sample_id, "zero_models": zero_models, "model_draws": rows})
            raise RuntimeError("smoke found no road-patch gradient for " + ",".join(zero_models))
        patch_update(patch, total_gradient, args.method, learning_rate, optimizer)
        counters["patch_updates"] += 1
        counters["scene_exposures"] += 1
        if args.mode == "validation" and step in CHECKPOINT_STEPS:
            name = "checkpoint_%04d.pt" % step
            save_patch(args.output / name, patch.detach(), {
                "state": "checkpoint", "candidate_id": CANDIDATE_ID, "method": args.method,
                "updates": step, "seed": SEED, "train_schedule_sha256": schedule_sha256})
            checkpoint_bindings[name] = sha256(args.output / name)
            # Keep completed checkpoints bound even if a later update fails.
            write_json(args.output / "checkpoint_bindings.json", checkpoint_bindings)
        event = {"step": step, "sample_id": record.sample_id, "dataset": record.dataset,
                 "gradient_l2": float(total_gradient.norm()), "max_rgb_delta": float((patch.detach()-before).abs().max()),
                 "seconds": time.monotonic()-step_started, "model_draws": rows}
        with trace_path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps({k: event[k] for k in ("step", "sample_id", "gradient_l2", "max_rgb_delta")}), flush=True)
    save_patch(args.output / "final_patch.pt", patch.detach(), {"state": "final", "candidate_id": CANDIDATE_ID,
               "method": args.method, "updates": len(schedule), "seed": SEED,
               "train_schedule_sha256": schedule_sha256})
    final_reload = load_patch(args.output / "final_patch.pt").to(device)
    if not torch.equal(final_reload, patch.detach()):
        raise RuntimeError("canonical save/reload differs")
    # Fixed first training image in each domain, never selected by attack scores.
    for dataset in ("kitti", "nuscenes"):
        record = next(r for r in calibration if r.dataset == dataset)
        context = build_context(record, car, mask, device)
        final = RoadRGBView(final_reload).render(context).image
        train_path = RoadRGBView(patch.detach()).render(context).image
        if not torch.equal(final, train_path):
            raise RuntimeError("train/eval RGB mismatch")
        zero = RoadRGBView(initial).render(context).image
        random_image = RoadRGBView(random_patch).render(context).image
        save_rgb(args.output / (dataset + "_clean_zero_random_final.png"),
                 torch.cat((context.benign, zero, random_image, final), dim=-2))
        torch.save({"clean": context.benign.cpu(), "zero": zero.cpu(), "random": random_image.cpu(),
                    "final": final.cpu(), "target_box": context.object_box,
                    "sample_id": record.sample_id, "split": "train"}, args.output / (dataset + "_preview.pt"))
    torch.cuda.synchronize(device)
    return {"status": "development_optimization_complete", "candidate_id": CANDIDATE_ID,
            "method": args.method, "mode": args.mode, "seed": SEED,
            "calibration_sample_ids": [record.sample_id for record in calibration],
            "training_manifest_sha256": sha256(args.training_manifest),
            "model_manifest_sha256": sha256(args.model_manifest),
            "adaptation": "shared road patch; not a faithful original-paper reproduction",
            "train_models": CORE, "updates": len(schedule), "learning_rate": learning_rate,
            "eot": {"patch_brightness": [.9, 1.1], "draws_per_update": EOT_SAMPLES},
            "budget_counts": counters, "domain_exposures": dict(Counter(r.dataset for r in schedule)),
            "train_schedule_sha256": schedule_sha256, "schedule_sha256": schedule_sha256,
            "checkpoint_bindings": checkpoint_bindings,
            "checkpoint_steps": list(CHECKPOINT_STEPS) if args.mode == "validation" else [],
            "gradient_diagnostics": gradient_diagnostics,
            "seconds": time.monotonic()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "formal_test_allowed": False, "test_data_opened": False,
            "final_patch_sha256": sha256(args.output / "final_patch.pt"),
            "patch_sha256": {name: sha256(args.output / name) for name in
                             ("zero_step_patch.pt", "random_control_patch.pt", "final_patch.pt")}}


def evaluate(args, records, device):
    from evaluator import detection_candidates, detection_summary, VEHICLE_CLASS_IDS, CAR_CLASS_IDS
    if args.run_output is None:
        raise ValueError("evaluation requires --run-output")
    if any(record.split not in ("train", "validation") for record in records):
        raise ValueError("only train/validation records are allowed")
    run = json.loads((args.run_output / "summary.json").read_text())
    if run.get("status") != "development_optimization_complete" or run.get("method") != args.method:
        raise ValueError("not a completed matching development run")
    if run.get("candidate_id") != CANDIDATE_ID or run.get("mode") not in UPDATES:
        raise ValueError("run identity does not match this v17 candidate")
    if sha256(args.run_output / "schedule.json") != run.get("train_schedule_sha256"):
        raise RuntimeError("training schedule changed since optimization")
    if sha256(args.run_output / "final_patch.pt") != run["final_patch_sha256"]:
        raise RuntimeError("final patch changed since optimization")
    for name in ("zero_step_patch.pt", "random_control_patch.pt", "final_patch.pt"):
        if sha256(args.run_output / name) != run.get("patch_sha256", {}).get(name):
            raise RuntimeError("patch/control changed since optimization: " + name)
    limit = 2 if run["mode"] == "smoke" else 20
    selected = [r for d in ("kitti", "nuscenes") for r in
                sorted((x for x in records if x.dataset == d and x.split == "validation"), key=lambda x: x.sample_id)[:limit]]
    if Counter(record.dataset for record in selected) != {"kitti": limit, "nuscenes": limit}:
        raise ValueError("fixed validation cohort is incomplete")
    car, mask = load_car(ROOT / "01_公共数据集/car_00.jpg", ROOT / "01_公共数据集/car_00_mask.jpg", device)
    patches = {state: load_patch(args.run_output / name).to(device) for state, name in
               (("zero", "zero_step_patch.pt"), ("random", "random_control_patch.pt"), ("final", "final_patch.pt"))}
    all_rows, summaries = [], []
    # Transfer models reserved for later locked evaluation, never short-run selection.
    for model_id in CORE:
        adapter = build_offline_detector(model_id, device, args.model_manifest, args.weights_root)
        by_state = {state: [] for state in patches}
        with torch.no_grad():
            for record in selected:
                context = build_context(record, car, mask, device)
                clean = adapter.predict(context.benign)[0]
                visual = record == next(r for r in selected if r.dataset == record.dataset)
                if visual:
                    prefix = model_id + "_" + record.dataset
                    save_detection_preview(args.output / (prefix + "_clean.png"), context.benign, clean, context.object_box)
                    raw = {"sample_id": record.sample_id, "split": "validation", "target_box": context.object_box,
                           "display_threshold_only": .3, "model": model_id,
                           "clean": {k: v.detach().cpu() for k, v in clean.items()}}
                for state, patch in patches.items():
                    image = RoadRGBView(patch).render(context).image
                    output = adapter.predict(image)[0]
                    if visual:
                        save_detection_preview(args.output / (prefix + "_" + state + ".png"), image, output, context.object_box)
                        raw[state] = {k: v.detach().cpu() for k, v in output.items()}
                    row = {"model": model_id, "sample_id": record.sample_id, "dataset": record.dataset,
                           "split": "validation", "state": state,
                           "clean_candidates": detection_candidates(clean, context.object_box, VEHICLE_CLASS_IDS),
                           "adversarial_candidates": detection_candidates(output, context.object_box, VEHICLE_CLASS_IDS),
                           "clean_car_candidates": detection_candidates(clean, context.object_box, CAR_CLASS_IDS),
                           "adversarial_car_candidates": detection_candidates(output, context.object_box, CAR_CLASS_IDS)}
                    by_state[state].append(row); all_rows.append(row)
                if visual:
                    torch.save(raw, args.output / (prefix + "_predictions.pt"))
            for state, rows in by_state.items():
                for domain in ("kitti", "nuscenes", "combined"):
                    domain_rows = [r for r in rows if domain == "combined" or r["dataset"] == domain]
                    summaries.append(dict(detection_summary(domain_rows, 1000, SEED, allow_zero_reference=True),
                                          model=model_id, state=state, dataset=domain))
        del adapter
        torch.cuda.empty_cache()
    write_json(args.output / "per_sample.json", all_rows)
    return {"status": "development_validation_complete", "candidate_id": CANDIDATE_ID,
            "records": len(all_rows), "summaries": summaries,
            "test_data_opened": False, "formal_test_allowed": False,
            "metric": "tnr_det=(target_vehicle_AP50_clean-target_vehicle_AP50_adv)/target_vehicle_AP50_clean",
            "note": "No outcome filtering; transfer evaluation withheld during development."}


def main():
    args = arguments()
    reserve_output(args.output)
    try:
        records = training_records(args.training_manifest, args.source_prefix, args.source_root)
        checks = preflight(records, CORE, args.model_manifest, ROOT / "01_公共数据集/car_00.jpg",
                           ROOT / "01_公共数据集/car_00_mask.jpg", args.weights_root)
        write_json(args.output / "input_preflight.json", checks)
        write_json(args.output / "invocation.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        import platform
        import torchvision
        write_json(args.output / "runtime.json", {
            "candidate_id": CANDIDATE_ID,
            "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "model_manifest_sha256": sha256(args.model_manifest), "training_manifest_sha256": sha256(args.training_manifest),
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.rglob("*.py")
                              if "runs" not in p.relative_to(ROOT).parts},
        })
        if args.mode == "preflight":
            result = {"status": "preflight_complete", "candidate_id": CANDIDATE_ID,
                      "inputs_ready": checks["inputs_ready"],
                      "missing_or_invalid": sum(c["status"] != "pass" for c in checks["checks"]),
                      "formal_test_allowed": False, "test_data_opened": False}
        else:
            if not checks["inputs_ready"]:
                raise RuntimeError("missing or altered inputs; see input_preflight.json")
            device = torch.device(args.device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("real pretrained-model smoke requires a CUDA server")
            result = evaluate(args, records, device) if args.mode == "evaluate" else optimize(args, records, device)
        write_json(args.output / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(args.output / "failure.json", {"candidate_id": CANDIDATE_ID,
                   "type": type(exc).__name__, "error": str(exc),
                   "status": "failed_retained", "test_data_opened": False, "formal_test_allowed": False})
        raise


if __name__ == "__main__":
    main()
