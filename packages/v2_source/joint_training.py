#!/usr/bin/env python3
"""Independent fixed100 R2 detection + depth candidate.

Frozen R2 helpers and all historical evidence remain read-only. Depth weight zero
takes the unchanged detector objective path without constructing a depth model.
"""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_key] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
IDENTITY = "r2_joint_depth_20261001"
CANDIDATE_ID = IDENTITY + "-two-detector-foa-plus-fixed-depth"
R2_MANIFEST_SHA = "92af9063dbe07898eb562b0d324e9d1bd746467ce1edea408f9b5f8940f82c85"
DEPTH_MODELS = ("monodepth2", "dpt_large")
CHILD_MANIFEST_SHA = "a4ee2aab2ab40fe9c122c16bc05fd5a32cbd06286413384f4ad4086c26b863cb"
SEED = 20260919
MODELS = ("fasterrcnn", "ssd300")
CONFIG = {"num_inference_steps": 100, "strength": .25, "outer_rounds": 4,
          "latent_lr": .05, "step_feedback_mode": "latent_chain", "guidance_scale": 4.5,
          "seed": SEED, "prompt": "top-down aerial view of clean asphalt road surface",
          "negative_prompt": "vehicle, car, obstacle"}
BUDGET = {"detector_forwards": 800, "autograd_calls": 100, "transformer_forward_calls": 100,
          "vae_encode_calls": 5, "vae_decode_calls": 205, "noise_draws": 4}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def configure_determinism():
    import numpy as np
    import torch
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError("set PYTHONHASHSEED=20260919 before Python starts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    assert_determinism()


def assert_determinism():
    import torch
    if (not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
        or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise RuntimeError("strict deterministic policy changed")


def verify_tree(directory, expected_sha):
    manifest = directory / "PACKAGE_MANIFEST.json"
    if sha(manifest) != expected_sha:
        raise RuntimeError("package manifest identity differs: " + str(directory))
    payload = json.loads(manifest.read_text())
    for entry in payload["files"]:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("unsafe package member")
        path = directory / rel
        if path.is_symlink() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise RuntimeError("package member changed: " + str(path))
    return {"manifest_sha256": expected_sha, "verified_files": len(payload["files"])}


def validate_depth_weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("depth weight must be a real scalar")
    if not math.isfinite(value) or value < 0:
        raise ValueError("depth weight must be finite and nonnegative")
    return float(value)


def execution_budget(depth_weight):
    budget = dict(BUDGET)
    if validate_depth_weight(depth_weight) != 0.0:
        budget.update(depth_clean_forwards=800, depth_adv_forwards=800)
    return budget


def bind_r2_source(directory):
    """Validate the exact accepted helper tree before making it importable."""
    directory = Path(directory).resolve()
    verified = verify_tree(directory, R2_MANIFEST_SHA)
    sys.path.insert(0, str(directory))
    return directory, verified


def preflight_training_assets(parent, records, model_manifest, car_path, mask_path,
                              weights_root, depth_weight):
    """The historical road preflight accepts detector IDs only."""
    depth_weight = validate_depth_weight(depth_weight)
    assets = parent.road_preflight(records, MODELS, model_manifest, car_path, mask_path, weights_root)
    depth_assets = None
    if depth_weight != 0.0:
        from ours_road_v16 import depth_asset_checks
        depth_assets = depth_asset_checks(DEPTH_MODELS, model_manifest)
    ready = assets["inputs_ready"] and (depth_assets is None or depth_assets["inputs_ready"])
    return assets, depth_assets, ready


def build_training_adapters(parent, device, model_manifest, weights_root, depth_weight):
    """Keep the historical detector construction order, with a true zero branch."""
    depth_weight = validate_depth_weight(depth_weight)
    adapters = parent.build_selected_adapters(MODELS, device, model_manifest, weights_root)
    for adapter in adapters.values():
        adapter.model.eval().requires_grad_(False)
    if depth_weight == 0.0:
        return adapters, None
    import numpy as np
    import torch
    # Pretrained constructors can initialize parameters before loading weights.
    # Their draws must not change the common pipeline initialization or RNG
    # evidence. Capture only after the unchanged detector-loading path above.
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        from depth_objective import DepthObjective
        depth_adapters = parent.build_selected_adapters(DEPTH_MODELS, device, model_manifest, weights_root)
        return adapters, DepthObjective(parent, depth_adapters, beta=.1)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def load_parent(child):
    # Load the old read-only helper under a unique name; our training.py is not
    # imported as the historical `training` package.
    name = "_meeting_full_historical_inputs"
    spec = importlib.util.spec_from_file_location(name, child / "training/inputs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.setup_parent_paths()
    parent = module.parent_runner()
    from projection_cpu_backward import install_road_rgb_projection_override
    install_road_rgb_projection_override()
    return module, parent


def denoise_with_evidence(pipeline, latent, timestep, prompt_pack, guidance_scale):
    """Same CFG/Euler arithmetic as frozen parent.standard_denoise_step."""
    import torch
    prompt, negative, pooled, negative_pooled = prompt_pack
    use_guidance = guidance_scale > 1.0
    with torch.no_grad():
        latent_input = torch.cat([latent] * 2) if use_guidance else latent
        expanded_timestep = timestep.expand(latent_input.shape[0])
        raw_prediction = pipeline.transformer(
            hidden_states=latent_input, timestep=expanded_timestep,
            encoder_hidden_states=torch.cat([negative, prompt]) if use_guidance else prompt,
            pooled_projections=torch.cat([negative_pooled, pooled]) if use_guidance else pooled,
            return_dict=False)[0]
        prediction = raw_prediction
        if use_guidance:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + guidance_scale * (conditional - unconditional)
        result = pipeline.scheduler.step(prediction, timestep, latent, return_dict=False)[0]
    return result, {"expanded_timestep": expanded_timestep,
                    "transformer_raw_prediction": raw_prediction,
                    "cfg_noise_prediction": prediction}


def compute_losses(parent, patch, context, views, adapters, ledger):
    import torch
    from components import soft_score_iou_loss
    from geometry_eot import transform_tensor, transform_box
    from evidence import fingerprint
    from losses import target_vehicle_candidates
    rendered = parent.RoadRGBView(patch).render(context)
    values = {name: [] for name in MODELS}
    details = []
    for view in views:
        image = transform_tensor(rendered.image, view)
        box = transform_box(context.object_box, view, *image.shape[-2:])
        row = {"view_index": view["index"], "input_fingerprint": fingerprint(image),
               "target_box": box, "models": {}}
        for name in MODELS:
            output = ledger.call("detector_forwards", adapters[name].predict, image)[0]
            scores, overlaps = target_vehicle_candidates(output, box)
            loss = soft_score_iou_loss(scores, overlaps, temperature=.1)
            if loss.ndim or not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite/nonscalar detector loss")
            values[name].append(loss)
            row["models"][name] = {"boxes": output["boxes"].detach(),
                "labels": output["labels"].detach(), "scores_all": output["scores"].detach(),
                "vehicle_scores": scores.detach(), "vehicle_overlaps": overlaps.detach(),
                "loss": loss.detach(), "empty_candidates": scores.numel() == 0}
        details.append(row)
    raw = [torch.stack(values[name]).mean() for name in MODELS]
    return raw, {"views": details, "rendered_image_fingerprint": fingerprint(rendered.image),
                  "rendered_support_fingerprint": fingerprint(rendered.support)}


def run_optimization(*, parent, config, output, pipeline, prompt_pack, initial_patch,
                     schedule, context_for, adapters, ledger, recorder, device, dtype, eot_config,
                     loss_evaluator=compute_losses, depth_weight=0.0, depth_objective=None):
    import torch
    from adaptive_loss import foa_weights, weighted_loss
    from scale_update import scaled_update
    from components import tensor_statistics
    from evidence import fingerprint, rng_snapshot, scheduler_snapshot
    from geometry_eot import sample_views
    if {key: getattr(config, key) for key in CONFIG} != CONFIG:
        raise ValueError("scientific SD3.5 parameters changed")
    depth_weight = validate_depth_weight(depth_weight)
    depth_enabled = depth_weight != 0.0
    if depth_enabled and depth_objective is None:
        raise ValueError("positive depth weight requires a depth objective")
    plan = parent.build_execution_plan(config, schedule)
    if len(plan) != 100 or config.adversarial_updates != 100:
        raise ValueError("exactly 100 effective updates required")
    trace, checkpoints = [], {}
    output = Path(output)
    trace_path = output / "attack_trace.jsonl"
    trace_path.open("x").close()
    previous_losses = None
    previous_source = None
    recorder.save("initialization", 0, {"config": config.to_dict(), "initial_patch": initial_patch,
        "prompt_pack": prompt_pack, "previous_raw_losses": None,
        "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
    with torch.no_grad():
        initial_latent = ledger.call("vae_encode_calls", parent.encode_patch, pipeline, initial_patch, dtype)
        reconstructed = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, initial_latent)
    parent.save_patch(output / "initial_vae_reconstruction.pt", reconstructed,
        {"role": "diagnostic_vae_reconstruction_not_zero_step_control"}, raw_generator_rgb=reconstructed)
    current_patch = initial_patch
    for outer_index in range(4):
        latent_start = ledger.call("vae_encode_calls", parent.encode_patch, pipeline, current_patch, dtype)
        pipeline.scheduler.set_timesteps(100, device=device)
        timesteps = pipeline.scheduler.timesteps[75:]
        sigmas = pipeline.scheduler.sigmas[75:75 + len(timesteps)].to(device=device, dtype=dtype)
        if len(timesteps) != 25 or len(sigmas) != 25:
            raise RuntimeError("scheduler must supply 25 active timesteps")
        noise_seed = SEED + 1000 * outer_index
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        rng_before_noise = rng_snapshot({"outer_noise": generator})
        noise = ledger.call("noise_draws", torch.randn, latent_start.shape,
                            generator=generator, device=device, dtype=dtype)
        latent = (1.0 - sigmas[0]) * latent_start + sigmas[0] * noise
        recorder.save("outer_start", 0, {"outer_round": outer_index + 1, "noise_seed": noise_seed,
            "noise": noise, "latent_start": latent_start, "latents": latent,
            "current_patch": current_patch, "timesteps": timesteps, "sigmas": sigmas,
            "previous_raw_losses": previous_losses, "previous_loss_source": previous_source,
            "rng_before_noise": rng_before_noise, "rng": rng_snapshot({"outer_noise": generator}),
            "scheduler": scheduler_snapshot(pipeline.scheduler)})
        for timestep in timesteps:
            planned = plan[len(trace)]
            step = planned.global_step
            record = schedule[step - 1]
            if record.split != "train":
                raise RuntimeError("non-train optimization input")
            assert_determinism()
            context = context_for(record)
            bindings = {}
            bindings["before_denoise"] = recorder.save("before_denoise", step, {
                "planned": asdict(planned), "legacy_planned_render_seed_unused": planned.render_seed,
                "actual_eot_seed": SEED + step * 1000, "sample_id": record.sample_id,
                "latents": latent, "timestep": timestep, "previous_raw_losses": previous_losses,
                "previous_loss_source": previous_source, "rng": rng_snapshot(),
                "scheduler": scheduler_snapshot(pipeline.scheduler)})
            denoised, denoise_evidence = ledger.call("transformer_forward_calls", denoise_with_evidence,
                pipeline, latent, timestep, prompt_pack, config.guidance_scale)
            z = denoised.detach().requires_grad_(True)
            patch = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, z)
            with torch.no_grad():
                rendered_geometry = parent.RoadRGBView(torch.zeros((1, 3, 64, 128), device=device)).render(context)
                yy, xx = torch.where(rendered_geometry.support[0, 0] > 0)
                if not xx.numel():
                    raise RuntimeError("road patch support is empty")
                patch_box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            height, width = rendered_geometry.image.shape[-2:]
            views = sample_views(height, width, list(context.object_box), patch_box,
                                 SEED + step * 1000, eot_config)
            try:
                raw, candidates = loss_evaluator(parent, patch, context, views, adapters, ledger)
                vector = torch.stack(raw)
                weights, weight_details = foa_weights(vector, previous_losses)
                total = weighted_loss(raw, weights)
                depth_state = {}
                if depth_enabled:
                    detection_total = total
                    depth_loss, depth_evidence = depth_objective.call(patch, context, views, ledger)
                    if (not isinstance(depth_loss, torch.Tensor) or depth_loss.ndim != 0
                        or not depth_loss.is_floating_point() or not bool(torch.isfinite(depth_loss))):
                        raise FloatingPointError("depth loss must be a finite floating scalar")
                    if depth_loss.device != detection_total.device or depth_loss.dtype != detection_total.dtype:
                        raise TypeError("depth and detector scalar losses must share dtype/device")
                    if not depth_loss.requires_grad:
                        raise RuntimeError("depth objective detached its adversarial computation graph")
                    json.dumps(depth_evidence, allow_nan=False)
                    weighted_depth_loss = depth_weight * depth_loss
                    total = detection_total + weighted_depth_loss
                    if not bool(torch.isfinite(total)):
                        raise FloatingPointError("joint total loss is nonfinite")
                    depth_state = {"detection_total_loss": detection_total.detach(),
                        "depth_loss": depth_loss.detach(), "depth_weight": depth_weight,
                        "weighted_depth_loss": weighted_depth_loss.detach(),
                        "depth_evidence": depth_evidence}
            except BaseException:
                recorder.save("failed_forward", step, {"latent_variable": z, "raw_patch": patch,
                    "denoise": denoise_evidence, "eot": views, "previous_raw_losses": previous_losses,
                    "previous_loss_source": previous_source, "rng": rng_snapshot(),
                    "scheduler": scheduler_snapshot(pipeline.scheduler)})
                raise
            bindings["pre_backward"] = recorder.save("pre_backward", step, {
                "latent_variable": z, "raw_patch": patch, "denoise": denoise_evidence,
                "raw_losses": vector, "previous_raw_losses": previous_losses,
                "previous_loss_source": previous_source, "adaptive_weights": weights,
                "weight_details": weight_details, "weighted_total_loss": total, **depth_state,
                "candidate_outputs": candidates, "eot": views,
                "background_fingerprint": fingerprint(context.background),
                "object_box": list(context.object_box), "road_quad": context.road_quad,
                "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
            assert_determinism()
            gradient, = ledger.call("autograd_calls", torch.autograd.grad, total, z)
            if not bool(torch.isfinite(gradient).all()):
                recorder.save("failed_gradient", step, {"latent_variable": z, "weighted_loss_gradient": gradient})
                raise FloatingPointError("weighted gradient contains NaN/Inf")
            # No model filtering, norm threshold, normalization or update clamp.
            try:
                next_latent, update_tensors, update_details = scaled_update(z, gradient, 100000.0, .05)
            except BaseException:
                recorder.save("failed_update", step, {"latent_variable": z, "weighted_loss_gradient": gradient,
                    "cast_gradient": gradient.detach().to(z), "amplified_gradient": 100000.0 * gradient.detach().to(z),
                    "proposed_update": .05 * (100000.0 * gradient.detach().to(z))})
                raise
            update_details.update(independent_one_step=False, trajectory="continuous_sd35_latent_chain")
            statistics = {"latent": tensor_statistics(z), "weighted_loss_gradient": tensor_statistics(gradient),
                          **{name: tensor_statistics(value) for name, value in update_tensors.items()}}
            bindings["pre_update"] = recorder.save("pre_update", step, {
                "latent_variable": z, "weighted_loss_gradient": gradient,
                "cast_gradient": gradient.detach().to(z), **update_tensors,
                "raw_losses": vector, "previous_raw_losses": previous_losses,
                "adaptive_weights": weights, "weighted_total_loss": total,
                "update_statistics": update_details, "tensor_statistics": statistics,
                "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
            latent = next_latent
            recorder.updates_applied += 1
            # Capture the exact detached current observation for the NEXT step.
            previous_losses = vector.detach().clone()
            previous_source = {"global_step": step, "sample_id": record.sample_id,
                               "eot_seed": SEED + step * 1000, "observation": "pre_update_raw_eot_means"}
            with torch.no_grad():
                patch_after = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latent)
            bindings["post_update"] = recorder.save("post_update", step, {
                "latents": latent, "raw_patch": patch_after,
                "next_previous_raw_losses": previous_losses, "next_previous_loss_source": previous_source,
                "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
            row = {**asdict(planned), "legacy_planned_render_seed_unused": planned.render_seed,
                   "actual_eot_seed": SEED + step * 1000, "split": record.split, "mathematical_objective": "meeting_LSE_tau0.1",
                   "training_models": list(MODELS), "eot_samples_per_update": 4,
                   "raw_losses": vector.detach().cpu().tolist(), "adaptive_weights": weights.cpu().tolist(),
                   "weighted_total_loss": float(total.detach()), "weight_details": weight_details,
                   "update_statistics": update_details, "tensor_statistics": statistics,
                   "state_bindings": bindings, "eot_seed": SEED + step * 1000,
                   "detector_forwards_this_step": 8, "backward_calls_this_step": 1}
            if depth_enabled:
                row.update(mathematical_objective="R2_detector_FOA_plus_fixed_depth_semantic_target_far",
                    training_depth_models=list(DEPTH_MODELS), training_tasks=["detection", "depth"],
                    depth_weight=depth_weight, depth_loss=float(depth_loss.detach()),
                    detection_total_loss=float(detection_total.detach()),
                    weighted_depth_loss=float(weighted_depth_loss.detach()), depth_evidence=depth_evidence,
                    depth_clean_forwards_this_step=8, depth_adv_forwards_this_step=8)
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            trace.append(row)
            print(f"meeting_full step={step}/100 sample={record.sample_id} loss={float(total.detach()):.9g}", flush=True)
            del raw, vector, total, gradient, z, patch, patch_after, candidates, denoise_evidence, update_tensors
            if depth_enabled:
                del depth_loss, detection_total, weighted_depth_loss, depth_evidence
            del depth_state
        with torch.no_grad():
            current_patch = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latent).detach()
        recorder.save("outer_end", 0, {"outer_round": outer_index + 1, "latents": latent,
            "raw_patch": current_patch, "previous_raw_losses": previous_losses,
            "previous_loss_source": previous_source, "rng": rng_snapshot(),
            "scheduler": scheduler_snapshot(pipeline.scheduler)})
        parent.save_rgb(current_patch, output / f"outer_{outer_index + 1:02d}_final.png")
        name, digest = parent.save_training_checkpoint(output, current_patch, len(trace))
        checkpoints[name] = digest
    if len(trace) != 100 or recorder.updates_applied != 100 or len(recorder.entries) != 409:
        raise RuntimeError("complete trajectory budget mismatch")
    ledger.require_complete()
    return current_patch, trace, checkpoints


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--r2-source", type=Path, required=True)
    parser.add_argument("--depth-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--sd35-snapshot", type=Path)
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_depth_weight(args.depth_weight)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if bool(args.source_prefix) != bool(args.source_root):
        parser.error("source remapping needs both paths")
    return args


def main(argv=None):
    args = parse_args(argv)
    if socket.gethostname() != args.expected_hostname:
        raise RuntimeError("live hostname mismatch")
    if args.output.exists():
        raise FileExistsError("output exists: no retry/resume/overwrite")
    for path in (args.output, *args.output.absolute().parents):
        if path.is_symlink():
            raise RuntimeError("output symlinks forbidden")
    child = args.child.resolve()
    verification = verify_tree(ROOT, args.package_manifest_sha256)
    child_verification = verify_tree(child, CHILD_MANIFEST_SHA)
    r2_source, r2_verification = bind_r2_source(args.r2_source)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU busy: " + active)
    existing = args.output.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < 2 * 1024 ** 3:
        raise RuntimeError("less than 2 GiB available for training evidence")
    configure_determinism()
    import torch
    import torchvision
    import diffusers
    import transformers
    import numpy as np
    import torch.nn.functional as F
    from evidence import CallLedger, StateRecorder
    if (torch.__version__.split("+")[0], torchvision.__version__.split("+")[0], diffusers.__version__.split("+")[0]) != ("2.8.0", "0.23.0", "0.38.0"):
        raise RuntimeError("unexpected frozen runtime versions")
    if not torch.cuda.is_available():
        raise RuntimeError("complete training requires CUDA")
    inputs, parent = load_parent(child)
    config = parent.StatefulOursConfig(**CONFIG)
    pr = child / "parent_v17"
    manifest = pr / "01_公共数据集/runtime_v1/train_validation_manifest.csv"
    records = parent.training_records(manifest, args.source_prefix, args.source_root)
    schedule = parent.seeded_round_robin(records, 100, SEED)
    expected_schedule = json.loads((child / "protocol/train_schedule.json").read_text())
    if [{"sample_id": r.sample_id, "dataset": r.dataset, "split": r.split} for r in schedule] != expected_schedule:
        raise RuntimeError("train order differs from frozen historical order")
    initial_record = next(r for r in records if r.sample_id == inputs.CALIBRATION_IDS[0])
    selected = {r.sample_id: r for r in (*schedule, initial_record)}
    if any(r.split != "train" for r in selected.values()):
        raise RuntimeError("training data include non-train record")
    model_manifest = pr / "02_模型/model_manifest.json"
    car_path, mask_path = (pr / "01_公共数据集" / name for name in ("car_00.jpg", "car_00_mask.jpg"))
    assets, depth_assets, assets_ready = preflight_training_assets(parent,
        list(selected.values()), model_manifest, car_path, mask_path, args.weights_root, args.depth_weight)
    inventory_path = pr / "00_冻结协议/sd35_snapshot_inventory.json"
    snapshot_path = args.sd35_snapshot or Path(json.loads(inventory_path.read_text())["snapshot_path"])
    snapshot = parent.validate_snapshot(snapshot_path, inventory_path, inputs.SD_MANIFEST_SHA)
    if not assets_ready or snapshot["status"] != "file_hashes_verified":
        raise RuntimeError("original train/detector/depth/car/full SD3.5 assets failed")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "PROTOCOL.md"
    if not protocol_path.is_file():
        raise RuntimeError("new explicit protocol missing")
    protocol_sha = sha(protocol_path)
    write_json(args.output / "preflight.json", {"verification": verification, "historical_source": child_verification,
        "assets": assets, "depth_assets": depth_assets, "sd35_snapshot": snapshot, "protocol_sha256": protocol_sha,
        "accepted_r2_source": r2_verification, "depth_weight": args.depth_weight,
        "invocation": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    shutil.copyfile(child / "protocol/train_schedule.json", args.output / "schedule.json")
    eot_config = json.loads((r2_source / "eot_config.json").read_text())
    ledger = CallLedger(execution_budget(args.depth_weight))
    recorder = StateRecorder(args.output, CANDIDATE_ID)
    started = time.monotonic()
    try:
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        car, mask = parent.load_car(car_path, mask_path, device)
        verified = set()
        context_for = lambda record: parent.build_context(record, car, mask, device, verified)
        initial_context = context_for(initial_record)
        initial_small = parent.extract_road_patch(initial_context.background, initial_context.road_quad)
        initial_patch = F.interpolate(initial_small, size=(512, 512), mode="bilinear", align_corners=False)
        parent.save_rgb(initial_patch, args.output / "input_patch.png")
        parent.save_patch(args.output / "zero_step_patch.pt", initial_small,
            {"role": "common_initial_road_crop_before_vae_or_noise", "candidate_id": CANDIDATE_ID})
        adapters, depth_objective = build_training_adapters(
            parent, device, model_manifest, args.weights_root, args.depth_weight)
        pipeline = parent.build_pipeline(snapshot_path, device, dtype)
        prompt_pack = parent.encode_prompt(pipeline, config)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()
        runtime = {"hostname": socket.gethostname(), "gpu_identity": gpu, "device": str(device),
            "python": platform.python_version(), "python_executable": sys.executable,
            "torch": torch.__version__, "torchvision": torchvision.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "deterministic_algorithms": True,
            "deterministic_warn_only": False, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"], "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "package_manifest_sha256": args.package_manifest_sha256, "protocol_sha256": protocol_sha,
            "scheduler_class": type(pipeline.scheduler).__module__ + "." + type(pipeline.scheduler).__qualname__,
            "scheduler_source_sha256": sha(inspect.getsourcefile(type(pipeline.scheduler))),
            "latent_dtype": str(dtype), "new_eot_sampling_device": "cpu", "global_spatial_override": False,
            "scoped_projection_override": {
                "scope": "road_patch_contract.perspective_project RGB sampler input VJP only",
                "forward": "unchanged native GPU bilinear/zeros/align_corners=True",
                "backward": "CPU aten.grid_sampler_2d_backward with output_mask=[True,False]",
                "input_dtype": "torch.float32", "grid_requires_grad": False,
                "alpha_and_homography_forward_unchanged": True,
                "module_sha256": sha(r2_source / "projection_cpu_backward.py"),
                "implementation_revision_sha256": sha(r2_source / "IMPLEMENTATION_REVISION.md")}}
        runtime.update(r2_source_manifest_sha256=R2_MANIFEST_SHA, depth_weight=args.depth_weight,
            depth_models=list(DEPTH_MODELS) if args.depth_weight != 0.0 else [],
            depth_objective="fixed equal mean of two semantic-target-far losses; beta=0.1",
            detector_foa_unchanged=True)
        runtime["repeatability_identity"] = dict(runtime)
        write_json(args.output / "runtime_provenance.json", runtime)
        assert_determinism()
        final, trace, checkpoints = run_optimization(parent=parent, config=config, output=args.output,
            pipeline=pipeline, prompt_pack=prompt_pack, initial_patch=initial_patch, schedule=schedule,
            context_for=context_for, adapters=adapters, ledger=ledger, recorder=recorder,
            device=device, dtype=dtype, eot_config=eot_config,
            depth_weight=args.depth_weight, depth_objective=depth_objective)
        parent.save_rgb(parent.canonicalize_rgb(final), args.output / "final_patch.png")
        parent.save_rgb(final, args.output / "raw_generator_rgb.png")
        parent.save_patch(args.output / "final_patch.pt", final,
                          {"role": "final_attack_patch", "candidate_id": CANDIDATE_ID}, raw_generator_rgb=final)
        state_sha = recorder.finish("complete", len(trace))
        summary = {"status": "train_optimization_complete", "identity": IDENTITY, "candidate_id": CANDIDATE_ID,
            "executed_updates": 100, "config": config.to_dict(), "training_models": list(MODELS),
            "training_tasks": ["detection", "depth"] if args.depth_weight != 0.0 else ["detection"],
            "training_depth_models": list(DEPTH_MODELS) if args.depth_weight != 0.0 else [],
            "depth_weight": args.depth_weight, "depth_beta": .1,
            "objective": "two-detector FOA scalar + depth_weight * equal-two-depth semantic-target-far scalar",
            "depth_disabled_path": args.depth_weight == 0.0,
            "accepted_r2_source_manifest_sha256": R2_MANIFEST_SHA,
            "seed": SEED, "latent_lr": .05, "gradient_multiplier": 100000.0, "update_clamp": None,
            "normalizations": [], "loss_temperature": .1, "adaptive_history": "previous actual pre-update EOT raw means",
            "history_resets_at_outer": False, "budget": ledger.snapshot(), "query_accounting": dict(ledger.completed),
            "state_evidence_manifest_sha256": state_sha, "protocol_sha256": protocol_sha,
            "package_manifest_sha256": args.package_manifest_sha256, "historical_source_manifest_sha256": CHILD_MANIFEST_SHA,
            "runtime_provenance_sha256": sha(args.output / "runtime_provenance.json"),
            "training_manifest_sha256": sha(manifest), "model_manifest_sha256": sha(model_manifest),
            "schedule_sha256": sha(args.output / "schedule.json"), "attack_trace_sha256": sha(args.output / "attack_trace.jsonl"),
            "checkpoint_bindings": checkpoints,
            "patch_bindings": {name: sha(args.output / name) for name in ("zero_step_patch.pt", "final_patch.pt", "initial_vae_reconstruction.pt")},
            "checkpoint_selection": "fixed final100 only", "formal_test_result": False,
            "test_data_opened": False, "validation_data_opened": False, "resume_enabled": False,
            "elapsed_seconds": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)}
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"status": summary["status"], "executed_updates": 100, "output": str(args.output)}, ensure_ascii=False), flush=True)
    except BaseException as error:
        if not (args.output / "state_evidence/MANIFEST.json").exists():
            recorder.finish("failed_retained", recorder.updates_applied)
        write_json(args.output / "FAILURE.json", {"status": "failed_retained", "executed_updates": recorder.updates_applied,
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
            "budget": ledger.snapshot(), "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False, "resume_enabled": False})
        raise


if __name__ == "__main__":
    main()
