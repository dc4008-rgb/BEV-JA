#!/usr/bin/env python3
"""Fixed v23 C0/C3 train-only run; metadata preflight is the default."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from training import inputs


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_output(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("output symlink is forbidden")
    resolved = inputs.candidate_run_path(path)
    if resolved.exists():
        raise FileExistsError("output already exists; no overwrite or resume: " + str(resolved))
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("C0", "C3"), required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--allow-unsealed-local-preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--package-manifest-sha256")
    parser.add_argument("--sd35-snapshot", type=Path)
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    args = parser.parse_args(argv)
    if args.allow_unsealed_local_preflight and args.mode != "preflight":
        parser.error("unsealed input is permitted only for local metadata preflight")
    if bool(args.source_prefix) != bool(args.source_root):
        parser.error("source remapping needs both --source-prefix and --source-root")
    args.output = args.output or ROOT / "runs" / ("train_" + args.objective)
    return args


def evaluate_model_losses(*, parent, loss_function, objective, ledger, patch, context,
                          record, run_config, method_contract, adapters, normalizers,
                          device, global_step, seed, phase):
    import torch
    from training.state_evidence import rng_snapshot
    if phase not in ("train_pre", "train_post"):
        raise ValueError("invalid optimization phase")
    ledger.phase = phase
    canonical = parent.canonicalize_rgb(patch)
    values = {name: [] for name in run_config.train_models}
    details = {name: {"components_by_eot_draw": [], "objective_diagnostics_by_eot_draw": []}
               for name in run_config.train_models}
    factors, draws = [], []
    for draw in range(2):
        generator = parent.make_generator(device, seed + 1000 * global_step + draw)
        before = rng_snapshot({"eot": generator})
        transformed, factor = parent.patch_brightness(canonical, generator, (0.9, 1.1))
        factors.append(factor)
        rendered = parent.RoadRGBView(transformed).render(context, eot=None, generator=generator)
        draws.append({"draw": draw, "seed": seed + 1000 * global_step + draw,
                      "brightness": factor, "rng_before": before,
                      "rng_after": rng_snapshot({"eot": generator})})
        for name in run_config.train_models:
            result = loss_function(run_config, method_contract, name, adapters[name], context,
                                   rendered.image, rendered.support, {}, record.sample_id,
                                   attack_objective=objective, phase=phase)
            if result.total.ndim != 0 or not torch.isfinite(result.total):
                raise RuntimeError("nonfinite/nonscalar objective: " + name)
            values[name].append(result.total)
            details[name]["components_by_eot_draw"].append(result.scalars())
            details[name]["objective_diagnostics_by_eot_draw"].append(getattr(result, "diagnostics", {}))
            del result
    losses = []
    for name in run_config.train_models:
        raw = torch.stack(values[name]).mean()
        normalized = raw / normalizers[name]
        losses.append(normalized)
        details[name].update(raw=float(raw.detach()), normalized=float(normalized.detach()),
                             eot_brightness_factors=list(factors))
    return losses, details, draws


def run_optimization(*, parent, config, protocol, objective, output, pipeline, prompt_pack,
                     initial_patch, schedule, context_for, run_config, method_contract,
                     adapters, normalizers, calibration_sha, ledger, recorder,
                     loss_function, device, dtype):
    """The actual fixed loop; injected dependencies enable synthetic CPU checks."""
    import torch
    from training.state_evidence import fingerprint, gradient_stats, rng_snapshot, scheduler_snapshot
    output = Path(output)
    calibration_path = output / "calibration.json"
    normalizer_fingerprint = fingerprint(normalizers)
    plan = parent.build_execution_plan(config, schedule)
    if len(plan) != 100 or config.adversarial_updates != 100:
        raise ValueError("only the fixed 100-update plan is implemented")
    trace, checkpoints = [], {}
    recorder.updates_applied = 0
    trace_path = output / "attack_trace.jsonl"
    with trace_path.open("x"):
        pass
    recorder.save("initialization", 0, {
        "config": config.to_dict(), "initial_patch": initial_patch, "prompt_pack": prompt_pack,
        "normalizers": normalizers, "calibration_sha256": calibration_sha,
        "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler),
        "optimizer": "manual latent subtraction; no optimizer state"})
    with torch.no_grad():
        initial_latent = ledger.call("vae_encode_calls", parent.encode_patch, pipeline, initial_patch, dtype)
        initial_reconstruction = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, initial_latent)
    parent.save_patch(output / "initial_vae_reconstruction.pt", initial_reconstruction,
                      {"role": "diagnostic_vae_reconstruction_not_zero_step_control"},
                      raw_generator_rgb=initial_reconstruction)
    current_patch = initial_patch
    for outer_index in range(config.outer_rounds):
        attack_progress = torch.zeros(2, device=device)
        latent_start = ledger.call("vae_encode_calls", parent.encode_patch, pipeline, current_patch, dtype)
        pipeline.scheduler.set_timesteps(config.num_inference_steps, device=device)
        start_index = config.num_inference_steps - config.active_steps_per_round
        timesteps = pipeline.scheduler.timesteps[start_index:]
        sigmas = pipeline.scheduler.sigmas[start_index:start_index + len(timesteps)].to(device=device, dtype=dtype)
        if len(timesteps) != 25 or len(sigmas) != 25:
            raise RuntimeError("scheduler did not provide exactly 25 active steps")
        noise_seed = config.seed + 1000 * outer_index
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        noise_rng_before = rng_snapshot({"outer_noise": generator})
        noise = ledger.call("noise_draws", torch.randn, latent_start.shape,
                            generator=generator, device=device, dtype=dtype)
        latents = (1.0 - sigmas[0]) * latent_start + sigmas[0] * noise
        recorder.save("outer_start", 0, {"outer_round": outer_index + 1, "noise_seed": noise_seed,
                      "noise": noise, "latent_start": latent_start, "latents": latents,
                      "current_patch": current_patch, "timesteps": timesteps, "sigmas": sigmas,
                      "attack_progress": attack_progress, "rng_before_noise": noise_rng_before,
                      "rng": rng_snapshot({"outer_noise": generator}),
                      "scheduler": scheduler_snapshot(pipeline.scheduler)})
        for timestep in timesteps:
            planned = plan[len(trace)]
            record = schedule[planned.global_step - 1]
            if record.split != "train":
                raise RuntimeError("non-train optimization input")
            if inputs.sha(calibration_path) != calibration_sha or fingerprint(normalizers) != normalizer_fingerprint:
                raise RuntimeError("calibration changed after freeze")
            context = context_for(record)
            bindings = {}
            bindings["before_denoise"] = recorder.save("before_denoise", planned.global_step, {
                "planned": asdict(planned), "sample_id": record.sample_id, "latents": latents,
                "timestep": timestep, "attack_progress": attack_progress,
                "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler),
                "calibration_sha256": calibration_sha, "normalizers": normalizers})
            denoised = ledger.call("transformer_forward_calls", parent.standard_denoise_step,
                                  pipeline, latents, timestep, prompt_pack, config.guidance_scale)
            latent_variable = denoised.detach().requires_grad_(True)
            patch_before = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latent_variable)
            common = dict(parent=parent, loss_function=loss_function, objective=objective, ledger=ledger,
                          context=context, record=record, run_config=run_config, method_contract=method_contract,
                          adapters=adapters, normalizers=normalizers, device=device,
                          global_step=planned.global_step, seed=config.seed)
            losses_pre, details_pre, draws_pre = evaluate_model_losses(patch=patch_before, phase="train_pre", **common)
            gradients = []
            try:
                for index, loss in enumerate(losses_pre):
                    gradients.append(ledger.call("autograd_calls", torch.autograd.grad,
                                     loss, latent_variable, retain_graph=index < len(losses_pre) - 1)[0])
            except BaseException:
                recorder.save("partial_gradients", 0, {"failed_global_step": planned.global_step,
                    "latent_variable": latent_variable, "raw_patch": patch_before,
                    "per_model_gradients": dict(zip(run_config.train_models, gradients)),
                    "losses_pre": details_pre, "eot": draws_pre,
                    "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
                raise
            gradient_evidence = {name: gradient_stats(value) for name, value in zip(run_config.train_models, gradients)}
            bindings["gradients"] = recorder.save("gradients", planned.global_step, {
                "latent_variable": latent_variable, "raw_patch": patch_before,
                "per_model_gradients": dict(zip(run_config.train_models, gradients)),
                "losses_pre": details_pre, "gradient_stats": gradient_evidence,
                "eot": draws_pre, "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
            control = parent.combine_detector_gradients(gradients)
            for index, name in enumerate(run_config.train_models):
                gradient_evidence[name]["controller_actual_float32_norm"] = float(control.gradient_norms[index])
                gradient_evidence[name]["controller_active"] = index not in control.zero_model_gradient_indices
            bindings["pre_update"] = recorder.save("pre_update", planned.global_step, {
                "latent_variable": latent_variable, "joint_gradient": control.joint_gradient,
                "joint_gradient_cast_for_update": control.joint_gradient.to(latent_variable),
                "latent_lr": config.latent_lr, "gradient_norms": control.gradient_norms,
                "model_weights": control.model_weights, "zero_model_gradient_indices": control.zero_model_gradient_indices,
                "gradient_state": bindings["gradients"], "attack_progress": attack_progress,
                "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
            latents = (latent_variable.detach() - config.latent_lr * control.joint_gradient.to(latent_variable)).detach()
            recorder.updates_applied += 1
            if not torch.isfinite(latents).all():
                recorder.save("failed_update", 0, {"failed_global_step": planned.global_step,
                              "latents": latents, "rng": rng_snapshot()})
                raise RuntimeError("latent update produced NaN/Inf")
            try:
                with torch.no_grad():
                    patch_after = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latents)
                    losses_post, details_post, draws_post = evaluate_model_losses(patch=patch_after, phase="train_post", **common)
            except BaseException:
                recorder.save("partial_post_update", 0, {"failed_global_step": planned.global_step,
                    "latents": latents, "pre_update_state": bindings["pre_update"],
                    "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
                raise
            attack_progress = parent.detector_progress(losses_pre, losses_post)
            bindings["post_update"] = recorder.save("post_update", planned.global_step, {
                "latents": latents, "raw_patch": patch_after, "losses_post": details_post,
                "attack_progress": attack_progress, "pre_update_state": bindings["pre_update"],
                "eot": draws_post, "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
            row = {**asdict(planned), "split": record.split, "attack_objective": objective,
                   "candidate_id": protocol["candidate_ids"][objective],
                   "normalizer_fingerprint": normalizer_fingerprint,
                   "model_count": 2, "training_models": list(run_config.train_models), "task_variant": "det-only",
                   "eot_samples_per_update": 2, "victim_forwards_minimum_this_update": 8,
                   "latent_backward_queries_this_update": 2, "timestep": float(timestep.detach()),
                   "latent_lr": config.latent_lr, "losses_pre": details_pre, "losses_post": details_post,
                   "attack_progress_next": parent.tensor_list(attack_progress),
                   "gradient_norms": parent.tensor_list(control.gradient_norms),
                   "model_weights": parent.tensor_list(control.model_weights), "gradient_conflict": None,
                   "task_cosine": None, "task_projection_applied": False,
                   "zero_model_gradient_indices": list(control.zero_model_gradient_indices),
                   "gradient_evidence": gradient_evidence, "texture": parent.texture_diagnostics(patch_after),
                   "state_bindings": bindings, "calibration_sha256": calibration_sha}
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            trace.append(row)
            print(f"objective={objective} global={len(trace)}/100 sample={record.sample_id}", flush=True)
            del losses_pre, losses_post, gradients, control, latent_variable, patch_before, patch_after
        current_patch = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latents).detach()
        recorder.save("outer_end", 0, {"outer_round": outer_index + 1, "latents": latents,
                      "raw_patch": current_patch, "attack_progress": attack_progress,
                      "rng": rng_snapshot(), "scheduler": scheduler_snapshot(pipeline.scheduler)})
        parent.save_rgb(current_patch, output / f"outer_{outer_index + 1:02d}_final.png")
        name, digest = parent.save_training_checkpoint(output, current_patch, len(trace))
        checkpoints[name] = digest
    parent.audit_candidate_trace(trace, config, "det-only")
    ledger.require_complete()
    return current_patch, trace, checkpoints


def runtime_gate(args, protocol):
    import inspect
    import platform
    import torch
    import torchvision
    import diffusers
    import transformers
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    if not args.expected_hostname or socket.gethostname() != args.expected_hostname:
        raise RuntimeError("live hostname must match explicit --expected-hostname")
    if not args.package_manifest_sha256 or inputs.sha(ROOT / "PACKAGE_MANIFEST.json") != args.package_manifest_sha256:
        raise RuntimeError("explicit sealed package SHA mismatch")
    if torch.__version__.split("+")[0] != protocol["required_torch_base_version"] or torchvision.__version__.split("+")[0] != protocol["required_torchvision_base_version"]:
        raise RuntimeError("unsupported torch/torchvision runtime")
    if diffusers.__version__.split("+")[0] != protocol["required_diffusers_base_version"]:
        raise RuntimeError("unsupported diffusers runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("real training requires CUDA")
    process = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=30)
    if process.stdout.strip():
        raise RuntimeError("GPU is not empty: " + process.stdout)
    existing = args.output.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < protocol["minimum_stage_free_disk_bytes"]:
        raise RuntimeError("less than 2 GiB free for train stage")
    source_objects = {"torch": torch, "torch_autograd": torch.autograd,
                      "torch_nn_functional": torch.nn.functional,
                      "flow_match_euler_scheduler": FlowMatchEulerDiscreteScheduler}
    sources = {}
    for name, value in source_objects.items():
        path = inspect.getsourcefile(value)
        if not path or not Path(path).is_file():
            raise RuntimeError("cannot bind installed runtime source: " + name)
        sources[name] = {"path": str(Path(path).resolve()), "sha256": inputs.sha(path)}
    return {"hostname": socket.gethostname(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "python": platform.python_version(), "python_executable": sys.executable,
            "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "installed_source_files": sources,
            "empty_gpu_process_query": process.stdout, "free_disk_bytes": free,
            "package_manifest_sha256": args.package_manifest_sha256}


def train(args, protocol, verification):
    import torch
    import torch.nn.functional as F
    from training.objective_loss import model_loss, calibrate_normalizers
    from training.state_evidence import CallLedger, CountingAdapter, StateRecorder
    parent = inputs.parent_runner()
    output = safe_output(args.output)
    runtime = runtime_gate(args, protocol)
    manifest = inputs.PARENT / "01_公共数据集/runtime_v1/train_validation_manifest.csv"
    model_manifest = inputs.PARENT / "02_模型/model_manifest.json"
    car_path, mask_path = (inputs.PARENT / "01_公共数据集" / name for name in ("car_00.jpg", "car_00_mask.jpg"))
    records = parent.training_records(manifest, args.source_prefix, args.source_root)
    config = parent.StatefulOursConfig(**protocol["stateful_sd35"])
    schedule = parent.seeded_round_robin(records, 100, config.seed)
    actual_schedule = [{"sample_id": record.sample_id, "dataset": record.dataset, "split": record.split}
                       for record in schedule]
    if actual_schedule != json.loads((ROOT / "protocol/train_schedule.json").read_text()):
        raise RuntimeError("live remapped train schedule differs from the exact frozen order")
    calibrators = parent.calibration_records(records)
    selected = {r.sample_id: r for r in (*schedule, *calibrators)}
    if any(r.split != "train" for r in selected.values()):
        raise RuntimeError("training assets include a non-train record")
    assets = parent.road_preflight(list(selected.values()), protocol["training_models"], model_manifest,
                                   car_path, mask_path, args.weights_root)
    snapshot = parent.validate_snapshot(args.sd35_snapshot,
        inputs.PARENT / "00_冻结协议/sd35_snapshot_inventory.json", inputs.SD_MANIFEST_SHA)
    if not assets["inputs_ready"] or snapshot["status"] != "file_hashes_verified":
        raise RuntimeError("original train/weights/car/full-SD asset hashes failed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", {"verification": verification, "runtime": runtime,
                                          "assets": assets, "sd35_snapshot": snapshot})
    # Preserve the byte-exact frozen schedule hash, not merely equivalent JSON.
    shutil.copyfile(ROOT / "protocol/train_schedule.json", output / "schedule.json")
    ledger = CallLedger(protocol["budget_per_run"])
    candidate_id = protocol["candidate_ids"][args.objective]
    recorder = StateRecorder(output, candidate_id, args.objective)
    started = time.monotonic()
    try:
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        parent.setup_seed(config.seed)
        _, contracts = parent.load_method_contracts(inputs.PARENT / "03_方法/production/method_contracts.json")
        run_config = parent.RunConfig(config_id=candidate_id, method_id="ours", task="detection",
                                     variant="v23_det_only_" + args.objective, train_models=tuple(protocol["training_models"]), depth_direction="far")
        car, car_mask = parent.load_car(car_path, mask_path, device)
        verified = set()
        context_for = lambda record: parent.build_context(record, car, car_mask, device, verified)
        context = context_for(calibrators[0])
        initial_small = parent.extract_road_patch(context.background, context.road_quad)
        initial_patch = F.interpolate(initial_small, size=(512, 512), mode="bilinear", align_corners=False)
        parent.save_rgb(initial_patch, output / "input_patch.png")
        parent.save_patch(output / "zero_step_patch.pt", initial_small,
                          {"role": "common_initial_road_crop_before_vae_or_noise", "candidate_id": candidate_id})
        adapters = {name: CountingAdapter(adapter, ledger) for name, adapter in
                    parent.build_selected_adapters(run_config.train_models, device, model_manifest, args.weights_root).items()}
        ledger.phase = "calibration"
        with torch.no_grad():
            normalizers, calibration = calibrate_normalizers(run_config, contracts["ours"], parent.RoadRGBView(initial_small.detach()),
                adapters, calibrators, car, car_mask, device, config.seed, verified, {}, attack_objective=args.objective)
        calibration.update(candidate_id=candidate_id, protocol_sha256=inputs.sha(inputs.PROTOCOL_PATH))
        write_json(output / "calibration.json", calibration)
        calibration_sha = inputs.sha(output / "calibration.json")
        pipeline = parent.build_pipeline(args.sd35_snapshot, device, dtype)
        import inspect
        scheduler_source = inspect.getsourcefile(type(pipeline.scheduler))
        write_json(output / "runtime_provenance.json", {**runtime,
            "actual_scheduler_class": type(pipeline.scheduler).__module__ + "." + type(pipeline.scheduler).__qualname__,
            "actual_scheduler_source": {"path": scheduler_source, "sha256": inputs.sha(scheduler_source)},
            "cuda_device_name": torch.cuda.get_device_name(device)})
        prompt_pack = parent.encode_prompt(pipeline, config)
        current_patch, trace, checkpoints = run_optimization(parent=parent, config=config, protocol=protocol,
            objective=args.objective, output=output, pipeline=pipeline, prompt_pack=prompt_pack,
            initial_patch=initial_patch, schedule=schedule, context_for=context_for, run_config=run_config,
            method_contract=contracts["ours"], adapters=adapters, normalizers=normalizers, calibration_sha=calibration_sha,
            ledger=ledger, recorder=recorder, loss_function=model_loss, device=device, dtype=dtype)
        parent.save_rgb(parent.canonicalize_rgb(current_patch), output / "final_patch.png")
        parent.save_rgb(current_patch, output / "raw_generator_rgb.png")
        parent.save_rgb(parent.canonicalize_rgb(current_patch), output / "final_patch_64x128.png")
        parent.save_patch(output / "final_patch.pt", current_patch,
                          {"role": "final_attack_patch", "candidate_id": candidate_id}, raw_generator_rgb=current_patch)
        state_sha = recorder.finish("complete", len(trace))
        summary = {"status": "train_optimization_complete", "identity": inputs.IDENTITY,
                   "candidate_id": candidate_id, "attack_objective": args.objective, "task_variant": "det-only",
                   "training_models": list(run_config.train_models), "model_count": 2, "road_eot": protocol["road_eot"],
                   "canonical_patch_hw": [64, 128], "formal_test_result": False, "test_data_opened": False,
                   "validation_data_opened": False, "config": config.to_dict(), "executed_updates": len(trace),
                   "seed": config.seed, "schedule_sha256": inputs.sha(output / "schedule.json"),
                   "attack_trace_sha256": inputs.sha(output / "attack_trace.jsonl"),
                   "checkpoint_bindings": checkpoints, "checkpoint_selection": "fixed final100 only",
                   "budget": ledger.snapshot(), "query_accounting": dict(ledger.completed),
                   "normalizers": normalizers, "calibration_sample_ids": [r.sample_id for r in calibrators],
                   "calibration_sha256": calibration_sha, "protocol_sha256": inputs.sha(inputs.PROTOCOL_PATH),
                   "package_manifest_sha256": verification["manifest_sha256"],
                   "runtime_provenance_sha256": inputs.sha(output / "runtime_provenance.json"),
                   "state_evidence_manifest_sha256": state_sha, "training_manifest_sha256": inputs.sha(manifest),
                   "model_manifest_sha256": inputs.sha(model_manifest),
                   "patch_bindings": {name: inputs.sha(output / name) for name in
                                      ("zero_step_patch.pt", "initial_vae_reconstruction.pt", "final_patch.pt")},
                   "elapsed_seconds": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
                   "final_texture": parent.texture_diagnostics(current_patch), "resume_enabled": False,
                   "recomputed_trajectory_bitwise_recovery_guaranteed": False}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except BaseException as error:
        if getattr(error, "progress_evidence", None) is not None:
            write_json(output / "CALIBRATION_FAILURE.json", error.progress_evidence)
        trace_path = output / "attack_trace.jsonl"
        completed = len(trace_path.read_text().splitlines()) if trace_path.exists() else 0
        applied = getattr(recorder, "updates_applied", 0)
        if not (output / "state_evidence/MANIFEST.json").exists():
            recorder.finish("failed_retained", applied)
        write_json(output / "FAILURE.json", {"status": "failed_retained", "candidate_id": candidate_id,
            "attack_objective": args.objective, "executed_updates": applied, "completed_trace_rows": completed,
            "budget": ledger.snapshot(),
            "exception_type": type(error).__name__, "exception": str(error),
            "elapsed_seconds": time.monotonic() - started, "resume_enabled": False})
        raise


def main(argv=None):
    args = parse_args(argv)
    verification = inputs.verify_inputs(require_sealed=not args.allow_unsealed_local_preflight)
    protocol = inputs.load_protocol()
    if args.package_manifest_sha256 and verification.get("manifest_sha256") != args.package_manifest_sha256:
        raise RuntimeError("explicit package SHA mismatch")
    if args.mode == "preflight":
        inputs.candidate_run_path(args.output)
        print(json.dumps({**verification, "candidate_id": protocol["candidate_ids"][args.objective],
              "attack_objective": args.objective, "status": "metadata_preflight_pass",
              "budget_per_run": protocol["budget_per_run"], "runtime_gpu_verified": False,
              "source_images_opened": False, "models_loaded": False}, ensure_ascii=False, indent=2))
        return
    train(args, protocol, verification)


if __name__ == "__main__":
    main()
