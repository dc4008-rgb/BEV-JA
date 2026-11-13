#!/usr/bin/env python3
"""Run the train/validation-only stateful SD3.5 ours candidate.

Independent v17 road-only candidate: common canonical patch serialization,
full SD3.5 denoising, detector-only or dual-task gradients, train/validation only.
Preflight is the default; no weights are fetched and formal test is forbidden.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "03_方法" / "production"
MODEL_DIR = ROOT / "02_模型"
DATA_DIR = ROOT / "01_公共数据集"
UPSTREAM = ROOT / "03_方法" / "ours" / "upstream_snapshot"
for path in (PRODUCTION, MODEL_DIR, DATA_DIR, UPSTREAM):
    sys.path.insert(0, str(path))

from advrm_insertion import load_car  # noqa: E402
from carriers import extract_road_patch  # noqa: E402
from data import (  # noqa: E402
    calibration_records,
    seeded_round_robin,
)
from road_patch_contract import canonicalize_rgb, RoadRGBView, save_patch
from road_runtime import (training_records, preflight as road_preflight, build_offline_detector,
                          patch_brightness)
from ours_road_v16 import (CANDIDATE_PREFIX, model_order, combine_detector_gradients,
                          detector_progress, audit_candidate_trace, depth_asset_checks, require_candidate_run_path)
from objectives import load_method_contracts  # noqa: E402
from ours_stateful import (  # noqa: E402
    StatefulOursConfig,
    build_execution_plan,
    combine_four_model_gradients,
    compute_attack_progress,
    reject_test_input,
    texture_diagnostics,
)
from trainer import (  # noqa: E402
    RunConfig,
    build_adapter,
    build_context,
    calibrate_normalizers,
    make_generator,
    model_loss,
)


# Compatibility name; the shared carrier owns all canonicalization and rendering.
DecodedRoadCarrier = RoadRGBView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd35-snapshot", type=Path)
    parser.add_argument("--sd35-file-manifest", type=Path)
    parser.add_argument("--sd35-file-manifest-sha256")
    parser.add_argument("--task-variant", choices=("det-only", "dual"), default="det-only")
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=ROOT / "01_公共数据集" / "runtime_v1" / "train_validation_manifest.csv",
    )
    parser.add_argument(
        "--method-contracts",
        type=Path,
        default=ROOT / "03_方法" / "production" / "method_contracts.json",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=ROOT / "02_模型" / "model_manifest.json",
    )
    parser.add_argument(
        "--car",
        type=Path,
        default=ROOT / "01_公共数据集" / "car_00.jpg",
    )
    parser.add_argument(
        "--car-mask",
        type=Path,
        default=ROOT / "01_公共数据集" / "car_00_mask.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--mode", choices=("preflight", "smoke", "validation"), default="preflight")
    parser.add_argument("--smoke-active-steps", type=int, default=1)
    parser.add_argument("--save-step-patches", action="store_true")
    args = parser.parse_args()
    args.contract = args.contract or ROOT / "00_冻结协议" / "ours_road_v16_candidate" / (args.task_variant + ".json")
    args.output = args.output or ROOT / "runs" / ("ours_" + args.task_variant) / args.mode
    args.output = require_candidate_run_path(args.output, ROOT)
    return args


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_contract(path: Path, task_variant: str = "det-only") -> tuple[dict[str, Any], StatefulOursConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scope") != "train_validation_only":
        raise ValueError("ours candidate contract must be train_validation_only")
    if payload.get("test_access_allowed") is not False:
        raise ValueError("ours candidate contract must forbid test access")
    if payload.get("candidate_id") != CANDIDATE_PREFIX + task_variant:
        raise ValueError("contract is not this independent v16 task variant")
    if payload.get("task_variant") != task_variant:
        raise ValueError("CLI task variant differs from candidate contract")
    if payload.get("formal_test_allowed") is not False:
        raise ValueError("formal testing remains forbidden for this candidate")
    if tuple(payload.get("training_models", ())) != model_order(task_variant):
        raise ValueError("candidate model order differs from selected task variant")
    if payload.get("road_eot") != {"samples_per_update": 2, "brightness": [0.9, 1.1], "seed_rule": "seed + 1000 * global_step + draw"}:
        raise ValueError("candidate road EOT contract differs")
    config = StatefulOursConfig(**payload["stateful_sd35"])
    config.validate()
    if (config.num_inference_steps, config.strength, config.outer_rounds,
        config.latent_lr, config.step_feedback_mode, config.seed) != (100, 0.25, 4, 0.05, "latent_chain", 20260919):
        raise ValueError("the first v16 candidate parameters differ from the accepted configuration")
    if config.to_dict()["adversarial_updates"] != 100:
        raise ValueError("the confirmed candidate must contain exactly 100 attack updates")
    return payload, config


def validate_snapshot(snapshot: Path, manifest: Path | None = None, manifest_sha: str | None = None) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise FileNotFoundError(f"SD3.5 snapshot directory is missing: {snapshot}")
    required = (
        "model_index.json",
        "scheduler",
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
    )
    missing = [name for name in required if not (snapshot / name).exists()]
    if missing:
        raise FileNotFoundError(f"incomplete SD3.5 snapshot; missing {missing}")
    for module in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        directory = snapshot / module
        if not (directory / "config.json").is_file() or not any(
            path.suffix in {".safetensors", ".bin"} for path in directory.rglob("*") if path.is_file()
        ):
            raise FileNotFoundError("SD3.5 module config/weights missing: " + module)
    if not (snapshot / "scheduler/scheduler_config.json").is_file():
        raise FileNotFoundError("SD3.5 scheduler config missing")
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    evidence = {
        "path": str(snapshot.resolve()),
        "model_index_sha256": sha256(snapshot / "model_index.json"),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "status": "structure_present_file_hashes_unbound",
    }
    if manifest is None and manifest_sha is None:
        return evidence
    if manifest is None or manifest_sha is None or sha256(manifest) != manifest_sha:
        raise ValueError("SD3.5 file manifest requires its matching explicit SHA-256")
    payload = json.loads(manifest.read_text())
    if payload.get("model_id") != "stabilityai/stable-diffusion-3.5-medium":
        raise ValueError("SD3.5 snapshot manifest has a different model identity")
    entries = payload.get("files", [])
    expected = [str(Path(entry["path"])) for entry in entries]
    actual = [str(path.relative_to(snapshot)) for path in files]
    if len(set(expected)) != len(expected) or set(expected) != set(actual):
        raise ValueError("SD3.5 snapshot file set differs from bound inventory")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("SD3.5 manifest file paths must be snapshot-relative")
        file = snapshot / relative
        if sha256(file) != entry["sha256"] or file.stat().st_size != entry["bytes"]:
            raise ValueError("SD3.5 snapshot artifact mismatch: " + str(relative))
    evidence.update(status="file_hashes_verified", manifest_sha256=manifest_sha,
                    manifest_path=str(manifest.resolve()))
    return evidence


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rgb(rgb: torch.Tensor, path: Path) -> None:
    array = (
        rgb.detach()
        .float()
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def build_pipeline(snapshot: Path, device: torch.device, dtype: torch.dtype):
    from diffusers import StableDiffusion3Pipeline

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        snapshot,
        torch_dtype=dtype,
        local_files_only=True,
        text_encoder_3=None,
        tokenizer_3=None,
    ).to(device)
    for module in (
        pipeline.transformer,
        pipeline.vae,
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
    ):
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    return pipeline


def encode_patch(pipeline, image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    normalized = image.to(pipeline.device, dtype=dtype) * 2.0 - 1.0
    with torch.no_grad():
        encoded = pipeline.vae.encode(normalized).latent_dist.mode()
    return (
        encoded - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor


def decode_patch(pipeline, latent: torch.Tensor) -> torch.Tensor:
    scaled = latent / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
    return ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def encode_prompt(pipeline, config: StatefulOursConfig):
    with torch.no_grad():
        return pipeline.encode_prompt(
            prompt=config.prompt,
            prompt_2=config.prompt,
            prompt_3=config.prompt,
            negative_prompt=config.negative_prompt,
            do_classifier_free_guidance=config.guidance_scale > 1.0,
            device=pipeline.device,
            num_images_per_prompt=1,
        )


def standard_denoise_step(
    pipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_pack: tuple[torch.Tensor, ...],
    guidance_scale: float,
) -> torch.Tensor:
    prompt_embeds, negative_prompt_embeds, pooled, negative_pooled = prompt_pack
    use_guidance = guidance_scale > 1.0
    with torch.no_grad():
        latent_input = torch.cat([latents] * 2) if use_guidance else latents
        expanded_timestep = timestep.expand(latent_input.shape[0])
        noise_prediction = pipeline.transformer(
            hidden_states=latent_input,
            timestep=expanded_timestep,
            encoder_hidden_states=(
                torch.cat([negative_prompt_embeds, prompt_embeds])
                if use_guidance
                else prompt_embeds
            ),
            pooled_projections=(
                torch.cat([negative_pooled, pooled]) if use_guidance else pooled
            ),
            return_dict=False,
        )[0]
        if use_guidance:
            unconditional, conditional = noise_prediction.chunk(2)
            noise_prediction = unconditional + guidance_scale * (conditional - unconditional)
        return pipeline.scheduler.step(
            noise_prediction, timestep, latents, return_dict=False
        )[0]


def evaluate_model_losses(
    *, patch, context, record, run_config, method_contract, adapters,
    normalizers, benign_depth_cache, device, global_step, seed,
):
    """Average two common patch-only EOT views before per-model autograd.

    Pre/post use identical seeds; vehicle/background pixels and patch support
    remain unchanged. There is no depth forward in a detector-only run.
    """
    canonical = canonicalize_rgb(patch)
    values = {model_id: [] for model_id in run_config.train_models}
    components = {model_id: [] for model_id in run_config.train_models}
    factors = []
    for draw in range(2):
        generator = make_generator(device, seed + 1000 * global_step + draw)
        transformed, factor = patch_brightness(canonical, generator, (0.9, 1.1))
        factors.append(factor)
        rendered = RoadRGBView(transformed).render(context, eot=None, generator=generator)
        for model_id in run_config.train_models:
            result = model_loss(run_config, method_contract, model_id, adapters[model_id],
                                context, rendered.image, rendered.support,
                                benign_depth_cache, record.sample_id)
            values[model_id].append(result.total)
            components[model_id].append(result.scalars())
    losses, details = [], {}
    for model_id in run_config.train_models:
        raw = torch.stack(values[model_id]).mean()
        normalized = raw / normalizers[model_id]
        losses.append(normalized)
        details[model_id] = {"raw": float(raw.detach()), "normalized": float(normalized.detach()),
                             "components_by_eot_draw": components[model_id],
                             "eot_brightness_factors": factors}
    return losses, details


def build_selected_adapters(order, device, manifest, weights_root=None):
    return {model_id: (build_offline_detector(model_id, device, manifest, weights_root)
                      if model_id in ("fasterrcnn", "ssd300")
                      else build_adapter(model_id, device, manifest))
            for model_id in order}


def tensor_list(value: torch.Tensor) -> list[Any]:
    return value.detach().float().cpu().tolist()


def save_training_checkpoint(output: Path, patch: torch.Tensor, step: int) -> tuple[str, str]:
    """Prespecified observation points; no best-checkpoint selection."""
    if step not in (25, 50, 75, 100):
        raise ValueError("checkpoint is not a prespecified 25-step observation")
    name = f"checkpoint_{step:04d}.pt"
    path = output / name
    if path.exists():
        raise FileExistsError(path)
    save_patch(path, patch, {"state": "train_checkpoint", "updates": step,
                            "selection": "fixed_step_not_score_selected"})
    return name, sha256(path)


def main() -> None:
    args = parse_args()
    contract_payload, config = load_candidate_contract(args.contract, args.task_variant)
    order = model_order(args.task_variant)
    if args.sd35_snapshot is None or not args.sd35_snapshot.is_dir():
        snapshot_evidence = {"status": "missing", "path": str(args.sd35_snapshot),
                             "required_for_gpu": True}
    else:
        snapshot_evidence = validate_snapshot(args.sd35_snapshot, args.sd35_file_manifest, args.sd35_file_manifest_sha256)
    records = training_records(args.training_manifest, args.source_prefix, args.source_root)
    reject_test_input(args.training_manifest, records)
    schedule = seeded_round_robin(records, config.adversarial_updates, config.seed)
    plan = build_execution_plan(config, schedule)
    preflight = {
        "candidate_id": contract_payload["candidate_id"],
        "scope": contract_payload["scope"],
        "task_variant": args.task_variant,
        "training_models": list(order),
        "formal_test_allowed": False,
        "road_eot": contract_payload["road_eot"],
        "input_assets": road_preflight(records, order[:2], args.model_manifest,
                                       args.car, args.car_mask, args.weights_root),
        "depth_assets": depth_asset_checks(order, args.model_manifest),
        "gpu_verified": False,
        "config": config.to_dict(),
        "training_manifest": str(args.training_manifest.resolve()),
        "training_manifest_sha256": sha256(args.training_manifest),
        "sd35_snapshot": snapshot_evidence,
        "plan_rows": len(plan),
        "dataset_counts": {
            dataset: sum(row.dataset == dataset for row in plan)
            for dataset in ("kitti", "nuscenes")
        },
        "test_data_opened": False,
    }
    if args.mode == "preflight":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    if snapshot_evidence["status"] != "file_hashes_verified":
        raise RuntimeError("complete local SD3.5 snapshot and full file inventory SHA must be bound before GPU execution")
    if not preflight["input_assets"]["inputs_ready"] or not preflight["depth_assets"]["inputs_ready"]:
        raise RuntimeError("train/validation source or selected model/car asset checks failed")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stateful SD3.5 validation requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    setup_seed(config.seed)
    started = time.monotonic()

    _, method_contracts = load_method_contracts(args.method_contracts)
    method_contract = method_contracts["ours"]
    run_config = RunConfig(
        config_id="ours__road_v17_" + args.task_variant,
        method_id="ours",
        task="detection" if args.task_variant == "det-only" else "joint",
        variant="road_v17_" + args.task_variant,
        train_models=order,
        depth_direction="far",
    )
    car, car_mask = load_car(args.car, args.car_mask, device)
    calibrators = calibration_records(records)
    verified: set[str] = set()
    initial_context = build_context(calibrators[0], car, car_mask, device, verified)
    initial_small = extract_road_patch(initial_context.background, initial_context.road_quad)
    initial_patch = F.interpolate(
        initial_small,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    save_rgb(initial_patch, args.output / "input_patch.png")
    save_patch(args.output / "zero_step_patch.pt", initial_small,
               {"role": "common_initial_road_crop_before_vae_or_noise", "candidate_id": contract_payload["candidate_id"]})

    adapters = build_selected_adapters(order, device, args.model_manifest, args.weights_root)
    benign_depth_cache: dict[tuple[str, str], torch.Tensor] = {}
    normalizer_carrier = RoadRGBView(initial_small.detach())
    normalizers = calibrate_normalizers(
        run_config,
        method_contract,
        normalizer_carrier,
        adapters,
        calibrators,
        car,
        car_mask,
        device,
        config.seed,
        verified,
        benign_depth_cache,
    )
    pipeline = build_pipeline(args.sd35_snapshot, device, dtype)
    prompt_pack = encode_prompt(pipeline, config)
    with torch.no_grad():
        initial_reconstruction = decode_patch(pipeline, encode_patch(pipeline, initial_patch, dtype))
    save_patch(args.output / "initial_vae_reconstruction.pt", initial_reconstruction,
               {"role": "diagnostic_vae_reconstruction_not_zero_step_control"},
               raw_generator_rgb=initial_reconstruction)
    current_patch = initial_patch
    trace: list[dict[str, Any]] = []
    checkpoint_bindings: dict[str, str] = {}
    (args.output / "schedule.json").write_text(json.dumps(
        [{"sample_id": r.sample_id, "dataset": r.dataset, "split": r.split} for r in schedule],
        ensure_ascii=False, indent=2) + "\n")
    effective_rounds = 1 if args.mode == "smoke" else config.outer_rounds
    plan_position = 0
    for outer_index in range(effective_rounds):
        attack_progress = torch.zeros(len(order), device=device)
        latent_start = encode_patch(pipeline, current_patch, dtype)
        pipeline.scheduler.set_timesteps(config.num_inference_steps, device=device)
        active = config.active_steps_per_round
        start_index = config.num_inference_steps - active
        timesteps = pipeline.scheduler.timesteps[start_index:]
        sigmas = pipeline.scheduler.sigmas[start_index : start_index + len(timesteps)].to(
            device=device, dtype=dtype
        )
        if args.mode == "smoke":
            if not 1 <= args.smoke_active_steps <= active:
                raise ValueError("smoke-active-steps must be within the active schedule")
            timesteps = timesteps[: args.smoke_active_steps]
            sigmas = sigmas[: args.smoke_active_steps]
        if len(timesteps) != len(sigmas) or len(timesteps) == 0:
            raise RuntimeError("invalid active timestep/sigma schedule")

        noise_seed = config.seed + 1000 * outer_index
        noise = torch.randn(
            latent_start.shape,
            generator=torch.Generator(device=device).manual_seed(noise_seed),
            device=device,
            dtype=dtype,
        )
        latents = (1.0 - sigmas[0]) * latent_start + sigmas[0] * noise
        for inner_index, timestep in enumerate(timesteps, start=1):
            planned = plan[plan_position]
            plan_position += 1
            record = schedule[planned.global_step - 1]
            if record.split != "train":
                raise RuntimeError("optimization attempted to consume a non-train row")
            context = build_context(record, car, car_mask, device, verified)
            denoised = standard_denoise_step(
                pipeline, latents, timestep, prompt_pack, config.guidance_scale
            )
            latent_variable = denoised.detach().requires_grad_(True)
            patch_before = decode_patch(pipeline, latent_variable)
            losses_pre, details_pre = evaluate_model_losses(
                patch=patch_before,
                context=context,
                record=record,
                run_config=run_config,
                method_contract=method_contract,
                adapters=adapters,
                normalizers=normalizers,
                benign_depth_cache=benign_depth_cache,
                device=device,
                global_step=planned.global_step,
                seed=config.seed,
            )
            gradients = [
                torch.autograd.grad(
                    loss,
                    latent_variable,
                    retain_graph=index < len(losses_pre) - 1,
                )[0]
                for index, loss in enumerate(losses_pre)
            ]
            control = (combine_detector_gradients(gradients) if args.task_variant == "det-only"
                       else combine_four_model_gradients(gradients, attack_progress))
            latents = (
                latent_variable.detach()
                - config.latent_lr * control.joint_gradient.to(latent_variable)
            ).detach()
            if not torch.isfinite(latents).all():
                raise RuntimeError("latent update produced NaN/Inf")
            with torch.no_grad():
                patch_after = decode_patch(pipeline, latents)
                losses_post, details_post = evaluate_model_losses(
                    patch=patch_after,
                    context=context,
                    record=record,
                    run_config=run_config,
                    method_contract=method_contract,
                    adapters=adapters,
                    normalizers=normalizers,
                    benign_depth_cache=benign_depth_cache,
                    device=device,
                    global_step=planned.global_step,
                    seed=config.seed,
                )
            attack_progress = (detector_progress(losses_pre, losses_post) if args.task_variant == "det-only"
                               else compute_attack_progress(losses_pre, losses_post))
            row = {
                **asdict(planned),
                "split": record.split,
                "model_count": len(order),
                "training_models": list(order),
                "task_variant": args.task_variant,
                "eot_samples_per_update": 2,
                "victim_forwards_minimum_this_update": len(order) * 2 * 2,
                "latent_backward_queries_this_update": len(order),
                "timestep": float(timestep.detach()),
                "latent_lr": config.latent_lr,
                "losses_pre": details_pre,
                "losses_post": details_post,
                "attack_progress_next": tensor_list(attack_progress),
                "gradient_norms": tensor_list(control.gradient_norms),
                "model_weights": tensor_list(control.model_weights),
                "gradient_conflict": None if args.task_variant == "det-only" else tensor_list(control.gradient_conflict),
                "task_cosine": None if args.task_variant == "det-only" else float(control.task_cosine.detach()),
                "task_projection_applied": False if args.task_variant == "det-only" else control.task_projection_applied,
                "zero_model_gradient_indices": list(
                    control.zero_model_gradient_indices
                ),
                "texture": texture_diagnostics(patch_after),
            }
            trace.append(row)
            with (args.output / "attack_trace.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.save_step_patches:
                save_rgb(
                    patch_after,
                    args.output / f"outer_{outer_index + 1:02d}_step_{inner_index:02d}.png",
                )
            print(
                f"outer={outer_index + 1}/{effective_rounds} "
                f"step={inner_index}/{len(timesteps)} global={len(trace)} "
                f"sample={record.sample_id}",
                flush=True,
            )
        current_patch = decode_patch(pipeline, latents).detach()
        save_rgb(current_patch, args.output / f"outer_{outer_index + 1:02d}_final.png")
        if args.mode == "validation":
            name, digest = save_training_checkpoint(args.output, current_patch, len(trace))
            checkpoint_bindings[name] = digest

    if args.mode == "validation":
        audit_candidate_trace(trace, config, args.task_variant)
    summary = {
        "status": "smoke_complete" if args.mode == "smoke" else "train_optimization_complete",
        "identity": "stateful SD3.5 unified road patch v17 validation candidate; not a formal result",
        "candidate_id": contract_payload["candidate_id"],
        "task_variant": args.task_variant,
        "training_models": list(order),
        "model_count": len(order),
        "road_eot": contract_payload["road_eot"],
        "zero_step_control": "common road crop before VAE/noise; reconstruction separately saved",
        "canonical_patch_hw": [64, 128],
        "victim_forwards_optimization_minimum": len(trace) * len(order) * 4,
        "victim_forwards_calibration_minimum": len(calibrators) * len(order),
        "benign_depth_reference_forwards_total": len(benign_depth_cache),
        "victim_forward_accounting": "minimum counts are attack loss calls; first-use clean depth forwards separately counted across calibration+optimization",
        "latent_backward_queries": len(trace) * len(order),
        "formal_test_result": False,
        "test_data_opened": False,
        "config": config.to_dict(),
        "executed_updates": len(trace),
        "seed": config.seed,
        "schedule_sha256": sha256(args.output / "schedule.json"),
        "schedule_scope": "planned 100 train rows; actual consumption is attack_trace.jsonl",
        "checkpoint_bindings": checkpoint_bindings,
        "checkpoint_selection": "fixed 25/50/75/100; final100 always evaluated, no best selection",
        "query_accounting": {
            "optimization_pre_model_forwards": len(trace) * len(order) * 2,
            "optimization_post_model_forwards": len(trace) * len(order) * 2,
            "calibration_model_forwards": len(calibrators) * len(order),
            "clean_depth_reference_forwards": len(benign_depth_cache),
            "autograd_calls": len(trace) * len(order),
            "gradient_bearing_model_eot_graphs": len(trace) * len(order) * 2,
            "transformer_forward_calls": len(trace),
            "vae_encode_calls": 1 + effective_rounds,
            "vae_decode_calls": 1 + 2 * len(trace) + effective_rounds,
            "noise_draws": effective_rounds,
            "note": "autograd calls combine two EOT graphs; not equal walltime or total queries to RGB baselines"
        },
        "normalizers": normalizers,
        "calibration_sample_ids": [r.sample_id for r in calibrators],
        "final_texture": texture_diagnostics(current_patch),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "next_gate": "frozen_validation_evaluation",
    }
    save_rgb(canonicalize_rgb(current_patch), args.output / "final_patch.png")
    save_rgb(current_patch, args.output / "raw_generator_rgb.png")
    save_rgb(
        canonicalize_rgb(current_patch),
        args.output / "final_patch_64x128.png",
    )
    save_patch(args.output / "final_patch.pt", current_patch,
               {"role": "final_attack_patch", "candidate_id": contract_payload["candidate_id"]},
               raw_generator_rgb=current_patch)
    summary["patch_bindings"] = {name: sha256(args.output / name)
                                 for name in ("zero_step_patch.pt", "final_patch.pt", "initial_vae_reconstruction.pt")}
    summary["candidate_contract_sha256"] = sha256(args.contract)
    summary["training_manifest_sha256"] = sha256(args.training_manifest)
    summary["model_manifest_sha256"] = sha256(args.model_manifest)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
