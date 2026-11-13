#!/usr/bin/env python3
"""Train one frozen normalized attack configuration."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from advrm_regularizer import AdvRMRegularizer, build_advrm_regularizer
from advrm_insertion import insert_car, load_car, load_scene, parse_road_quad
from artifacts import capture_zero_step, implementation_fingerprint, save_artifact, sha256, validate_zero_step
from carriers import (
    BaseCarrier,
    FrozenRgbCarrier,
    SceneContext,
    build_carrier,
    carrier_color_budget_loss,
    carrier_metadata,
    carrier_shape_loss,
    carrier_trainable_parameters,
    extract_road_patch,
    project_method_constraints,
)
from data import (
    SceneRecord,
    calibration_records,
    load_training_manifest,
    seeded_round_robin,
    validation_records,
    verify_source,
)
from losses import detection_suppression_loss, resize_mask
from model_adapters import build_depth_adapter, build_detection_adapter
from objectives import (
    LossBreakdown,
    attack_objective,
    carrier_regularization,
    load_method_contracts,
)
from stable_optimizers import Float64Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_冻结协议"))
from protocol_runtime import load_protocol, require_run_allowed


SOURCE_MODEL_TASKS = {
    "fasterrcnn": "detection",
    "ssd300": "detection",
    "monodepth2": "depth",
    "dpt_large": "depth",
}
THREED2FOOL_PALETTE = (
    (0, 0, 0),
    (255, 255, 255),
    (0, 18, 79),
    (5, 80, 214),
    (71, 178, 243),
    (178, 159, 211),
    (77, 58, 0),
    (211, 191, 167),
    (247, 110, 26),
    (110, 76, 16),
)


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    method_id: str
    task: str
    variant: str
    train_models: tuple[str, ...]
    depth_direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--method-contracts", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--method-source-manifest", type=Path, required=True)
    parser.add_argument("--car", type=Path, required=True)
    parser.add_argument("--car-mask", type=Path, required=True)
    parser.add_argument("--printable-palette", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("00_冻结协议"))
    parser.add_argument("--purpose", choices=("formal", "validation", "diagnostic"), default="validation")
    parser.add_argument(
        "--advrm-vgg19-checkpoint",
        type=Path,
        default=Path("/root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth"),
    )
    parser.add_argument(
        "--sd35-snapshot",
        type=Path,
        default=Path(
            "/root/autodl-tmp/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-medium/"
            "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("implementation-smoke", "pilot", "production"),
        default="production",
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument(
        "--full-budget-validation",
        action="store_true",
        help="Run all 500 frozen updates against train/validation only; never authorizes test evaluation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_candidate_training_args(args: argparse.Namespace) -> dict[str, Any]:
    """Bind CLI inputs and actual selected model files before model construction."""
    root = Path(__file__).resolve().parents[2]
    ctx = load_protocol(root, args.protocol_dir)
    full_budget_validation = bool(getattr(args, "full_budget_validation", False))
    purpose = "formal" if args.mode == "production" else args.purpose
    require_run_allowed(ctx, purpose, args.config_id)
    if args.mode == "production" and args.purpose != "formal":
        raise ValueError("production training requires explicit --purpose formal")
    if args.mode != "production" and args.purpose == "formal":
        raise ValueError("formal purpose requires production mode")
    row = ctx.row(args.config_id)
    paths = {
        "run_matrix": ctx.protocol["run_matrix_path"],
        "training_manifest": ctx.protocol["training_manifest_path"],
        "method_contracts": ctx.protocol["method_contracts_path"],
        "model_manifest": ctx.protocol["model_manifest_path"],
        "method_source_manifest": ctx.protocol["method_source_manifest_path"],
        "car": "01_公共数据集/car_00.jpg", "car_mask": "01_公共数据集/car_00_mask.jpg",
        "printable_palette": "03_方法/production/printable30values.txt",
        "output": row["artifact_output"],
    }
    for field, relative in paths.items():
        if Path(getattr(args, field)).resolve() != ctx.path(relative):
            raise ValueError(f"training CLI path differs from candidate binding: {field}")
    if full_budget_validation and (
        args.mode != "pilot" or args.purpose == "formal" or args.updates != ctx.num_steps
    ):
        raise ValueError("full-budget validation requires pilot mode, validation/diagnostic purpose and explicit frozen updates")
    expected_updates = (
        ctx.num_steps
        if args.mode == "production" or full_budget_validation
        else 1 if args.mode == "implementation-smoke" else 4
    )
    if args.updates not in (None, expected_updates):
        raise ValueError(f"candidate {args.mode} requires {expected_updates} updates")
    selected = set(row["train_models"].split("+"))
    if row["method"] == "advrm":
        selected.add("advrm_vgg19")
        vgg = next(item for item in ctx.protocol["external_inputs"] if item["id"] == "model:advrm_vgg19:0")
        if args.advrm_vgg19_checkpoint.absolute() != Path(vgg["path"]).absolute():
            raise ValueError("VGG19 path differs from candidate binding")
    if row["method"] == "ours":
        if args.sd35_snapshot.absolute() != Path(ctx.protocol["sd35_vae_snapshot_path"]).parent.absolute():
            raise ValueError("SD3.5 snapshot differs from candidate VAE binding")
    verified_models = {}
    for item in ctx.protocol["external_inputs"]:
        parts = item["id"].split(":")
        needed = (parts[0] == "model" and parts[1] in selected) or (parts[0] == "sd35_vae" and row["method"] == "ours")
        if needed:
            path = Path(item["path"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError(f"actual model input hash mismatch/missing: {item['id']}")
            verified_models[item["id"]] = item["sha256"]
    for model_id in selected:
        if model_id == "advrm_vgg19":
            continue
        if not any(key.startswith(f"model:{model_id}:") for key in verified_models):
            raise ValueError(f"selected model has no bound verified artifacts: {model_id}")
    return {
        "protocol_id": ctx.protocol["protocol_id"], "run_id": ctx.run_id,
        "protocol_sha256": sha256(ctx.protocol_path),
        "candidate_fingerprint": ctx.protocol["candidate_fingerprint"],
        "run_matrix_sha256": ctx.protocol["run_matrix_sha256"],
        "experiment_purpose": purpose,
        "verified_execution_model_files": verified_models,
    }


def load_run_config(path: Path, config_id: str) -> RunConfig:
    with path.open(encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"run matrix must contain config exactly once: {config_id}")
    row = matches[0]
    if row["test_time_optimization"].lower() != "false":
        raise ValueError("run config must forbid test-time optimization")
    models = tuple(row["train_models"].split("+"))
    if not models or any(model not in SOURCE_MODEL_TASKS for model in models):
        raise ValueError(f"run config contains an unsupported source model: {models}")
    return RunConfig(
        config_id=row["config_id"],
        method_id=row["method"],
        task=row["task"],
        variant=row["variant"],
        train_models=models,
        depth_direction=row["depth_direction"],
    )


def resolve_updates(mode: str, requested: int | None, frozen: int) -> int:
    if frozen != 500:
        raise ValueError(f"unexpected frozen production budget: {frozen}")
    if mode == "production":
        if requested not in (None, frozen):
            raise ValueError("production mode requires exactly the frozen 500 updates")
        return frozen
    default = 1 if mode == "implementation-smoke" else 4
    updates = default if requested is None else requested
    if updates < 1 or updates > frozen:
        raise ValueError("smoke/pilot updates must be in [1,500]")
    return updates


def validate_method_model_compatibility(config: RunConfig) -> None:
    if config.method_id == "upc":
        unsupported = sorted(set(config.train_models) - {"fasterrcnn"})
        if unsupported:
            raise ValueError(
                "UPC source-structured objective is not defined for: "
                + ", ".join(unsupported)
            )


def load_palette(path: Path, method_id: str, device: torch.device) -> tuple[torch.Tensor, str]:
    if method_id == "3d2fool":
        return torch.tensor(THREED2FOOL_PALETTE, device=device).float() / 255.0, (
            "3D2Fool attack_base.py color_set"
        )
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(tuple(float(value) for value in line.split(",")))
    if len(values) != 30 or any(len(value) != 3 for value in values):
        raise ValueError("APA printable palette must contain 30 RGB triplets")
    return torch.tensor(values, device=device), str(path)


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


def build_adapter(
    model_id: str, device: torch.device, model_manifest: Path | None = None
) -> object:
    if SOURCE_MODEL_TASKS[model_id] == "detection":
        adapter = build_detection_adapter(model_id, device)
    else:
        adapter = build_depth_adapter(model_id, device, model_manifest=model_manifest)
    modules = [
        value
        for value in vars(adapter).values()
        if isinstance(value, torch.nn.Module)
    ]
    if not modules or any(parameter.requires_grad for module in modules for parameter in module.parameters()):
        raise RuntimeError(f"source model is not frozen: {model_id}")
    return adapter


def model_loss(
    config: RunConfig,
    contract: dict[str, Any],
    model_id: str,
    adapter: object,
    context: SceneContext,
    image: torch.Tensor,
    support: torch.Tensor,
    benign_depth_cache: dict[tuple[str, str], torch.Tensor],
    sample_id: str,
) -> LossBreakdown:
    task = SOURCE_MODEL_TASKS[model_id]
    if task == "detection":
        if config.method_id == "upc":
            internals = adapter.predict_upc_fasterrcnn(image)
            return attack_objective(
                config.method_id,
                task,
                contract,
                target_box=context.object_box,
                detection_internals=internals,
                direction=config.depth_direction,
            )
        if config.method_id == "das":
            cam = adapter.predict_das_cam(image, context.object_box)
            return attack_objective(
                config.method_id,
                task,
                contract,
                target_box=context.object_box,
                detection_cam=cam,
                direction=config.depth_direction,
            )
        if config.method_id == "fca":
            internals = adapter.predict_fca_internals(image)
            return attack_objective(
                config.method_id,
                task,
                contract,
                target_box=context.object_box,
                detection_internals=internals,
                direction=config.depth_direction,
            )
        output = adapter.predict(image)[0]
        return attack_objective(
            config.method_id,
            task,
            contract,
            target_box=context.object_box,
            detection_output=output,
            image=image,
            object_mask=context.object_mask,
            direction=config.depth_direction,
        )

    cache_key = (sample_id, model_id)
    benign_depth = benign_depth_cache.get(cache_key)
    if benign_depth is None:
        with torch.no_grad():
            benign_depth = adapter.predict_for_training(context.benign).detach()
        benign_depth_cache[cache_key] = benign_depth
    adversarial_depth = adapter.predict_for_training(image)
    return attack_objective(
        config.method_id,
        task,
        contract,
        adversarial_depth=adversarial_depth,
        benign_depth=benign_depth,
        object_mask=context.object_mask,
        patch_support=support,
        direction=config.depth_direction,
        depth_semantics=adapter.training_output_semantics,
    )


def regularization(
    config: RunConfig,
    contract: dict[str, Any],
    carrier: BaseCarrier,
    palette: torch.Tensor,
    advrm_regularizer: AdvRMRegularizer | None = None,
) -> LossBreakdown:
    regularization_rgb = (
        carrier.regularization_rgb()
        if config.method_id == "3d2fool"
        else carrier.canonical_rgb()
    )
    advrm_losses = None
    if config.method_id == "advrm":
        if advrm_regularizer is None:
            raise ValueError("AdvRM requires an initialized VGG19 regularizer")
        advrm_losses = advrm_regularizer.losses(regularization_rgb)
    result = carrier_regularization(
        config.method_id,
        contract,
        regularization_rgb,
        carrier.initial,
        printable_palette=palette,
        shape_loss=carrier_shape_loss(carrier),
        color_budget_loss=(
            carrier_color_budget_loss(
                carrier, float(contract["regularizers"]["color_linf_radius"])
            )
            if config.method_id == "bittaa"
            else None
        ),
        advrm_losses=advrm_losses,
    )
    if config.method_id == "advrm":
        scale = float(contract["regularizers"].get("unified_adaptation_regularizer_scale", 1.0))
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError(f"invalid AdvRM unified-adaptation regularizer scale: {scale}")
        result = LossBreakdown(
            result.total * scale,
            {name: value * scale for name, value in result.components.items()},
        )
    return result


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def calibrate_normalizers(
    config: RunConfig,
    contract: dict[str, Any],
    carrier: BaseCarrier,
    adapters: dict[str, object],
    records: tuple[SceneRecord, ...],
    car: torch.Tensor,
    car_mask: torch.Tensor,
    device: torch.device,
    seed: int,
    verified: set[str],
    benign_depth_cache: dict[tuple[str, str], torch.Tensor],
) -> dict[str, float]:
    losses = {model_id: [] for model_id in config.train_models}
    for record_index, record in enumerate(records):
        context = build_context(record, car, car_mask, device, verified)
        rendered = carrier.render(
            context,
            eot=None,
            generator=make_generator(device, seed + record_index),
        )
        for model_id in config.train_models:
            result = model_loss(
                config,
                contract,
                model_id,
                adapters[model_id],
                context,
                rendered.image,
                rendered.support,
                benign_depth_cache,
                record.sample_id,
            )
            value = float(result.total.detach().abs())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite calibration loss: {model_id}")
            losses[model_id].append(value)
            del result
    normalizers = {
        model_id: max(sum(values) / len(values), 1e-6)
        for model_id, values in losses.items()
    }
    if any(not math.isfinite(value) or value <= 0 for value in normalizers.values()):
        raise RuntimeError(f"invalid ensemble normalizers: {normalizers}")
    return normalizers


def build_optimizer(
    contract: dict[str, Any], carrier: BaseCarrier
) -> torch.optim.Optimizer | None:
    specification = contract["optimizer"]
    name = specification["name"]
    parameters = carrier_trainable_parameters(carrier)
    if name == "bim":
        return None
    kwargs = {
        "lr": float(specification["learning_rate"]),
        "weight_decay": float(specification.get("weight_decay", 0.0)),
    }
    if name == "adam":
        if "betas" in specification:
            kwargs["betas"] = tuple(specification["betas"])
        return Float64Adam(parameters, **kwargs)
    if name == "adam_amsgrad":
        return Float64Adam(parameters, amsgrad=True, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(parameters, **kwargs)
    raise ValueError(f"unsupported optimizer: {name}")



def model_objective_weight(
    config: RunConfig, contract: dict[str, Any], model_id: str
) -> float:
    if config.method_id != "bittaa":
        return 1.0
    aggregation = contract.get("task_loss_aggregation", {})
    key = "detector_weight" if SOURCE_MODEL_TASKS[model_id] == "detection" else "depth_weight"
    value = float(aggregation[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid BiTAA task weight: {key}={value}")
    return value

def validate_gradients(parameters: list[torch.nn.Parameter]) -> float:
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("carrier has a missing gradient")
    selected = [gradient for gradient in gradients if gradient is not None]
    if any(not torch.isfinite(gradient).all() for gradient in selected):
        raise RuntimeError("carrier gradient is non-finite")
    norm = math.sqrt(
        sum(float(gradient.detach().double().square().sum()) for gradient in selected)
    )
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("carrier gradient is zero or non-finite")
    return norm


def optimizer_update(
    carrier: BaseCarrier,
    contract: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> float:
    parameters = carrier_trainable_parameters(carrier)
    norm = validate_gradients(parameters)
    if optimizer is None:
        learning_rate = float(contract["optimizer"]["learning_rate"])
        carrier.bim_step_(learning_rate)
    else:
        optimizer.step()
    return norm


def _flatten_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.detach().float().reshape(-1)


def unit_gradient_update(
    carrier: BaseCarrier,
    model_gradients: list[torch.Tensor],
    model_tasks: list[str],
    progress: torch.Tensor,
    regularizer: torch.Tensor,
    learning_rate: float,
) -> tuple[float, dict[str, Any]]:
    parameters = carrier_trainable_parameters(carrier)
    if len(parameters) != 1:
        raise RuntimeError("unit-gradient method requires one trainable carrier tensor")
    flattened = torch.stack([_flatten_gradient(gradient) for gradient in model_gradients])
    norms = torch.linalg.vector_norm(flattened, dim=1, dtype=torch.float64)
    if not torch.isfinite(flattened).all() or not torch.isfinite(norms).all():
        raise RuntimeError("unit-gradient input is non-finite")
    active = norms.ne(0.0)
    if not active.any():
        raise RuntimeError("all model gradients are zero")
    if ((norms <= 1e-8) & active).any():
        raise RuntimeError("nonzero model gradient is too small to normalize reliably")
    safe_norms = torch.where(active, norms, torch.ones_like(norms))
    unit = (flattened.double() / safe_norms[:, None]).to(dtype=flattened.dtype)
    cosine = (unit @ unit.T).clamp(-1.0, 1.0)
    negative = torch.relu(-cosine)
    negative.fill_diagonal_(0.0)
    denominator = max(len(model_gradients) - 1, 1)
    conflict = negative.sum(dim=1) / denominator
    raw_weights = (1.0 - progress) * (1.0 - conflict) * active.to(progress)
    if raw_weights.sum() <= 1e-8:
        weights = active.to(raw_weights) / active.sum()
        fallback = True
    else:
        weights = raw_weights / raw_weights.sum()
        fallback = False

    task_gradients = {}
    for task in ("detection", "depth"):
        indices = [index for index, value in enumerate(model_tasks) if value == task]
        if indices:
            task_gradients[task] = sum(weights[index] * unit[index] for index in indices)
    if set(task_gradients) != {"detection", "depth"}:
        raise RuntimeError("joint method requires detection and depth gradients")
    detection = task_gradients["detection"]
    depth = task_gradients["depth"]
    dot = torch.dot(detection, depth)
    task_cosine = dot / (
        torch.linalg.vector_norm(detection) * torch.linalg.vector_norm(depth)
    ).clamp_min(1e-8)
    projected = bool(dot < 0)
    if projected:
        detection_original = detection
        detection = detection - dot / depth.square().sum().clamp_min(1e-8) * depth
        depth = depth - dot / detection_original.square().sum().clamp_min(1e-8) * detection_original
    joint = detection + depth
    joint_rms = joint.square().mean().sqrt()
    if not torch.isfinite(joint_rms) or joint_rms <= 1e-8:
        raise RuntimeError("combined unit gradient is non-finite or zero")
    joint = joint / joint_rms

    parameter = parameters[0]
    regularizer_gradient = torch.autograd.grad(
        regularizer, parameter, allow_unused=True
    )[0]
    if regularizer_gradient is None:
        regularizer_gradient = torch.zeros_like(parameter)
    update = joint.reshape_as(parameter).to(parameter) + regularizer_gradient
    if not torch.isfinite(update).all():
        raise RuntimeError("unit-gradient update is non-finite")
    with torch.no_grad():
        parameter.add_(update, alpha=-learning_rate)
    return float(torch.linalg.vector_norm(update).detach()), {
        "model_gradient_norms": [float(value) for value in norms],
        "zero_model_gradient_indices": torch.nonzero(~active).flatten().tolist(),
        "active_model_gradient_count": int(active.sum()),
        "model_weights": [float(value) for value in weights],
        "gradient_conflict": [float(value) for value in conflict],
        "task_cosine": float(task_cosine),
        "task_projection_applied": projected,
        "weight_fallback": fallback,
    }


def scheduled_validation_steps(updates: int, interval: int) -> tuple[int, ...]:
    if updates < 1 or interval < 1:
        raise ValueError("updates and validation interval must be positive")
    return tuple(range(interval, updates + 1, interval))


@torch.no_grad()
def validation_snapshot(
    config: RunConfig,
    contract: dict[str, Any],
    carrier: BaseCarrier,
    adapters: dict[str, object],
    records: tuple[SceneRecord, ...],
    car: torch.Tensor,
    car_mask: torch.Tensor,
    device: torch.device,
    seed: int,
    verified: set[str],
    benign_depth_cache: dict[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {
        model_id: {"kitti": [], "nuscenes": []}
        for model_id in config.train_models
    }
    signed_depth: dict[str, dict[str, list[float]]] = {
        model_id: {"kitti": [], "nuscenes": []}
        for model_id in config.train_models
        if SOURCE_MODEL_TASKS[model_id] == "depth"
    }
    for index, record in enumerate(records):
        context = build_context(record, car, car_mask, device, verified)
        rendered = carrier.render(
            context,
            eot=None,
            generator=make_generator(device, seed + index),
        )
        for model_id in config.train_models:
            adapter = adapters[model_id]
            if SOURCE_MODEL_TASKS[model_id] == "detection":
                output = adapter.predict(rendered.image)[0]
                metric = detection_suppression_loss(output, context.object_box)
                values[model_id][record.dataset].append(float(metric))
                continue
            cache_key = (record.sample_id, model_id)
            benign_depth = benign_depth_cache.get(cache_key)
            if benign_depth is None:
                benign_depth = adapter.predict_for_training(context.benign).detach()
                benign_depth_cache[cache_key] = benign_depth
            adversarial_depth = adapter.predict_for_training(rendered.image)
            from losses import semantic_far_change
            change = semantic_far_change(
                adversarial_depth, benign_depth, adapter.training_output_semantics
            )
            roi = resize_mask(context.object_mask, change)
            values[model_id][record.dataset].append(float(change[roi].abs().mean()))
            signed_depth[model_id][record.dataset].append(float(change[roi].mean()))

    model_metrics = {}
    for model_id in config.train_models:
        domains = {}
        for dataset in ("kitti", "nuscenes"):
            domain_values = values[model_id][dataset]
            domains[dataset] = {
                "scene_count": len(domain_values),
                "monitor_mean": sum(domain_values) / len(domain_values),
            }
            if model_id in signed_depth:
                signed = signed_depth[model_id][dataset]
                domains[dataset]["signed_semantic_far_change_mean"] = sum(signed) / len(signed)
        model_metrics[model_id] = {
            "task": SOURCE_MODEL_TASKS[model_id],
            "monitor": (
                "target_vehicle_suppression"
                if SOURCE_MODEL_TASKS[model_id] == "detection"
                else "mean_abs_semantic_depth_change_clean_ref"
            ),
            "domains": domains,
        }
    return {
        "scene_count": len(records),
        "selection_used": False,
        "eot": "disabled_for_validation",
        "models": model_metrics,
    }


def save_checkpoint(
    path: Path,
    *,
    config: RunConfig,
    implementation_sha256: str,
    updates: int,
    step: int,
    carrier: BaseCarrier,
    optimizer: torch.optim.Optimizer | None,
    progress: torch.Tensor,
    normalizers: dict[str, float],
    trace: list[dict[str, Any]],
    zero_step: dict[str, Any],
    protocol_binding: dict[str, Any] | None = None,
) -> None:
    validate_zero_step(zero_step)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in carrier.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "schema_version": 2,
        "config_id": config.config_id,
        "implementation_sha256": implementation_sha256,
        "updates": updates,
        "step": step,
        "carrier_state": state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "progress": progress.detach().cpu(),
        "normalizers": normalizers,
        "trace": trace,
        "zero_step": zero_step,
        "protocol_binding": protocol_binding,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    *,
    config: RunConfig,
    implementation_sha256: str,
    updates: int,
    carrier: BaseCarrier,
    optimizer: torch.optim.Optimizer | None,
    normalizers: dict[str, float],
    device: torch.device,
    protocol_binding: dict[str, Any] | None = None,
) -> tuple[int, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != 2 or payload.get("config_id") != config.config_id:
        raise ValueError("checkpoint identity mismatch")
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError("checkpoint implementation hash mismatch")
    if payload.get("protocol_binding") != protocol_binding:
        raise ValueError("checkpoint candidate protocol binding mismatch")
    if payload.get("updates") != updates or payload.get("normalizers") != normalizers:
        raise ValueError("checkpoint budget or normalizers mismatch")
    parameters = {
        name: parameter
        for name, parameter in carrier.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(payload["carrier_state"]):
        raise ValueError("checkpoint carrier state names differ")
    zero_step = payload.get("zero_step")
    validate_zero_step(zero_step)
    if set(parameters) != set(zero_step["trainable_state"]):
        raise ValueError("checkpoint zero-step state names differ")
    buffers = dict(carrier.named_buffers(recurse=False))
    if set(buffers) != set(zero_step["reference_buffers"]):
        raise ValueError("checkpoint reference buffer names differ")
    with torch.no_grad():
        for name, buffer in buffers.items():
            reference = zero_step["reference_buffers"][name]
            if reference.shape != buffer.shape:
                raise ValueError(f"checkpoint reference buffer shape differs: {name}")
            buffer.copy_(reference.to(buffer))
        for name, parameter in parameters.items():
            parameter.copy_(payload["carrier_state"][name].to(parameter))
    optimizer_state = payload.get("optimizer_state")
    if optimizer is None and optimizer_state is not None:
        raise ValueError("checkpoint unexpectedly contains optimizer state")
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError("checkpoint is missing optimizer state")
        optimizer.load_state_dict(optimizer_state)
    step = int(payload["step"])
    trace = payload["trace"]
    if step < 0 or step > updates or len(trace) != step:
        raise ValueError("checkpoint step/trace length mismatch")
    progress = payload["progress"].to(device=device, dtype=torch.float32)
    if progress.shape != (len(config.train_models),):
        raise ValueError("checkpoint progress shape mismatch")
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
    return step, progress, trace, zero_step


@torch.no_grad()
def evaluate_losses_without_gradients(
    config: RunConfig,
    contract: dict[str, Any],
    carrier: BaseCarrier,
    adapters: dict[str, object],
    context: SceneContext,
    sample_id: str,
    device: torch.device,
    seed: int,
    benign_depth_cache: dict[tuple[str, str], torch.Tensor],
) -> list[float]:
    rendered = carrier.render(
        context,
        eot=contract.get("eot"),
        generator=make_generator(device, seed),
    )
    values = []
    for model_id in config.train_models:
        result = model_loss(
            config,
            contract,
            model_id,
            adapters[model_id],
            context,
            rendered.image,
            rendered.support,
            benign_depth_cache,
            sample_id,
        )
        values.append(float(result.total.detach()))
    return values


def main() -> None:
    args = parse_args()
    protocol_binding = validate_candidate_training_args(args)
    checkpoint_path = args.output / ".training_checkpoint.pt"
    existing = set(args.output.iterdir()) if args.output.exists() else set()
    if existing and not (args.resume and existing == {checkpoint_path}):
        raise FileExistsError(f"refusing to overwrite non-checkpoint output: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = load_run_config(args.run_matrix, args.config_id)
    validate_method_model_compatibility(config)
    contract_payload, contracts = load_method_contracts(args.method_contracts)
    contract = contracts[config.method_id]
    if contract["task"] != config.task:
        raise ValueError("run matrix and method contract tasks differ")
    updates = resolve_updates(
        args.mode, args.updates, int(contract_payload["normalized_budget"]["updates"])
    )
    seed = int(contract_payload["normalized_budget"]["seed"])
    validation_interval = int(contract_payload["normalized_budget"]["validation_interval"])
    implementation_sha256, implementation_files = implementation_fingerprint(
        Path(__file__).resolve().parent
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    records = load_training_manifest(args.training_manifest)
    calibrators = calibration_records(records)
    validators = validation_records(records)
    schedule = seeded_round_robin(records, updates, seed)
    car, car_mask = load_car(args.car, args.car_mask, device)
    verified: set[str] = set()
    initialization_started = time.monotonic()
    if contract["carrier"] in {
        "road_patch",
        "apa_rgb_road_patch",
        "saam_rgb_road_patch",
        "advrm_rgb_road_patch",
        "sd35_vae_latent_road_patch",
    }:
        initial_context = build_context(calibrators[0], car, car_mask, device, verified)
        initial_patch = extract_road_patch(initial_context.background, initial_context.road_quad)
    else:
        initial_patch = torch.empty((1, 3, 1, 1), device=device)
    carrier = build_carrier(
        contract["carrier"],
        car=car,
        car_mask=car_mask,
        initial_patch=initial_patch,
        device=device,
        sd35_snapshot=args.sd35_snapshot,
    )
    if carrier.kind == "vehicle":
        initialization = dict(carrier.initialization_report)
    else:
        initialization = {
            "source": "fixed_first_training_calibration_scene_road_crop",
            "sample_id": calibrators[0].sample_id,
            "source_sha256": calibrators[0].source_sha256,
            "fit_steps": 0,
            "reference": "actual_zero_step_canonical_rgb",
            "victim_models_used": False,
        }
    initialization.update({
        "car_sha256": sha256(args.car),
        "car_mask_sha256": sha256(args.car_mask),
        "elapsed_seconds": time.monotonic() - initialization_started,
        "included_in_attack_update_budget": False,
    })
    zero_step = capture_zero_step(carrier, initialization)
    advrm_regularizer = (
        build_advrm_regularizer(carrier.initial, args.advrm_vgg19_checkpoint)
        if config.method_id == "advrm"
        else None
    )
    auxiliary_model_hashes = (
        {"advrm_vgg19": sha256(args.advrm_vgg19_checkpoint)}
        if config.method_id == "advrm"
        else {}
    )
    palette, palette_source = load_palette(args.printable_palette, config.method_id, device)
    adapters = {
        model_id: build_adapter(model_id, device, args.model_manifest)
        for model_id in config.train_models
    }
    benign_depth_cache: dict[tuple[str, str], torch.Tensor] = {}
    normalizers = (
        calibrate_normalizers(
            config,
            contract,
            carrier,
            adapters,
            calibrators,
            car,
            car_mask,
            device,
            seed,
            verified,
            benign_depth_cache,
        )
        if len(config.train_models) > 1
        else {config.train_models[0]: 1.0}
    )
    optimizer = build_optimizer(contract, carrier)
    progress = torch.zeros(len(config.train_models), device=device)
    trace = []
    start_step = 0
    if args.resume and checkpoint_path.is_file():
        start_step, progress, trace, zero_step = restore_checkpoint(
            checkpoint_path,
            config=config,
            implementation_sha256=implementation_sha256,
            updates=updates,
            carrier=carrier,
            optimizer=optimizer,
            normalizers=normalizers,
            device=device,
            protocol_binding=protocol_binding,
        )
        print(f"RESUME step={start_step}/{updates} checkpoint={checkpoint_path}", flush=True)
    long_validation = args.mode == "production" or args.full_budget_validation
    if long_validation and "validation" not in zero_step:
        validation_started = time.monotonic()
        zero_step["validation"] = validation_snapshot(
            config, contract,
            FrozenRgbCarrier(zero_step["canonical_rgb"].to(device), carrier.kind),
            adapters, validators, car, car_mask, device, seed + 1_000_000,
            verified, benign_depth_cache,
        )
        zero_step["validation"]["elapsed_seconds"] = time.monotonic() - validation_started
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step, record in enumerate(schedule[start_step:], start=start_step + 1):
        if record.split != "train":
            raise ValueError("attack optimization may only consume frozen train records")
        step_started = time.monotonic()
        context = build_context(record, car, car_mask, device, verified)
        render_seed = seed + step
        rendered = carrier.render(
            context,
            eot=contract.get("eot"),
            generator=make_generator(device, render_seed),
        )
        raw_losses = []
        component_values = {}

        if config.method_id == "ours":
            gradients = []
            parameters = carrier_trainable_parameters(carrier)
            if len(parameters) != 1:
                raise RuntimeError("ours requires exactly one latent tensor")
            for index, model_id in enumerate(config.train_models):
                result = model_loss(
                    config,
                    contract,
                    model_id,
                    adapters[model_id],
                    context,
                    rendered.image,
                    rendered.support,
                    benign_depth_cache,
                    record.sample_id,
                )
                loss = result.total / normalizers[model_id]
                gradient = torch.autograd.grad(
                    loss,
                    parameters[0],
                    retain_graph=index < len(config.train_models) - 1,
                )[0]
                gradients.append(gradient)
                raw_losses.append(float(result.total.detach()))
                component_values[model_id] = result.scalars()
            regularizer = (
                regularization(config, contract, carrier, palette, advrm_regularizer)
                if config.method_id == "advrm"
                else regularization(config, contract, carrier, palette)
            )
            gradient_norm, control = unit_gradient_update(
                carrier,
                gradients,
                [SOURCE_MODEL_TASKS[model_id] for model_id in config.train_models],
                progress,
                regularizer.total,
                float(contract["optimizer"]["learning_rate"]),
            )
            project_method_constraints(carrier, config.method_id, contract)
            post_losses = evaluate_losses_without_gradients(
                config,
                contract,
                carrier,
                adapters,
                context,
                record.sample_id,
                device,
                render_seed,
                benign_depth_cache,
            )
            before = torch.tensor(raw_losses, device=device).abs().clamp_min(1e-8)
            after = torch.tensor(post_losses, device=device)
            progress = ((before - after) / before).clamp(0.0, 1.0)
            total_value = sum(
                (loss / normalizers[model_id]) * model_objective_weight(config, contract, model_id)
                for loss, model_id in zip(raw_losses, config.train_models)
            ) / len(raw_losses) + float(regularizer.total.detach())
        else:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                for parameter in carrier_trainable_parameters(carrier):
                    parameter.grad = None
            for index, model_id in enumerate(config.train_models):
                result = model_loss(
                    config,
                    contract,
                    model_id,
                    adapters[model_id],
                    context,
                    rendered.image,
                    rendered.support,
                    benign_depth_cache,
                    record.sample_id,
                )
                normalized = (result.total / normalizers[model_id]) * model_objective_weight(config, contract, model_id)
                (normalized / len(config.train_models)).backward(
                    retain_graph=index < len(config.train_models) - 1
                )
                raw_losses.append(float(result.total.detach()))
                component_values[model_id] = result.scalars()
            regularizer = (
                regularization(config, contract, carrier, palette, advrm_regularizer)
                if config.method_id == "advrm"
                else regularization(config, contract, carrier, palette)
            )
            regularizer.total.backward()
            gradient_norm = optimizer_update(carrier, contract, optimizer)
            project_method_constraints(carrier, config.method_id, contract)
            control = {}
            total_value = sum(
                (loss / normalizers[model_id]) * model_objective_weight(config, contract, model_id)
                for loss, model_id in zip(raw_losses, config.train_models)
            ) / len(raw_losses) + float(regularizer.total.detach())

        canonical = carrier.canonical_rgb()
        if not torch.isfinite(canonical).all() or canonical.min() < 0 or canonical.max() > 1:
            raise RuntimeError("carrier update produced invalid canonical RGB")
        trace_record = {
            "step": step,
            "sample_id": record.sample_id,
            "dataset": record.dataset,
            "split": record.split,
            "source_sha256": record.source_sha256,
            "raw_model_losses": dict(zip(config.train_models, raw_losses)),
            "model_components": component_values,
            "regularizer_components": regularizer.scalars(),
            "total_loss": total_value,
            "gradient_norm": gradient_norm,
            "elapsed_seconds": time.monotonic() - step_started,
            "control": control,
        }
        if long_validation and step % validation_interval == 0:
            validation_started = time.monotonic()
            trace_record["validation"] = validation_snapshot(
                config,
                contract,
                carrier,
                adapters,
                validators,
                car,
                car_mask,
                device,
                seed + 1_000_000 + step,
                verified,
                benign_depth_cache,
            )
            trace_record["validation"]["elapsed_seconds"] = (
                time.monotonic() - validation_started
            )
        trace.append(trace_record)
        print(
            f"STEP {step}/{updates} sample={record.sample_id} "
            f"loss={total_value:.8g} grad={gradient_norm:.8g} "
            f"seconds={trace_record['elapsed_seconds']:.3f}",
            flush=True,
        )
        if not long_validation or step % validation_interval == 0:
            save_checkpoint(
                checkpoint_path,
                config=config,
                implementation_sha256=implementation_sha256,
                updates=updates,
                step=step,
                carrier=carrier,
                optimizer=optimizer,
                progress=progress,
                normalizers=normalizers,
                trace=trace,
                zero_step=zero_step,
                protocol_binding=protocol_binding,
            )

    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    model_hashes = {
        model["id"]: [artifact["sha256"] for artifact in model["artifacts"]]
        for model in model_manifest["models"]
        if model["id"] in config.train_models
    }
    package_versions = {}
    for package in ("torch", "torchvision", "transformers", "diffusers", "timm"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    metadata = {
        **protocol_binding,
        "config_id": config.config_id,
        "method_id": config.method_id,
        "variant": config.variant,
        "carrier_id": contract["carrier"],
        "carrier_kind": carrier.kind,
        "train_models": list(config.train_models),
        "model_checkpoint_sha256": model_hashes,
        "auxiliary_model_checkpoint_sha256": auxiliary_model_hashes,
        "seed": seed,
        "updates": updates,
        "run_mode": args.mode,
        "full_budget_validation": args.full_budget_validation,
        "depth_direction": config.depth_direction,
        "test_time_optimization": False,
        "dataset_manifest_sha256": sha256(args.training_manifest),
        "method_contracts_sha256": sha256(args.method_contracts),
        "model_manifest_sha256": sha256(args.model_manifest),
        "method_source_manifest_sha256": sha256(args.method_source_manifest),
        "implementation_sha256": implementation_sha256,
        "implementation_files_sha256": implementation_files,
        "source_anchor": contract["source_anchor"],
        "implementation_label": contract["implementation_label"],
        "limitations": contract["limitations"],
        "normalizer_calibration_samples": [record.sample_id for record in calibrators],
        "loss_normalizers": normalizers,
        "numerical_policy": {
            "carrier_dtype": "float32",
            "gradient_norm_dtype": "float64",
            "adam_moment_dtype": "float64",
            "gradient_clipping": False,
            "adabins_attention_backend": "pytorch_math",
            "finite_zero_model_gradient": "zero_contribution_renormalize_active_weights",
            "all_model_gradients_zero": "reject",
        },
        "validation_policy": {
            "interval": validation_interval,
            "scene_count": len(validators),
            "selection_used": False,
            "completed_steps": [
                record["step"] for record in trace if "validation" in record
            ],
        },
        "zero_step_validation": zero_step.get("validation"),
        "printable_palette_source": palette_source,
        "carrier": carrier_metadata(carrier),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "packages": package_versions,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    complete = save_artifact(
        args.output,
        carrier,
        carrier.canonical_rgb(),
        metadata,
        trace,
        zero_step=zero_step,
    )
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(complete, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
