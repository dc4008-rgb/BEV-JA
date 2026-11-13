#!/usr/bin/env python3
"""SOURCE-ONLY derived budget1/3/5 SD3.5 generation/attack engine.

New budget whitelist/identity only; former abd6 and original SCI stay frozen.

Original SCI1940 sample_training.py numerical loop is copied into this NEW
library, with only a new identity plus explicit real-generation guards and
scheduler begin-index instrumentation. Original files/results remain unchanged.
No CLI, SSH, victim/model construction, weight download or process/Gate action
is implemented here. A separately reviewed caller must bind actual sources,
weights, two TRAIN inputs, resource/lifecycle admission and new output roots.

Four-source losses, two VJPs, original posterior.mode encode, FP32 decoder,
noise, EOT1+C1, periodic weighting/EMA/conflict/norm and update math are retained.
The older clip.denoise_with_evidence field is deliberately DISCARDED when the
new TrainingKernels is built. Real Transformer CFG and scheduler.step execute in
the new explicit callback below. This is not an old-return/prefix reuse path.

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
             "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
from benchmark_mode import total_views as benchmark_total_views, current_mode as benchmark_current_mode, policy as benchmark_policy
from contextlib import nullcontext
from benchmark_mode import uses_prefetch as benchmark_uses_prefetch
IDENTITY = "bevja_conditional_two_clip_budget135_training_v1"
ORIGINAL_SCI_MANIFEST_SHA256 = "1940406257965a181371af89634a7378433392147719a4b16ae28d7f33c9081d"
ORIGINAL_TRAINING_MEMBER_SHA256 = "7a4fc20f362ccf6934050e57f8e745afe207085b1b9b676db0dcc1b891f1cea7"
CONDITIONAL_KERNEL_SHA256 = "946c415b14c8119e70b389f3ec3c03444e51dd854742878cd0cf191495c263cb"
DETECTOR_MODELS = ("fasterrcnn", "ssd300")
DEPTH_MODELS = ("monodepth2", "dpt_large")
SEED = 20260919
CONFIG = {"num_inference_steps": 100, "strength": .25, "latent_lr": .05,
          "step_feedback_mode": "latent_chain", "guidance_scale": 4.5,
          "seed": SEED, "prompt": "top-down aerial view of clean asphalt road surface",
          "negative_prompt": "vehicle, car, obstacle"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def reset_sample_rng(seed=SEED):
    """Reset scientific RNG for each case, independently of preceding cases."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_dual_backward_runtime():
    """Allow two first-order VJPs through frozen AOT decoder/depth graphs.

    The accepted single-VJP runner did not need graph retention. AOT buffer
    donation is incompatible with retain_graph, so this new runtime profile
    explicitly disables it rather than editing the accepted helper source.
    """
    import torch._functorch.config as functorch_config
    if not hasattr(functorch_config, "donated_buffer"):
        raise RuntimeError("dual-VJP runtime requires the donated_buffer configuration flag")
    functorch_config.donated_buffer = False
    if functorch_config.donated_buffer is not False:
        raise RuntimeError("AOT donated buffers remain enabled")
    return {"torch_functorch_donated_buffer": False,
            "reason": "two_first_order_task_vjps_share_decoder_graph",
            "scientific_loss_or_model_changed": False}


def configure_dual_runtime(torch_module=None):
    """Batch-facing name; configure before constructing compiled victim graphs."""
    return configure_dual_backward_runtime()


def planned_budget(updates):
    if type(updates) is not int or updates not in (25, 75, 125):
        raise ValueError("new budget135 updates must be exactly25/75/125")
    outer = updates // 25
    return {"detector_forwards": 2 * benchmark_total_views() * updates + 2, "autograd_calls": 2 * updates,
            "transformer_forward_calls": updates, "vae_encode_calls": 1 + outer,
            "vae_decode_calls": 1 + 2 * updates + outer, "noise_draws": outer,
            "depth_clean_forwards": 2 * benchmark_total_views() * updates, "depth_adv_forwards": 2 * benchmark_total_views() * updates}


@dataclass(frozen=True)
class TrainingKernels:
    denoise: Callable
    detector_losses: Callable
    foa_weights: Callable
    weighted_loss: Callable
    clipped_update: Callable
    sample_views: Callable
    fingerprint: Callable
    rng_snapshot: Callable
    scheduler_snapshot: Callable
    CallLedger: Callable
    assert_determinism: Callable = lambda: None


class SampleRecorder:
    """Small phase hashes plus outer-boundary states, with no resume claim."""
    def __init__(self, output, kernels, metadata):
        self.output = Path(output)
        self.directory = self.output / "state_evidence"
        self.directory.mkdir(exist_ok=False)
        self.kernels, self.metadata = kernels, metadata
        self.entries, self.checkpoints, self.math_checks = [], [], []

    def record(self, phase, step, state, *, retain=False, retain_kind="initial_or_final_state"):
        import torch
        before = self.kernels.fingerprint(self.kernels.rng_snapshot())
        entry = {"phase": phase, "global_step": step,
                 "state_fingerprint": self.kernels.fingerprint(state)}
        if retain:
            if retain_kind not in ("initial_or_final_state", "fixed_update_math_probe"):
                raise ValueError("unknown fixed evidence retention kind")
            index = len(self.checkpoints) + len(self.math_checks)
            path = self.directory / f"{index:03d}_{phase}_{step:04d}.pt"
            from evidence import safe_snapshot
            with path.open("xb") as stream:
                torch.save({"metadata": self.metadata, "phase": phase,
                            "global_step": step, "state": safe_snapshot(state)}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            entry.update(path=str(path.relative_to(self.output)), sha256=sha(path),
                         retention_kind=retain_kind,
                         bytes=path.stat().st_size)
            collection = self.math_checks if retain_kind == "fixed_update_math_probe" else self.checkpoints
            collection.append(dict(entry))
        after = self.kernels.fingerprint(self.kernels.rng_snapshot())
        if before != after:
            raise RuntimeError("evidence serialization consumed scientific RNG")
        entry["recording_preserved_rng"] = True
        self.entries.append(entry)
        return entry["state_fingerprint"]

    def finish(self, updates):
        path = self.directory / "MANIFEST.json"
        write_json(path, {"schema": "sample-specific-state-evidence-v1",
            "identity": IDENTITY, **self.metadata, "status": "complete",
            "executed_updates": updates, "entries": self.entries,
            "outer_boundary_checkpoints": self.checkpoints,
            "fixed_update_math_checks": self.math_checks,
            "fixed_update_math_steps": [step for step in (1, 25, 500) if step <= updates],
            "resume_enabled": False,
            "evidence_policy": "all phase hashes; initialization and final outer states; fixed 1/25/500 update arithmetic tensors; no tested resume"})
        return sha(path)


def _binding(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def run_sample_optimization(*, parent, config, output, pipeline, prompt_pack,
                            initial_patch, initial_small, sample, context, adapters,
                            depth_objective, device, dtype, eot_config, kernels,
                            plan_sha256, source_bindings, scientific_config):
    """Optimize only this sample with freshly reset latent, scheduler, per-outer periodic controller and RNG.

    ``context`` is this sample's fixed road/car geometry. The scene never rotates
    during optimization. No previous case's patch or latent is accepted. Each
    denoising output is detached before differentiating the current joint loss.
    """
    import torch
    backward_runtime = configure_dual_backward_runtime()
    output = Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError("core requires a newly created empty output directory")
    if sample["source_split"] not in ("train", "validation"):
        raise ValueError("only recorded train/validation attack-input cases are allowed; test forbidden")
    if {key: getattr(config, key) for key in CONFIG} != CONFIG:
        raise ValueError("fixed SD3.5 scientific configuration changed")
    from sample_contract import CONTROLLER_CONFIG, scientific_config_for
    from detection_control import MatchingConfig, target_matched_confidence, detection_goal_deficits
    from periodic_controller import PeriodicConfig, PeriodicController
    from task_gradient_balance import combine_task_gradients
    switches = scientific_config["switches"]
    if scientific_config != scientific_config_for(scientific_config["ablation_variant"], config.adversarial_updates, switches=switches):
        raise ValueError("joint sample-specific scientific configuration changed")
    if eot_config != scientific_config["eot"]:
        raise ValueError("EOT configuration differs from selected plan")
    updates = config.adversarial_updates
    budget = planned_budget(updates)
    if config.outer_rounds != updates // 25 or config.active_steps_per_round != 25:
        raise ValueError("25 active diffusion steps per outer round required")
    if depth_objective is None:
        raise ValueError("joint candidate requires the original two-model depth objective")
    if tuple(initial_small.shape) != (1, 3, 64, 128) or tuple(initial_patch.shape) != (1, 3, 512, 512):
        raise ValueError("own-road initialization must be canonical64x128 then raw512x512")
    _require_conditional_run(kernels, pipeline, prompt_pack, config, sample, output)
    reset_sample_rng(config.seed)
    # Rebuild the scheduler from the same frozen config, even if a CPU caller
    # deliberately reuses a pipeline object between isolated cases.
    pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config)
    if benchmark_current_mode() == "C1_PREFETCH_SERIAL" and updates != 25:
        raise ValueError("prefetch-serial shadow is diagnostic25 only")
    if benchmark_uses_prefetch():
        from benchmark_execution import ConcurrentLedger
        ledger = ConcurrentLedger(kernels.CallLedger, budget)
    else:
        ledger = kernels.CallLedger(budget)
    benchmark_steps = []
    metadata = {"identity": IDENTITY, "sample_id": sample["sample_id"],
        "case_id": sample["case_id"], "dataset": sample["dataset"],
        "source_split": sample["source_split"], "source_sha256": sample["source_sha256"],
        "plan_sha256": plan_sha256, "source_bindings": source_bindings,
        "input_role": "sample_specific_attack_input", "test_data_opened": False}
    metadata["dual_backward_runtime"] = backward_runtime
    recorder = SampleRecorder(output, kernels, metadata)
    parent.save_patch(output / "zero_step_patch.pt", initial_small,
        {**metadata, "role": "sample_specific_initial_road_crop", "updates": 0})
    parent.save_patch(output / "input_raw512.pt", initial_patch,
        {**metadata, "role": "sample_specific_raw512_initialization", "updates": 0},
        raw_generator_rgb=initial_patch)
    trace_path = output / "attack_trace.jsonl"
    trace_path.open("x").close()
    trace = []
    started = time.monotonic()
    controller_config = MatchingConfig()
    periodic_config = PeriodicConfig(**{key: CONTROLLER_CONFIG[key] for key in
        ("model_mu", "task_mu", "model_temperature", "task_temperature", "ema_decay", "ratio_delta", "ratio_min", "ratio_max")},
        period=switches["period"], model_adaptive=switches["model_adaptive"],
        task_adaptive=switches["task_adaptive"], ema=switches["ema"])
    periodic = PeriodicController(periodic_config)
    cycles = []
    cycle_path = output / "cycle_trace.jsonl"
    cycle_path.open("x").close()
    if scientific_config["controller_config"] != CONTROLLER_CONFIG:
        raise ValueError("frozen periodic configuration differs")
    # One detached identity reference per source detector, once per sample.
    # Keeping native unmatched/low-clean cases is an explicit neutral-deficit
    # extension. No scene is discarded, and no model is considered successful
    # merely because the benign vehicle was not confidently detected.
    clean_before_rng = kernels.fingerprint(kernels.rng_snapshot())
    clean_detector_reference = {}
    clean_confidences = []
    with torch.no_grad():
        for name in DETECTOR_MODELS:
            native = ledger.call("detector_forwards", adapters[name].predict, context.benign.detach())[0]
            confidence, match_details = target_matched_confidence(native, context.object_box, controller_config)
            clean_confidences.append(confidence)
            clean_detector_reference[name] = {"boxes": native["boxes"].detach(),
                "labels": native["labels"].detach(), "scores": native["scores"].detach(),
                "matched_confidence": confidence, "matching": match_details}
    clean_scores = torch.stack(clean_confidences).detach()
    if clean_before_rng != kernels.fingerprint(kernels.rng_snapshot()):
        raise RuntimeError("frozen clean detector reference consumed scientific RNG")
    recorder.record("initialization", 0, {"initial_patch": initial_patch,
        "initial_small": initial_small, "config": config.to_dict(),
        "controller_config": CONTROLLER_CONFIG,
        "target_box": list(context.object_box),
        "clean_detector_reference": clean_detector_reference, "sample_id": sample["sample_id"],
        "rng": kernels.rng_snapshot(), "scheduler": kernels.scheduler_snapshot(pipeline.scheduler)}, retain=True)
    with torch.no_grad():
        initial_latent = ledger.call("vae_encode_calls", parent.encode_patch, pipeline, initial_patch, dtype)
        reconstructed = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, initial_latent)
    parent.save_patch(output / "initial_vae_reconstruction.pt", reconstructed,
        {**metadata, "role": "diagnostic_not_zero_control", "updates": 0}, raw_generator_rgb=reconstructed)
    current_patch = initial_patch
    for outer_index in range(config.outer_rounds):
        periodic.reset_outer(outer_index)
        latent_start = ledger.call("vae_encode_calls", parent.encode_patch, pipeline, current_patch, dtype)
        pipeline.scheduler.set_timesteps(100, device=device)
        pipeline.scheduler.set_begin_index(75)
        kernels.denoise.begin_round(pipeline.scheduler, outer_index)
        timesteps = pipeline.scheduler.timesteps[75:]
        sigmas = pipeline.scheduler.sigmas[75:75 + len(timesteps)].to(device=device, dtype=dtype)
        if len(timesteps) != 25 or len(sigmas) != 25:
            raise RuntimeError("scheduler must supply exactly25 active steps")
        noise_seed = config.seed + 1000 * outer_index
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        noise = ledger.call("noise_draws", torch.randn, latent_start.shape,
                            generator=generator, device=device, dtype=dtype)
        latent = (1.0 - sigmas[0]) * latent_start + sigmas[0] * noise
        recorder.record("outer_start", len(trace), {"outer_round": outer_index + 1,
            "noise_seed": noise_seed, "noise": noise, "latents": latent,
            "latent_start": latent_start, "controller_config": CONTROLLER_CONFIG,
            "rng": kernels.rng_snapshot({"outer_noise": generator}),
            "scheduler": kernels.scheduler_snapshot(pipeline.scheduler)})
        for inner_index, timestep in enumerate(timesteps):
            step = len(trace) + 1
            kernels.assert_determinism()
            eot_seed = config.seed + step * 1000
            before_hash = recorder.record("before_denoise", step, {"latents": latent,
                "timestep": timestep, "controller_state": periodic.weights(),
                "rng": kernels.rng_snapshot(),
                "scheduler": kernels.scheduler_snapshot(pipeline.scheduler)})
            denoised, denoise_evidence = ledger.call("transformer_forward_calls", kernels.denoise,
                pipeline, latent, timestep, prompt_pack, config.guidance_scale)
            z = denoised.detach().requires_grad_(True)
            patch = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, z)
            with torch.no_grad():
                geometry = parent.RoadRGBView(torch.zeros((1, 3, 64, 128), device=device)).render(context)
                yy, xx = torch.where(geometry.support[0, 0] > 0)
                if not xx.numel():
                    raise RuntimeError("sample-specific road support is empty")
                patch_box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            height, width = geometry.image.shape[-2:]
            views = kernels.sample_views(height, width, list(context.object_box), patch_box, eot_seed, eot_config)
            if len(views) != benchmark_total_views():
                raise RuntimeError("new benchmark requires the explicitly declared shared view count")
            prepared = None
            if benchmark_uses_prefetch():
                from benchmark_execution import Prepared
                prepared = Prepared(parent=parent, patch=patch, context=context, views=views,
                    adapters=adapters, depth_objective=depth_objective, ledger=ledger, mode=benchmark_current_mode())
            with (prepared.consume_context() if prepared else nullcontext()):
                raw, candidates = kernels.detector_losses(prepared.parent if prepared else parent, patch, context, views, prepared.adapters if prepared else adapters, prepared.ledger if prepared else ledger)
                vector = torch.stack(raw)
                # The loss averages the declared differentiable shared views. Only
                # allocation deficits use view 0, whose exact native predictions
                # are already recorded by the unchanged frozen detector helper.
                identity_rows = [r for r in candidates["views"] if r["view_index"] == 0]
                if len(identity_rows) != 1 or views[0].get("kind") != "identity":
                    raise ValueError("allocation needs exactly one first identity EOT view")
                adv_confidences, adv_matching = [], {}
                for name in DETECTOR_MODELS:
                    observed = identity_rows[0]["models"][name]
                    native = {"boxes": observed["boxes"], "labels": observed["labels"],
                              "scores": observed["scores_all"]}
                    confidence, details = target_matched_confidence(
                        native, identity_rows[0]["target_box"], controller_config)
                    adv_confidences.append(confidence)
                    adv_matching[name] = details
                det_deficits, det_deficit_details = detection_goal_deficits(
                    torch.stack(adv_confidences), clean_scores, controller_config)
                used_weights = periodic.weights()
                det_weights = torch.tensor(used_weights["detector"], device=vector.device, dtype=vector.dtype).detach()
                det_weight_details = {"rule": "periodic_gap_gated_foa", "weights": det_weights.cpu().tolist(), "weight_sum_target": 2.0}
                detection_total = kernels.weighted_loss(raw, det_weights)
                depth_model_losses = {}
                depth_equal_loss, depth_evidence = (prepared.depth_objective if prepared else depth_objective).call(
                    patch, context, views, prepared.ledger if prepared else ledger, model_losses_out=depth_model_losses)
                if set(depth_model_losses) != set(DEPTH_MODELS):
                    raise ValueError("depth objective did not expose both live source model losses")
                depth_identity_rows = [r for r in depth_evidence["views"] if r["view_index"] == 0]
                if len(depth_identity_rows) != 1 or depth_identity_rows[0]["view_kind"] != "identity":
                    raise ValueError("allocation requires the same identity depth EOT view")
                identity_far_shifts = torch.tensor([
                    depth_identity_rows[0]["models"][name]["semantic_far_change_roi_mean"]
                    for name in DEPTH_MODELS], device=detection_total.device, dtype=torch.float32)
                dep_deficits = torch.tensor([depth_evidence["control_deficits"][name] for name in DEPTH_MODELS], device=vector.device, dtype=torch.float32).detach()
                dep_deficit_details = {"rule": "identity_roi_hinge_rms_over_beta_plus_epsilon", "deficits": dep_deficits.cpu().tolist(), "detached": True, "extra_clipping": False}
                dep_weights = torch.tensor(used_weights["depth"], device=vector.device, dtype=vector.dtype).detach()
                dep_weight_details = {"rule": "periodic_gap_gated_foa", "weights": dep_weights.cpu().tolist(), "weight_sum_target": 1.0}
                depth_loss = (torch.stack([depth_model_losses[name] for name in DEPTH_MODELS]) * dep_weights).sum()
            if prepared:
                benchmark_steps.append(dict(step=step, **prepared.finish()))
            del prepared
            for name, value in (("detection", detection_total), ("depth", depth_loss)):
                if (not isinstance(value, torch.Tensor) or value.ndim != 0
                    or not value.is_floating_point() or not bool(torch.isfinite(value))
                    or not value.requires_grad):
                    raise FloatingPointError(name + " task must remain a finite live scalar")
            if depth_loss.dtype != detection_total.dtype or depth_loss.device != detection_total.device:
                raise TypeError("depth and detector losses must share dtype/device")
            allocation_before_gradient = {"controller_config": CONTROLLER_CONFIG,
                "detection_deficits": det_deficit_details, "depth_deficits": dep_deficit_details,
                "detection_model_weights": det_weight_details, "depth_model_weights": dep_weight_details,
                "identity_detection_matches": adv_matching,
                "clean_detector_reference_confidences": clean_scores,
                "depth_identity_far_shifts": identity_far_shifts}
            prebackward_hash = recorder.record("pre_backward", step, {"latent_variable": z,
                "raw_patch": patch, "denoise": denoise_evidence, "raw_losses": vector,
                "raw_depth_model_losses": depth_model_losses,
                "detection_task_loss": detection_total, "depth_task_loss": depth_loss,
                "depth_equal_mean_reference": depth_equal_loss,
                "allocation": allocation_before_gradient, "depth_evidence": depth_evidence,
                "candidate_outputs": candidates, "eot": views})
            kernels.assert_determinism()
            # Two first-order task VJPs share the decoder/render graph. The
            # first preserves it only for the second; no combined third VJP.
            det_gradient, = ledger.call("autograd_calls", torch.autograd.grad,
                detection_total, z, retain_graph=True, create_graph=False)
            dep_gradient, = ledger.call("autograd_calls", torch.autograd.grad,
                depth_loss, z, retain_graph=False, create_graph=False)
            gradient, task_details = combine_task_gradients(
                det_gradient, dep_gradient, used_weights["task"],
                task_balance=switches["task_balance"], conflict_guard=switches["conflict_guard"],
                norm_match=switches["norm_match"], n_min=CONTROLLER_CONFIG["n_min"])
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("combined latent gradient is nonfinite")
            task_weight_tensor = torch.tensor(used_weights["task"],
                device=detection_total.device, dtype=detection_total.dtype)
            total = (task_weight_tensor[0] * detection_total.detach() +
                     task_weight_tensor[1] * depth_loss.detach())
            next_latent, update_tensors, update_details = kernels.clipped_update(
                z, gradient, 100000.0, .05, max_norm=120.0)
            with torch.no_grad():
                realized_direction = z.detach().float() - next_latent.float()
                realized_first_order_dots = [float((g.detach().float() * realized_direction).sum()) for g in (det_gradient, dep_gradient)]
            preupdate_hash = recorder.record("pre_update", step,
                {"latent_variable": z, "detection_task_gradient": det_gradient,
                 "depth_task_gradient": dep_gradient, "joint_gradient": gradient,
                 "detection_deficits": det_deficits, "depth_deficits": dep_deficits,
                 "identity_detector_predictions": {
                     name: {"boxes": identity_rows[0]["models"][name]["boxes"],
                            "labels": identity_rows[0]["models"][name]["labels"],
                            "scores": identity_rows[0]["models"][name]["scores_all"]}
                     for name in DETECTOR_MODELS},
                 "identity_target_box": list(identity_rows[0]["target_box"]),
                 "clean_matched_confidences": clean_scores,
                 "identity_depth_semantic_far_roi_means": identity_far_shifts,
                 "task_allocation": task_details, "requested_task_priorities": used_weights["task"], "current_period_weights": used_weights, "switches": switches, **update_tensors},
                 retain=step in (1, 25, 500), retain_kind="fixed_update_math_probe")
            latent = next_latent
            with torch.no_grad():
                patch_after = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latent)
            postupdate_hash = recorder.record("post_update", step, {"latents": latent,
                "raw_patch": patch_after, "controller_state": periodic.weights(),
                "rng": kernels.rng_snapshot(),
                "scheduler": kernels.scheduler_snapshot(pipeline.scheduler)})
            cycle = periodic.observe(inner_index + 1, det_deficits, dep_deficits)
            if cycle is not None:
                cycles.append(cycle)
                with cycle_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(cycle, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
            row = {"cycle_summary": cycle, "current_period_weights": used_weights,
                "switches": switches, "realized_first_order_dots": realized_first_order_dots, "global_step": step, "outer_round": outer_index + 1,
                "inner_step": inner_index + 1, "sample_id": sample["sample_id"],
                "case_id": sample["case_id"], "dataset": sample["dataset"],
                "split": sample["source_split"], "input_role": "sample_specific_attack_input",
                "noise_seed": noise_seed, "eot_seed": eot_seed,
                "updates_per_sample": updates, "raw_losses": vector.detach().cpu().tolist(),
                "adaptive_weights": det_weights.cpu().tolist(),
                "detection_model_weights": det_weights.cpu().tolist(),
                "depth_model_weights": dep_weights.cpu().tolist(),
                "weight_details": det_weight_details,
                "detection_deficit_details": det_deficit_details,
                "depth_deficit_details": dep_deficit_details,
                "depth_weight_details": dep_weight_details,
                "task_allocation": task_details,
                "identity_detection_matches": adv_matching,
                "controller_config": CONTROLLER_CONFIG,
                "raw_depth_model_losses": {name: float(value.detach()) for name, value in depth_model_losses.items()},
                "detection_total_loss": float(detection_total.detach()),
                "task_loss_dtype": str(detection_total.dtype),
                "depth_loss": float(depth_loss.detach()),
                "depth_equal_mean_reference": float(depth_equal_loss.detach()),
                "task_weights": used_weights["task"],
                "weighted_total_loss_role": "detached_diagnostic_not_third_backward",
                "autograd_calls_this_step": 2,
                "weighted_total_loss": float(total.detach()), "update_statistics": update_details,
                "clipping_diagnostics": {key: update_details[key] for key in
                    ("max_norm", "clipped", "clip_scale", "norm_before", "norm_after", "realized_update_rms")},
                "state_bindings": {"before_denoise": before_hash, "pre_backward": prebackward_hash,
                    "pre_update": preupdate_hash, "post_update": postupdate_hash}}
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            trace.append(row)
            print(f"sample_specific sample={sample['sample_id']} step={step}/{updates} loss={float(total.detach()):.9g}", flush=True)
            del raw, vector, det_weights, dep_weights, total, gradient, z, patch, patch_after, candidates, denoise_evidence, update_tensors
            del detection_total, depth_loss, depth_equal_loss, depth_evidence, depth_model_losses, det_gradient, dep_gradient
            del allocation_before_gradient, identity_rows, depth_identity_rows
        with torch.no_grad():
            current_patch = ledger.call("vae_decode_calls", parent.decode_patch, pipeline, latent).detach()
        recorder.record("outer_end", len(trace), {"outer_round": outer_index + 1,
            "latents": latent, "raw_patch": current_patch,
            "controller_state": periodic.weights(),
            "rng": kernels.rng_snapshot(), "scheduler": kernels.scheduler_snapshot(pipeline.scheduler)},
            retain=outer_index == config.outer_rounds - 1)
    if len(trace) != updates:
        raise RuntimeError("per-sample update budget mismatch")
    ledger.require_complete()
    kernels.denoise.require_complete(ledger.snapshot(), updates)
    checkpoint = f"checkpoint_{updates:04d}.pt"
    final_metadata = {**metadata, "role": "sample_specific_final_patch", "updates": updates,
                      "checkpoint_selection": "fixed_final"}
    parent.save_patch(output / "final_patch.pt", current_patch, final_metadata, raw_generator_rgb=current_patch)
    parent.save_patch(output / checkpoint, current_patch, final_metadata, raw_generator_rgb=current_patch)
    state_sha = recorder.finish(updates)
    summary = {**metadata, "status": "train_optimization_complete", "executed_updates": updates,
        "scientific_config": scientific_config,
        "updates_per_sample": updates, "cumulative_prior_updates": 0,
        "cumulative_updates_for_this_sample": updates, "outer_rounds": config.outer_rounds,
        "checkpoint_selection": "fixed_final", "fresh_sample_initialization": True,
        "initialization": "this_sample_road_crop_canonical64x128_then512",
        "no_cross_sample_state": True, "training_models": ["fasterrcnn", "ssd300"],
        "training_depth_models": ["monodepth2", "dpt_large"],
        "controller_config": CONTROLLER_CONFIG,
        "controller_rule": "periodic_gap_gated_foa_outer_reset",
        "ablation_variant": scientific_config["ablation_variant"],
        "switches": switches, "statistical_cycles": len(cycles),
        "weight_refreshes_used": sum(int(c["applied_next"]) for c in cycles),
        "cycle_trace_sha256": sha(cycle_path),
        "legacy_depth_constructor_enable_flag": 1.0,
        "clean_detector_reference_confidences": clean_scores.cpu().tolist(),
        "uncertified_detector_reference_count": int((clean_scores <= controller_config.theta).sum()),
        "all_predeclared_scenes_retained": True,
        "detector_forward_breakdown": {"identity_clean_once_per_case": 2, "attack_shared_eot": 2 * benchmark_total_views() * updates},
        "task_autograd_calls": 2 * updates, "combined_scalar_autograd_calls": 0,
        "gradient_clipping": {"max_norm": 120.0, "coefficient": 100000.0, "lr": .05,
            "stage": "amplified_total_latent_gradient_before_lr"},
        "trace_sha256": sha(trace_path), "budget": ledger.snapshot(),
        "state_evidence_manifest_sha256": state_sha,
        "patch_bindings": {name: _binding(output / name) for name in
            ("zero_step_patch.pt", "input_raw512.pt", "final_patch.pt", "initial_vae_reconstruction.pt")},
        "checkpoint_bindings": {checkpoint: _binding(output / checkpoint)},
        "elapsed_seconds": time.monotonic() - started, "resume_enabled": False,
        "validation_data_opened": sample["source_split"] == "validation",
        "optimization_on_original_validation": sample["source_split"] == "validation",
        "gpu_acceptance_claim": "not_established_by_this_training_receipt"}
    summary["benchmark_policy"] = benchmark_policy()
    execution_path = output / "BENCHMARK_EXECUTION_001.json"
    write_json(execution_path, {"schema": "new-benchmark-execution-evidence-v1", "policy": benchmark_policy(), "steps": benchmark_steps, "scientific_acceptance_claim": False})
    summary["benchmark_execution_evidence"] = {"path": str(execution_path), "bytes": execution_path.stat().st_size, "sha256": sha(execution_path)}
    write_json(output / "summary.json", summary)
    return summary


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_verified_helpers(paths):
    """Caller must run source_bindings.verify_sources before this import."""
    for key in ("r2_source", "v2_source", "clip_source"):
        sys.path.insert(0, str(paths[key]))
    geometry_path = Path(__file__).resolve().parent / "geometry_eot.py"
    if "geometry_eot" not in sys.modules:
        _load_module("geometry_eot", geometry_path)
    if Path(sys.modules["geometry_eot"].__file__).resolve() != geometry_path:
        raise RuntimeError("new benchmark geometry must be bound from the new scientific package")
    clip = _load_module("_sample_specific_frozen_clip_training", paths["clip_source"] / "clip_training.py")
    from adaptive_loss import foa_weights, weighted_loss
    from clip_update import clipped_scaled_update
    from geometry_eot import sample_views
    from evidence import fingerprint, rng_snapshot, scheduler_snapshot, CallLedger
    inputs, parent = clip.load_parent(paths["child"])
    kernels = TrainingKernels(clip.denoise_with_evidence, clip.compute_losses,
        foa_weights, weighted_loss, clipped_scaled_update, sample_views,
        fingerprint, rng_snapshot, scheduler_snapshot, CallLedger, clip.assert_determinism)
    return inputs, parent, clip, kernels




def validate_ledger_snapshot(snapshot, expected):
    """Use the actual four-key ABI; never compare three count groups to PLAN.

    The fourth value is retained exactly from the source-bound original ledger,
    not invented or rewritten. Its scientific source/whole-file identity is
    verified by the caller. Only the three count groups are compared to budget.
    """
    if not isinstance(snapshot, dict) or set(snapshot) != {
            "planned", "attempted", "completed", "counter_semantics"}:
        raise ValueError("original four-key callback ledger ABI required")
    for group in ("planned", "attempted", "completed"):
        values = snapshot[group]
        if (not isinstance(values, dict) or values != expected or
                any(type(v) is not int or v < 0 for v in values.values())):
            raise ValueError("all actual original callbacks must finish exactly: " + group)
    # This is an exact retained JSON copy, not a replacement semantic assertion.
    return json.loads(json.dumps(snapshot["counter_semantics"], allow_nan=False))


def with_real_conditioning(original_kernels, callback):
    """Build a NEW dataclass; never modify the old module/object or denoiser."""
    if type(callback) is not ConditionalTrainingDenoise:
        raise TypeError("the new explicit conditional callback is required")
    from dataclasses import fields
    return TrainingKernels(**{field.name: callback if field.name == "denoise"
                              else getattr(original_kernels, field.name)
                              for field in fields(TrainingKernels)})


# Admission metadata only: same original200, no input sampling or numerical change.
FORMAL200_MASTER_REFERENCE_SHA = "cca50857551d8125dc83a0c77de34fca955471b840e78dfdfea39a5059e3dc33"
FORMAL200_MASTER_SAMPLES_SHA = "78fc617bcc0534c95288b6f1a91af5eb68707af49d2bcaa9b0a631788dc6a96a"
# Immutable rows: (original global ordinal, sample_id, recorded split, full-dict canonical SHA256).
FORMAL200_CASE_ADMISSION = ((0, 'kitti_000000', 'validation', '05d1f2b28f81eb2856193bc9819b9b2104f1860fc2fb20d6257b4b53a872212b'), (1, 'kitti_000046', 'train', 'b5ed0f39b56715e6231ffa1d6c545a2bef9b6020c23ceabb036a66ee8742a94e'), (2, 'kitti_000097', 'train', 'bdf0d94225553070f3e68accbd1fec948094874ca8dbf1f858f7e4819d5cbc69'), (3, 'kitti_000143', 'train', 'd282780295bbb2f8c138c9502e0441f0db2baf6f844aa0ad1b9b9a158033c528'), (4, 'kitti_000155', 'train', '18da9c9b9be8e998f70d92ce5717a13aef1e59240318527405ecb9c2655e2003'), (5, 'kitti_000164', 'train', 'cc57880b977a632b7beabe8b651bb8d169e94bcdd5394fca8c2471ebdb5510ed'), (6, 'kitti_000186', 'train', '268d269bd1b7744051545d31dba6f865eed245e97bf06a11ab1fae6b708ec756'), (7, 'kitti_000192', 'train', 'a157f0f76908ef1b3a050198435f114e218e3e949b830cff16be0e4e61d86255'), (8, 'kitti_000251', 'train', '9918ac1ce9f09390f605c8f799cae7d4823a5c2e851f40aa1e20d89b3ffb2d69'), (9, 'kitti_000275', 'train', '4f22666bb0cbe4dccc40068eda6a264c1f5e64a2c13dc5e074d81967fbb47dd0'), (10, 'kitti_000295', 'validation', '5bcaa3969f80967ec2c941818d09f701059308188166f83ad86346a76987957b'), (11, 'kitti_000563', 'train', '3b07c0572afedca06243333f1ccace274ebee8822acec8a508449e3a4640c7cb'), (12, 'kitti_000633', 'validation', '75ed9cf8ee9bf7f3838d99b7720f1a0c6f3ced7b02f62657060a50380649c43d'), (13, 'kitti_000673', 'train', '911300767385be0ca58be741ccb0ae23c921e0184a57a954c06b53c172e2a684'), (14, 'kitti_000708', 'train', '01447a3e837953b01f49a31a0ad16627940db50dc0cc511f33e05c0adfb57d01'), (15, 'kitti_000864', 'validation', '133b20901ad062debce0a32b174695cd802efe6f8629e2c59dd656c2b14d055b'), (16, 'kitti_001034', 'validation', '1ab09099d24d556c73420725097229b4b75183c1f9455400f18c34f505760fc7'), (17, 'kitti_001039', 'train', '8bf3c808e41488d2d240f332d6b78d71462a84366efb48f23f3629481db41336'), (18, 'kitti_001117', 'train', '0ca1700416cf99e2df82426708781deaa5bcb043b827050bc0745b6633eb17de'), (19, 'kitti_001119', 'train', '4cc650d63a04e76a2bcdf7bb03bf76c88566fba9e5751b2481fc11f459a36429'), (20, 'kitti_001149', 'train', '1a50203296b373e80068860e3745366a4af00a3dd93659b93699401b78c03d1e'), (21, 'kitti_001309', 'train', '9aec37b5b747121c021f41603835d7c04fc32700960a7c6c1e1796ad582ec97e'), (22, 'kitti_001351', 'train', 'b31258a34375a9af642ca690793c6b4f2b3a619142d4d9ff606c4a67f781b56d'), (23, 'kitti_001389', 'train', '03951669c9d260617187a23132e9f90f0c836815345dae2c4b5884d2ed34f213'), (24, 'kitti_001481', 'train', 'c1046e6decf3e6e136cb4f7199af87813dc5df0390335949e95f1637c1644251'), (25, 'kitti_001512', 'train', '779ccc5708020b1f1b33ed5aa87359471e6a78652901a6f6dbd9c09e83acecf8'), (26, 'kitti_001610', 'train', '12675c53aebe274ebb98568726e2aa7c48ab9fd3299fcca9b502d5f49176ed17'), (27, 'kitti_001613', 'train', '967e900230ef3607bde3774c4cdff6a50537f627d5dfe8df0fbb1f300e71931a'), (28, 'kitti_001616', 'train', '6f69b1cc41ae470119fbeefbd7f72fe548d707608ac9354d3f4748070a3f3768'), (29, 'kitti_001642', 'train', 'eddcdba02b137ea998e4e51499b9bc9ea37007113656fc3e6a2ff25dc8985f50'), (30, 'kitti_001671', 'train', '02db291baec647f44aa3a7429fa11cc68de72844becfa698716cd535025a3807'), (31, 'kitti_001893', 'train', '6babd9725ec3cd2c49c3c8a6078733432595bf33c03b52db81c06cae3075ec22'), (32, 'kitti_001894', 'validation', '9420e227fb80c843f7fc4c555c19e3ffe8fb8784b5ccfa7433df799ebca145c2'), (33, 'kitti_001984', 'train', '189fc022849223105acdc46a392b54040c94e20d112d91779e5334ada50b782a'), (34, 'kitti_002015', 'train', '4fe8dfd610f38e364996e0a13013827cc835da05e3bf2af548cfac0004a8a205'), (35, 'kitti_002060', 'train', '1a491e066025f5a24fb64a0b3e41a16d06a5eaffc5487edb491883bb451982c5'), (36, 'kitti_002119', 'train', 'f280b9bf5d995573cf72ce2562f2ce0c612437b6980f9fa7301d054f1acb4390'), (37, 'kitti_002236', 'train', 'bee4f415454c7dc9b4bbc39b85a53ccdc877eaf2b372bc35fd357c463cad3917'), (38, 'kitti_002434', 'train', 'd50ca08e3feac12d67c5d1967e488d1fec668c6af6e814b5950650db5e5dbd54'), (39, 'kitti_002633', 'train', '28c6e80aa67b631a148a3fd8d54f7a68c1dec9de3447b3d0780fc39c5fd74d41'), (40, 'kitti_002636', 'validation', '5587d9abd733f2ab505ee164268de086f8aa7c17d20909c1085d4db9d7700dea'), (41, 'kitti_002672', 'train', '2db75f1cad1decaa04b23023cb5ebd8c14f88a6b9d9f3b6e1f5142d28e508aa9'), (42, 'kitti_002808', 'train', '8145c21cf0743b2c84329035c1f38f8c94e2df7352f187897b53b27155906d08'), (43, 'kitti_002839', 'train', '45718793f59955742a49a13391b028f5aa6d2f00d0f0af2fe282f316fd73a1ce'), (44, 'kitti_002884', 'train', '1beac001ec16a507ecd6d67f5eb045dc5b590d6f8ca89078a14410fc08dbbed0'), (45, 'kitti_002902', 'train', 'ec916b4361ffaee55acd832df2b3ca0d4d00d8c88be4d97eee1a8e4c358b55af'), (46, 'kitti_002919', 'train', '16de14597f199d96b55e39da507867aa3a5228e7b21a8f13cd7f4f32c7230710'), (47, 'kitti_002951', 'validation', 'a8754091d45da0799b63827df9c25080e1779cae8cdfd0680f415b9c90b12386'), (48, 'kitti_003070', 'train', 'e9f5b3b93a0cb5c6868e45dd9a5044f3a203506caa070e9fb04b73a4e5081757'), (49, 'kitti_003089', 'validation', '23778af3aa1eebee768a40f3d08da38ff44cfc0de653a4e50cdd1809fb8faaa1'), (50, 'kitti_003180', 'train', '616b9b38c0cd1dac8a3ae9d14c8f99459bcc890a59aa242a5a91fc3699895e78'), (51, 'kitti_003277', 'train', '73baf47b3b0a1beeae08605b7cf567f71863b6a37f35a2f985d486cd8c76c8fd'), (52, 'kitti_003334', 'train', '8ae7d7f70000e6c530555ec39c188c4bf204d7bc206a99f0667df676665a74bf'), (53, 'kitti_003399', 'train', '0a76b93fd6643a4c78f55be636fa30ba466c7fbc5643b2fcdfda0ac2a0059267'), (54, 'kitti_003424', 'validation', '54d201e705506b3c2572be5669c68f4fb2134128b380507d81f0e3db527fea0a'), (55, 'kitti_003462', 'train', '908c52a53914573599fe71805b3d8f96d370866377a8bfa581f58e170dfbc16c'), (56, 'kitti_003625', 'train', 'eddb10ccb3e00d7534798df593bc52e192d3a5508c0829a7c1d3f95508324ee2'), (57, 'kitti_003666', 'train', '4b4984a48489b9cef42b00348c304edfcc8d37352ada366a1e0cb26d6a7210dd'), (58, 'kitti_003954', 'validation', '74da8810bb9a6f7985f2e9823397cfd7d59561c9e4a84a4a4c7d1e5478745c1b'), (59, 'kitti_004442', 'train', '4375b34f8d392c96e0692bcea7dd769a8d069c293eb45ad590ed56f560074840'), (60, 'kitti_004478', 'validation', '221ec6be58f9d531b90c4479b4a958dcc466d82b654a0df308a1942de77a2c4a'), (61, 'kitti_004513', 'train', 'abbb0e76e61b4c8e3f37d01672483ec7434fd6a26db1ce303657efb6d2b745ab'), (62, 'kitti_004591', 'train', 'bd2450663bd120e139fb9f3c732da474e39e51107eda2a406f1d9f8829f0f40e'), (63, 'kitti_004617', 'train', 'e611221f851f6044dad73b601d722577aede4f76c4dfbd719366ac520cb256d0'), (64, 'kitti_004707', 'train', 'eabbe04747a1177f999ae88a52e4be05ecda283ae63751b3e72a089a715ed5d2'), (65, 'kitti_004749', 'train', 'f28c9d1bb4f37b0859df4f21917ac5dcb5ad69eee4443f5878552b73d6a37258'), (66, 'kitti_004760', 'train', '6a123ce3ca293516689044f76c9f5b69310afd52ebc2e7c432a085d59e50730f'), (67, 'kitti_004909', 'train', '3740048045f1a0eb1ae7a8d1cd22ed990bf18478f46e4826f048987fc706cc37'), (68, 'kitti_004994', 'validation', '54e0d4ea37b592b524c59838cf468b22e6d3fe3e2a3c10d7b9935a29ecb0f804'), (69, 'kitti_005190', 'validation', 'dcd144b1b1125af926ff43c7ab523b5f8a72d66d24fdc2c92b6c65c9ef068ecb'), (70, 'kitti_005390', 'train', '33dbbaf222f79c795dc14f298a64a8b51730082bedeae287392c0dfc0f8efa75'), (71, 'kitti_005411', 'train', '90c0ebc201edbd5d455b9d60730182295c7f301a9c87e9aca97670b1a9d6302c'), (72, 'kitti_005455', 'train', '9e9764f37ab9f0ebfbdce1a8324b7dbf4df7619e44bd826664da4ae909fdeb01'), (73, 'kitti_005564', 'train', 'baf379049252409fb9a2a8281e74272de647a0f8a054409505289733bb25c054'), (74, 'kitti_005619', 'train', '91b02220de11067c54255c27ff8010a155af0079909cb1f6024f5e545c93fac7'), (75, 'kitti_005765', 'train', '5bf70b0580d34086465dfb16ebdee5c54436daf008aff6057954b7f8ebd92ee3'), (76, 'kitti_005902', 'validation', 'd852450a872acffc936cf00bc9de797254b7bf6a52fc86406538c28a1d75e256'), (77, 'kitti_005965', 'train', 'eb2ab212d6461e2dca5d1c194f78cf5ccfa11c575afcf80370d230908caac8c4'), (78, 'kitti_006149', 'train', 'd6613db4fa61c409a94173566df7991727db6495150fa3fba3fd4fc218ea0b2e'), (79, 'kitti_006153', 'validation', 'dd71fea3e981be39fed7ebef03b97c06fc2e0a4552c2290727cfcc55edab4a9d'), (80, 'kitti_006399', 'validation', '56fe5f523fdcc9909980925f581b16cfa4bb10142bbfac5850ebedd365e8a01f'), (81, 'kitti_006444', 'train', '08b1611fceaa230d2f350fed257e29bbf6bd826fdc6aedc6d9fca5e0347e7de9'), (82, 'kitti_006529', 'train', '10988c2c87b8a55f8670c9450562b4eb8626093537d650f143fad864603cfd96'), (83, 'kitti_006637', 'validation', '60eda5b92d54a4c1012d19ec0533905358084162b9e308881173a7ed2d504136'), (84, 'kitti_006646', 'train', '20bd9c3fbdad6548951473d5aa4520f83c132a9870f01645162c647f20e801c7'), (85, 'kitti_006665', 'train', 'f757429d8fee4d0b10a550b320b47bd450d576b29937874882c985381674a08b'), (86, 'kitti_006748', 'train', '5271bd8efe3a02b7f4353e4a83cb407da241bb9f3f626eca68a4b832a02542a3'), (87, 'kitti_006749', 'train', '82c3f75ccca71efbb0f48027d6a31e7b730e6cba84fe54e81a3c1d64a2ca5416'), (88, 'kitti_006778', 'train', '96f82189182de3df2d0b68688f93b61793ea72f675c3631e3f1b029b77642c66'), (89, 'kitti_006825', 'train', 'b0c9b927ccc41712e9b48b0f54f3ecf9c936863228ede99cfc6ce8fecee72a9d'), (90, 'kitti_006863', 'train', 'f8d1c8aa4352dfdcbb87144c7471b012adec54290cf20d72ddda1e9c5d542cc6'), (91, 'kitti_006953', 'train', '0d9efb8818434787e3ca2437998115fac56cfb0a1be8f0c41218eb4e4f7e4757'), (92, 'kitti_007067', 'train', '0f385cdd98bdfa068b49d89156db9243775815d757222c1924bd52054183f98b'), (93, 'kitti_007073', 'train', 'b598d1f239d3e6af8ec8307db93e6f4774a4a81586512ec21f8ae475a9aba01b'), (94, 'kitti_007075', 'train', 'c2273f5b44512ff457f77536abcbaaaa5faf82d0b5671248afe0262890eb19b7'), (95, 'kitti_007156', 'train', 'df7d0c31aa6256c76b6a2c8328c945feb895793f421076a5d86e9f48d1951119'), (96, 'kitti_007200', 'validation', 'ec30eabde2bc75cef2f4b35f173f566af9be1c8ae0285b3cae65ddde6dc801ac'), (97, 'kitti_007260', 'validation', '5f8c93f77f86397b002ae3fabb65bc2109ee869027850b4fd95f71142b2a5c2b'), (98, 'kitti_007377', 'train', 'd613589b0bc53fef15213023b17c4045905b507c4c3229712ce7390a0beb0718'), (99, 'kitti_007416', 'train', '16657a30dea0e5d08744370016d52f4fe0a31dd2725e4b9dde7b4c58886c9470'), (100, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404', 'train', '830004aaa232d1c41d4a34fad27c5445ceb06e9a576470e82870addf19c0bdcd'), (101, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604512404', 'train', '2855b5bbc7a9ff36968658ce8196362a96c07ff8b3331d941129102df68111fa'), (102, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605012404', 'train', 'da7c628add8ca10a9e6a1aac5a9ec2bdba194a7456ef3265009af7c1395f0289'), (103, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151606012404', 'train', '2d0b9166ed242303b698e481f67fb9313dfca9a1d590887e015de259b31b9f91'), (104, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151606512404', 'train', '8bf17511ab387fff1dec86d2abfddf948c62de194687f0d3a5cb5a9c2de9957f'), (105, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151607012404', 'train', 'fa1e6cac1f510f61bb2c1c4b385c0974cc11f070cfe3f2b7163d3d2f06165af8'), (106, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151607512404', 'train', '2e30bdf20586eb9a39344074c7c2d14d89beb39fe66ce2ab9448bdd380aad6e2'), (107, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609012404', 'train', '42fc40a8ccda1732fbf7f5b68811103832a89f18c6ae04393bc023ca95311e0f'), (108, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609512404', 'train', '4191d80e5abe8609e8f60e73446ce3e8a6f4f0db81f1f37e6a101292de494985'), (109, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609912404', 'train', 'e7994a148bef9b848b428fba748df6886178436369068d8879600624ab7c610e'), (110, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610912404', 'train', 'e693d5ca17be26ef1374294be6e9073099b1f38be74d5c1f166312ccf0b2c83c'), (111, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151611412404', 'train', '37fe0f515fc253d329ade815f6f757b89a8632ac8457f2e47d0f83a56e1199b6'), (112, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151612862404', 'train', '7b5d3a92790f75807748c387cfe77188dde327e8bd8af8fcf19d53bd0034e77f'), (113, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151614412404', 'train', 'e142b0604d064d7f2336d8fb92028e08379bb283e2f137d6d6f9f0fd0ac35606'), (114, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151614912404', 'train', '154646c8b356277c1f9c066a059e9c1e57f885aa2d6ebbaa3c24c723ec74a721'), (115, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151615412404', 'train', 'd470881cbc695506a1df0e5a8ef10d2c15bb160dacd9e15b074b701be27ff288'), (116, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151615912404', 'train', '5549fced9e2db2a0402ef32f7a44c528378350dc5a66816b754039ca4911dd87'), (117, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151616412404', 'train', 'd60aadb613e7701a40563941a1bc3e2e3bd57135f0db8449ae90a4abe7d49b5b'), (118, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151616912404', 'train', '9c632e0a660a314de278ac92d93249b077bcf7e979587b7f4bc33747e50618d8'), (119, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151617912404', 'train', '788492e887a105da27062aade7f5fea2f6200a1592e8edfb695103e395261332'), (120, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151618412404', 'train', 'a65debe1bc812d7afaa81429e8d197a1ba19abc7efb6c0af5044ed4300b857ee'), (121, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151618912404', 'train', 'df5dec4e6bca2538c4159077f8b8e8da636ea236f373601848253343f9342da0'), (122, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151619412404', 'train', '9abf7ca108f7f4c82067cb9a4b0e7f537ea89535f4f2390034081faae1f7cd48'), (123, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621012404', 'train', '5c13b6fe76b967e9d23cd8dc2dd4c8fb848aacb1d08d88384ac3ae6a19d259dc'), (124, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621412404', 'train', '68559d64a9a055ced814a9d7025b7ebd83a37af0a2391322833c94de7fbeffdd'), (125, 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151622912404', 'train', '93e2685d545c22a4334c1c705061d600387200903551b6ac895b7cfdf0ce258e'), (126, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385092112404', 'train', 'dbf57026d694bcd88a902fdde05c3a058a81f4c0645ff9310cb67a850d314233'), (127, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385092662404', 'train', '1f38a2d6190a493f788ddc9ed3d31062b9a4795dd8818a85de96150c3fb61512'), (128, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385093612404', 'train', '35b0486409e51faacb27623919badd3d4e65c93e6687aa192f73bffa2f4ec71e'), (129, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385094512404', 'train', '910b2a1ffae621383bee886d8d6a985245573a75340d7159da688bdb87479f85'), (130, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385094912404', 'train', '7f0729be5ebcad2cd8d83b861798d44123c3f7e282f1f0ce65a4bf8427e5c7f3'), (131, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385095412404', 'train', '287313dc5e0b328e3895c4c069f1231732603339075861e51d0d43dc981c9dab'), (132, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385095912404', 'train', '462491b4d375bc4fe5232cd5bcec9e032127cc544739c1091ae813b0d637f9dc'), (133, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385097862411', 'train', '29caa32896efb143810d8a7fb102cd183931dc13e8f1884fab311a0b68594601'), (134, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385098362404', 'train', 'ed197acb59113ee2c5d2f2c4213f52390988a74b2dde612382fc10251db47b75'), (135, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385098862404', 'train', 'ed7c4e50e93bc9d82b9f54442fa54ae0a80878a370b466b14bdb652e63bbbd9d'), (136, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385099862404', 'train', '60a2e14d24e27ba0db56a7098b991a9b8ca5188f25e026f515b9778e1261140f'), (137, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385100862404', 'train', 'f63752339202013fc35219ad845b3d7462bd92bf2cc5614a2f33268dc5b09f30'), (138, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385101362404', 'train', '46fd9090cacdd06584b66420ecf22343c5d0d07544c1b36b920b9ce06f167f6b'), (139, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385102912404', 'train', '93b6e715a9350a87d6f03777ed49184abdeeec0f8c8426636f6c8c4af2c14fe8'), (140, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385103412404', 'train', '2d7b2d49cd237786b2f2ed46b15ccc319e227cb8f5cb1b65aa7afb326ae4639c'), (141, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385103912404', 'train', '17268d5b7ba3a90487bec1bbf6a1e5c04d6be9da9fc512a8cae63e73b16126d0'), (142, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385104912404', 'train', '9551ade9b6327c5342d31a011bb6ee6116244432783bdfd306386f79ac90b7d8'), (143, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385105412404', 'train', '7688b424adfce27d5432c745bad454aafd74b7fe0aca5603c0a09c9f28a8e6cb'), (144, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385105912404', 'train', 'c214defc962a98ae0cec462a3c8a6fb8101575dac547ec6924649507b2d95365'), (145, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385106912404', 'train', 'fcdad7045686a130f2ce0c38519c587fd639b8b961ba5a927c98418e9defd95f'), (146, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385107912404', 'train', '65fd353861a928dfd8e94a7e2223ea1f0f4bcaf0d0437f75fea449a81735e17a'), (147, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385108412412', 'train', '962ceebb7dc1f739e7d622d51c92adbf0c34b85ff009076f98ce80ad8a765266'), (148, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385109362404', 'train', '83ddf362ecf84ac0a5f5ccce4b9672c8437fb9dadddf6fe3d524d7f9001055f6'), (149, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385109862404', 'train', '2c94f21403d547561355f1378f91fe6c34d6089c1e248511f448b44a6e88b8fe'), (150, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385110412404', 'train', 'fb36d7da58278f5cba7d9c1c7f908c37dc0a69cbf96b0ec4d69464706ddbd0ee'), (151, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385110912404', 'train', 'd20923df7d66ec642e96f2952ea60b422aabc22510b1622207d6f132f9534273'), (152, 'nuscenes_n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385111412404', 'train', '6aa7b9c23455f41ef72d827a0f25a5f7dde46fe5b75cf928e5e73239582d6666'), (153, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489296512409', 'validation', '000f9a7d5ec6f6c8a0d6a10ab7e24430ea6b9ee559ccfbc682304642bb2b7a60'), (154, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489298012404', 'validation', '72e2c3e8daa3c7531a4d4574bdd774c228eabb0cb1c1cef50b69fcd53b8272ac'), (155, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489300512404', 'validation', '9c8fdbd3b8b4832091b4cc31616dff473574e517246ecd0c314ab5c8bfee92e6'), (156, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489302512404', 'validation', '7e693e50ebfffc48a336e26382523bcd01f9e2a3ec572e531bfc7b9e3b5d4189'), (157, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489305412404', 'validation', '25a2fd8223781817d4fc33f56b80fbd110669ab481368424dc53a13a8a13fdc5'), (158, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489306912404', 'validation', '394b2fad51a3a2cfa047009e31fffdbe2cecb0cec4f9baa0c26997de46eabe9f'), (159, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489308362404', 'validation', 'f7140f53d501377f14805a488c14f4d317777b23143d47af0e9219bb5c53836f'), (160, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489310362404', 'validation', 'ac90fc501695b074c91630f0d728c69c8cbe16f50b1cd436017abbdee012fb14'), (161, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489312362404', 'validation', '62a88e448d0e5da95c603962366eb2b6f0c8359c5e0998ba097407632a93dc2a'), (162, 'nuscenes_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489315412404', 'validation', '0ad9786794c872a3368c88f9c34312f090d8877e5c0d2004cbbc38668350fce8'), (163, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657108762404', 'validation', '3cb6f9a69e94266a1fd7385bb0d376b10994f3ad8ed244660a43faa6643f1917'), (164, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657111612404', 'validation', '3740648a44cf691f72a8fb30b3a56604bf4901874a19757001eea5d14dfdf078'), (165, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657113612404', 'validation', '4a591ecf1fcedd95bdbf937f07d66450c0984c8f0a007cdee3a0450549257273'), (166, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657115112404', 'validation', '73080d4b874b08adcdb8a5431e32011c8879d1411153e63daa88a898f6220ec9'), (167, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657116612404', 'validation', '52b0c4174b9309324a06250ba05c85e8f66bb628dca25210eb5d55b0f34bc6ed'), (168, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119612404', 'validation', '9544f3841c3c6dead029d21dae722359445e45f2d4e4437daa63e905154c4c1d'), (169, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657121112404', 'validation', 'd3f700f8853353c52f2a9ad34b78f29f8aab750e0c91d471fb06d4ef3c8b51ae'), (170, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657123112404', 'validation', '163034d3462356129307a049e18b5717908897aecb31aafb1378de2b6d72897f'), (171, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657125112404', 'validation', 'ddfe2aed7c5e40e13896103d823adbef18dd1f8098628ba71525a0208a1ff98d'), (172, 'nuscenes_n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657126612404', 'validation', '52b3035a59048e02580b9b01a2d0f79fe7fa6e525187f3979af08959562638d5'), (173, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984233512470', 'train', '922349f6a76e1632794f5a00898e7d5b2e23888afd2e5bf8d1f27641c427d23c'), (174, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984234012472', 'train', 'da71a4b6e69b2b9471d03990214cbf822119e2c6bddf53276efde02e93f13e77'), (175, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984235412460', 'train', '43315a5fff787d768af3ed6c48657012bb9e6c2d17af4503edbf8b58de13b97d'), (176, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984236912460', 'train', 'e79760042e65fbc9f1d3edeeee67a24cb133ed542df65de2c79d58987dae12e8'), (177, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984237412460', 'train', 'e2bb3a36a0630a29755398a8671333ba52b4b6386a5ebe5385a1dc544850f33c'), (178, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984237912460', 'train', 'aa76c7bb26c246327539c80cb1ee7b79ac796d510bc7b7270d1249d7e01ae8b9'), (179, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984238912460', 'train', '205a9faaa2729c39deec8dde531221643398c2dd5983f6d73ec59be42b4fee9b'), (180, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984239412460', 'train', '905acabb51907167ad0bed83851ecc5b6832667ac86ffe9d6ec308196c34e278'), (181, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984239912460', 'train', '22f7363be2f4dac8cc43e530b29f039e78c2e5108d54258ab7fce042546434cd'), (182, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984240412460', 'train', 'c65f8eb306672df974b01a7c526368093e6bfcff60c43cca8b7ff1c2054e47ba'), (183, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984240912467', 'train', '0da68c042a5ff8a32ca1f745beff67013375c11eaed42ed198a5cc7b5609ad90'), (184, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984242412460', 'train', '246613fbd7ce9c2bf8aeac9218fe0fc233e105cdd2654a446ac9455a7e48f2c9'), (185, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984242912460', 'train', 'bf8387476ff6068ad1c8756cb514d6b2767f6a68fcc09de37ea5dd266b5ade63'), (186, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984243912460', 'train', '1783da38f25d4d99e7f753781b6f8f85fd055acabe97c9e3c0b35755c2c06ae8'), (187, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984244912460', 'train', '25f388c5de98aebf481b0f1f0ec15255cf2445a072f595e3b9e90368702286f9'), (188, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984245412460', 'train', '5d867c9050b72ba7fa275d649fe66a862aaa0a97a716aa0ddd4cc15b3694d17d'), (189, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984245912460', 'train', 'd97537ef6799736e43c523c65d91e5bad5a4de7fc52a0a0a6dee3fb8e717a2d7'), (190, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984246412460', 'train', '91e430f7a7aaed8fa79a88292d578a351706fb3c0dbd1d016eb70f7c5b3d8ea5'), (191, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984247912460', 'train', 'f46349ffc14426ade78cb8889757486425b34ff167c2b806feef38055a064835'), (192, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984248412460', 'train', '3536c358d3cd2670c33076bf68e539d9b6888282534a016a4abc48f490e1f01c'), (193, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984248912460', 'train', 'cbd37bbeb80c6a2872b0fe0112abab356fabb4676de710387faab28bfed3bca1'), (194, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984249412460', 'train', '6df4c432dcb8f45263a1fd04249de0e8b20653607f58e532b60ddf1beb0d3419'), (195, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984249912460', 'train', '91db3f3a3cc2e618fe899ef2bad4265c2ba67256fc154701c326c83072b7549e'), (196, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250412460', 'train', 'df58a734f247f70b7866ec336613aa739bf546f11276b305e757901923613d45'), (197, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984251412460', 'train', '224d60ccbf09c05779f732dec323d3e9ae34d955e1fb3c946a26e295d248456c'), (198, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984252412460', 'train', 'daa8508092a3220f47fa249da54dbc519457a296c52b054f103638776acb84b3'), (199, 'nuscenes_n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984253412460', 'train', 'badf3b602877df5e6cb6e73315addff4356a9aa65fccc028d0b051ae93cfba11'))


def _require_conditional_run(kernels, pipeline, prompt_pack, config, sample, output):
    if type(kernels.denoise) is not ConditionalTrainingDenoise:
        raise TypeError("original fake/identity denoising callbacks cannot enter this derived run")
    if benchmark_current_mode() != "C1" or benchmark_total_views() != 2:
        raise ValueError("this isolated generation comparison requires original EOT1+C1")
    # Each full sample dictionary is bound to its original immutable MASTER200
    # ordinal, source split, source SHA, geometry and all remaining metadata.
    # The unchanged caller additionally verifies exact ordered originalpart50.
    if type(sample) is not dict or type(config.seed) is not int or config.seed != 20260919:
        raise ValueError("original MASTER200 dictionary and seed20260919 required")
    admission = next((row for row in FORMAL200_CASE_ADMISSION if row[1] == sample.get("sample_id")), None)
    if admission is None:
        raise ValueError("sample is outside the same fixed original MASTER200 cohort")
    ordinal, sample_id, source_split, sample_sha = admission
    canonical = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (not 0 <= ordinal < 200 or FORMAL200_CASE_ADMISSION[ordinal] != admission or
            sample.get("sample_id") != sample_id or sample.get("source_split") != source_split or
            source_split not in ("train", "validation") or hashlib.sha256(canonical).hexdigest() != sample_sha):
        raise ValueError("exact original MASTER200 ordinal/sample/source/geometry/split required; TEST forbidden")
    kernels.denoise.begin_run(pipeline, prompt_pack, config, sample, output)


class ConditionalTrainingDenoise:
    """Real no-T5 generation callback using the ORIGINAL loop-owned scheduler.

    This intentionally does NOT call kernel.start_round/generated_leaf or
    decode_leaf: doing so would duplicate encode/noise/scheduler work and change
    the original FP32 attack decoder. The original run owns encoding/noise,
    detach().requires_grad_(True), four sources/two VJPs and updates. The kernel
    is used only for full module/type checks, actual two-CLIP encoding and hooks.

    The callback copies the standard SD3 CFG/Euler operator sequence, referenced
    against the admitted parent.standard_denoise_step source, without invoking
    that mutable parent pointer or the old clip fake/identity helper.
    """
    def __init__(self, pipeline, bound_kernel_module, *, deadline_unix):
        import math
        if type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix):
            raise ValueError("actual finite operator deadline required")
        if not 0 < deadline_unix - time.time() <= 172800:
            raise ValueError("operator deadline expired or exceeds this library's finite48h ceiling")
        source_path = Path(bound_kernel_module.__file__)
        if sha(source_path) != CONDITIONAL_KERNEL_SHA256:
            raise ValueError("the actual imported explicit two-CLIP kernel source differs")
        if bound_kernel_module.IDENTITY != "conditional_generation_two_clip_no_t5_v1":
            raise ValueError("wrong explicit conditioning kernel identity")
        self.pipeline = pipeline
        self.kernel_module = bound_kernel_module
        self.kernel = bound_kernel_module.SD35ConditionalTwoClipKernel(
            pipeline, bound_kernel_module.ConditionalTwoClipConfig())
        self.deadline_unix = float(deadline_unix)
        self.prompt_pack = None
        self.condition_metadata = None
        self.output = None
        self.trace_path = None
        self._closed = False
        self._started = False
        self._completed = False
        self._round = -1
        self._in_round = 0
        self._scheduler = None
        self._updates = None
        self._sample_id = None
        self._shape_device_dtype = None
        self.transformer_attempted = self.transformer_completed = 0
        self.scheduler_attempted = self.scheduler_completed = 0
        self.rows = []
        self.fixed_state_files = []
        self.counter_semantics_observed = None

    def _live(self):
        if self._closed:
            raise RuntimeError("conditional callback is closed")
        if time.time() >= self.deadline_unix:
            raise TimeoutError("actual conditional callback operator deadline expired")

    def encode_condition(self, device):
        """One actual encode; recover original four-tuple by lossless chunk views."""
        self._live()
        if self.prompt_pack is not None or self._started:
            raise RuntimeError("condition already encoded; no extra condition pass")
        self.condition_metadata = self.kernel.encode_condition(device)
        # The source-bound946c kernel stores [negative,positive] concatenations.
        # These are split views of ACTUAL returned values; no embedding rerun,
        # zero replacement, cast or new model inference occurs here.
        joined, joined_pooled = self.kernel._condition
        negative, positive = joined.chunk(2)
        negative_pooled, positive_pooled = joined_pooled.chunk(2)
        self.prompt_pack = (positive, negative, positive_pooled, negative_pooled)
        return self.prompt_pack, self.condition_metadata

    def begin_run(self, pipeline, prompt_pack, config, sample, output):
        self._live()
        if self._started or pipeline is not self.pipeline or prompt_pack is not self.prompt_pack:
            raise RuntimeError("one new run must consume this callback's actual condition tuple")
        if self.prompt_pack is None:
            raise RuntimeError("real condition must be encoded before new training")
        for name in ("guidance_scale", "seed", "prompt", "negative_prompt", "num_inference_steps"):
            if getattr(config, name) != getattr(self.kernel.config, name):
                raise ValueError("actual conditioning differs from the fixed attack config: " + name)
        if config.adversarial_updates not in (25, 75, 125) or config.outer_rounds != config.adversarial_updates // 25:
            raise ValueError("only independent fresh25/75/125 (outer1/3/5), never checkpoint continuation")
        self.kernel_module.require_two_clip_component_presence(pipeline)
        self.output = Path(output)
        if not self.output.is_dir() or any(self.output.iterdir()):
            raise FileExistsError("conditional run requires the same NEW empty case output")
        self.trace_path = self.output / "conditional_generation_trace.jsonl"
        self._updates = config.adversarial_updates
        self._sample_id = sample["sample_id"]
        self._started = True

    def begin_round(self, scheduler, outer_index):
        self._live()
        if not self._started or self._completed or type(outer_index) is not int or outer_index != self._round + 1:
            raise RuntimeError("conditional outer round order differs")
        if self._round >= 0 and self._in_round != 25:
            raise RuntimeError("cannot reset a partly executed generation round")
        if outer_index >= self._updates // 25:
            raise RuntimeError("declared generation round budget exceeded")
        if type(scheduler) is not self.kernel._scheduler_type:
            raise TypeError("actual original-loop FlowMatch scheduler required")
        if (len(scheduler.timesteps) != 100 or len(scheduler.sigmas) != 101 or
                scheduler.begin_index != 75 or scheduler.step_index is not None):
            raise RuntimeError("actual scheduler must be freshly reset100 with begin75 before generation")
        self._scheduler = scheduler
        self._round = outer_index
        self._in_round = 0
        self._shape_device_dtype = None

    def __call__(self, pipeline, latent, timestep, prompt_pack, guidance_scale):
        self._live()
        import torch
        if (not self._started or self._completed or self._round < 0 or self._in_round >= 25 or
                pipeline is not self.pipeline or pipeline.scheduler is not self._scheduler or
                prompt_pack is not self.prompt_pack or guidance_scale != 4.5):
            raise RuntimeError("unbound, repeated or out-of-order conditional generation call")
        if (not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (1, 16, 64, 64) or
                not latent.is_floating_point() or not bool(torch.isfinite(latent).all())):
            raise ValueError("finite actual1x16x64x64 latent required")
        shape_binding = (tuple(latent.shape), latent.device, latent.dtype)
        if self._shape_device_dtype is None:
            self._shape_device_dtype = shape_binding
        if shape_binding != self._shape_device_dtype or latent.dtype != pipeline.transformer.dtype:
            raise ValueError("round latent shape/device/dtype changed")
        expected = self._scheduler.timesteps[75 + self._in_round]
        if (not isinstance(timestep, torch.Tensor) or timestep.ndim != 0 or
                timestep.device != expected.device or timestep.dtype != expected.dtype or not torch.equal(timestep, expected)):
            raise ValueError("actual next scheduled timestep differs")
        expected_before = None if self._in_round == 0 else 75 + self._in_round
        before_index = self._scheduler.step_index
        if before_index != expected_before:
            raise RuntimeError("actual scheduler index before step differs")
        positive, negative, pooled, negative_pooled = self.prompt_pack
        hook_before = self.kernel.counts.module_forward_completed["transformer"]
        # Exactly the official SD3 operator order; never call a legacy denoiser.
        with torch.no_grad():
            model_input = torch.cat((latent.detach(), latent.detach()))
            condition = torch.cat((negative, positive))
            pooled_condition = torch.cat((negative_pooled, pooled))
            self.transformer_attempted += 1
            raw_prediction = pipeline.transformer(
                hidden_states=model_input, timestep=timestep.expand(model_input.shape[0]),
                encoder_hidden_states=condition, pooled_projections=pooled_condition,
                joint_attention_kwargs=None, return_dict=False)[0]
            self.transformer_completed += 1
            if (raw_prediction.shape != model_input.shape or not bool(torch.isfinite(raw_prediction).all()) or
                    self.kernel.counts.module_forward_completed["transformer"] != hook_before + 1):
                raise FloatingPointError("one actual finite Transformer CFG-batch prediction required")
            guided_prediction = self.kernel_module._cfg_merge(raw_prediction, guidance_scale)
            self.scheduler_attempted += 1
            generated = pipeline.scheduler.step(guided_prediction, timestep, latent.detach(), return_dict=False)[0]
            self.scheduler_completed += 1
        if generated.shape != latent.shape or not bool(torch.isfinite(generated).all()) or generated.requires_grad:
            raise FloatingPointError("actual no-grad scheduler output differs")
        after_index = self._scheduler.step_index
        if after_index != 76 + self._in_round:
            raise RuntimeError("actual scheduler index did not advance exactly one")
        if time.time() >= self.deadline_unix:
            raise TimeoutError("actual generation returned after the bounded operator deadline")
        global_step = len(self.rows) + 1
        descriptor = self.kernel_module.embedding_descriptor
        row = {"global_step": global_step, "outer_round": self._round + 1,
            "inner_step": self._in_round + 1, "sample_id": self._sample_id,
            "conditioning_mode": "two_clip_no_t5", "T5_execution": "not_executed",
            "actual_scheduler_begin_index": self._scheduler.begin_index,
            "actual_scheduler_step_index_before": before_index,
            "actual_scheduler_step_index_after": after_index,
            "actual_timestep": float(timestep),
            "actual_sigma_before": float(self._scheduler.sigmas[75 + self._in_round]),
            "actual_sigma_after": float(self._scheduler.sigmas[76 + self._in_round]),
            "actual_transformer_hook_delta": self.kernel.counts.module_forward_completed["transformer"] - hook_before,
            "actual_tensor_descriptors": {"input_latent": descriptor(latent),
                "raw_CFG_batch_prediction": descriptor(raw_prediction),
                "guided_prediction": descriptor(guided_prediction), "generated_latent": descriptor(generated)},
            "generation_no_grad": True, "attack_leaf_detach_and_two_VJPs_owned_by_original_derived_loop": True}
        if global_step in (1, 25, 500):
            state_path = self.output / ("conditional_generator_state_%04d.SERVER_ONLY.pt" % global_step)
            payload = {"metadata": row, "input_latent": latent.detach().cpu(),
                "raw_CFG_batch_prediction": raw_prediction.detach().cpu(),
                "guided_prediction": guided_prediction.detach().cpu(), "generated_latent": generated.detach().cpu(),
                "timestep": timestep.detach().cpu()}
            with state_path.open("xb") as f:
                torch.save(payload, f); f.flush(); os.fsync(f.fileno())
            state_binding = _binding(state_path)
            self.fixed_state_files.append(state_binding)
            row["fixed_server_only_state"] = state_binding
        # A failed later callback leaves the prior whole rows intact; no retry.
        with self.trace_path.open("x" if not self.rows else "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        self.rows.append(row)
        self._in_round += 1
        return generated, row

    def require_complete(self, snapshot, updates):
        self._live()
        if (not self._started or self._completed or updates != self._updates or
                self._round + 1 != updates // 25 or self._in_round != 25 or len(self.rows) != updates):
            raise RuntimeError("all new conditional rounds/steps have not actually returned")
        if (self.transformer_attempted != updates or self.transformer_completed != updates or
                self.scheduler_attempted != updates or self.scheduler_completed != updates):
            raise RuntimeError("all true Transformer/scheduler calls must finish once per update")
        self.counter_semantics_observed = validate_ledger_snapshot(snapshot, planned_budget(updates))
        hooks = self.kernel.counts.module_forward_completed
        expected = {"transformer": updates, "vae_encoder": 1 + updates // 25,
                    "vae_decoder": 1 + 2 * updates + updates // 25}
        if any(hooks[k] != v for k, v in expected.items()):
            raise RuntimeError("actual module hooks disagree with declared original VAE/Transformer calls")
        for name in ("text_encoder", "text_encoder_2"):
            if hooks[name] != self.condition_metadata["actual_module_forward_deltas"][name] or hooks[name] < 1:
                raise RuntimeError("actual CLIP encoding was repeated or absent")
        self._completed = True

    def observation(self):
        # After a wall-clock failure, read-only partial facts must remain
        # available; this method executes no model, update or scheduler call.
        if self._closed:
            raise RuntimeError("callback observation scope is closed")
        return {"schema": "conditional-derived-training-generation-observation-v1", "identity": IDENTITY,
            "scope": "new-generation callback plus actual module hooks, not independent scientific acceptance",
            "conditioning_mode": "two_clip_no_t5", "T5_execution": "not_executed",
            "full_T5_or_effect_equivalence_claim": False,
            "kernel_source_sha256": CONDITIONAL_KERNEL_SHA256,
            "original_training_source_sha256": ORIGINAL_TRAINING_MEMBER_SHA256,
            "original_SCI_manifest_sha256": ORIGINAL_SCI_MANIFEST_SHA256,
            "original_FP32_attack_decoder_and_posterior_mode_encoder_preserved": True,
            "kernel_start_round_generated_leaf_decode_leaf_NOT_called": True,
            "original_loop_scheduler_owned_by_original_derived_function": True,
            "condition": self.condition_metadata,
            "actual_application_calls": {"transformer_attempted": self.transformer_attempted,
                "transformer_completed": self.transformer_completed, "scheduler_attempted": self.scheduler_attempted,
                "scheduler_completed": self.scheduler_completed},
            "actual_module_forward_completed": dict(self.kernel.counts.module_forward_completed),
            "completed_generation_steps": len(self.rows), "declared_updates": self._updates,
            "callback_completion_checks_passed": self._completed,
            "counter_semantics_observed_unchanged": self.counter_semantics_observed,
            "server_only_fixed_generation_states": list(self.fixed_state_files),
            "generation_trace": _binding(self.trace_path) if self.trace_path is not None and self.trace_path.exists() else None,
            "four_source_gradient_scientific_acceptance": False,
            "root_lifecycle_GPU_or_process_admission_verified_by_library": False}

    def close(self):
        if self._closed:
            raise RuntimeError("callback already closed")
        self.kernel.close()
        self._closed = True


if __name__ == "__main__":
    raise SystemExit("Source-only library: requires a separately admitted caller; no automatic training CLI")
