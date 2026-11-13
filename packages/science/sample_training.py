#!/usr/bin/env python3
"""Independent single-input SD3.5 joint optimization with periodic FOA and guarded task balance.

This candidate reuses verified scientific helpers, never the historical runner's
500-step schedule, OFF comparison gate, recorder, output roots, or run identity.
Only one explicitly selected PLAN sample is attacked by each CLI invocation.
CPU fakes can exercise the real orchestration without loading victim weights.
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
IDENTITY = "ours_complete_eot1_parallel200_v1"
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
    if type(updates) is not int or updates not in (25, 500):
        raise ValueError("per-sample updates must be exactly 25 or 500")
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan", "child", "r2-source", "v2-source", "clip-source", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--sd35-snapshot", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from sample_contract import load_plan, select_sample, record_for_sample
    from source_bindings import verify_sources
    plan = load_plan(args.plan)
    sample = select_sample(plan, args.sample_id)
    if args.output.name != "train" or args.output.parent.name != sample["case_id"] or args.output.parent.parent.name != "samples":
        raise ValueError("--output must be output_root/samples/<case_id>/train")
    if args.output.exists() or any(path.is_symlink() for path in (args.output, *args.output.absolute().parents)):
        raise FileExistsError("output exists or is a symlink; no retry/resume/overwrite")
    paths = verify_sources(plan, child=args.child, r2_source=args.r2_source,
        v2_source=args.v2_source, clip_source=args.clip_source)
    inputs, parent, clip, kernels = load_verified_helpers(paths)
    import torch
    import torchvision
    import diffusers
    if not torch.cuda.is_available():
        raise RuntimeError("real sample-specific training requires CUDA; CPU fakes are tests only")
    if (torch.__version__.split("+")[0], torchvision.__version__.split("+")[0], diffusers.__version__.split("+")[0]) != ("2.8.0", "0.23.0", "0.38.0"):
        raise RuntimeError("unexpected frozen runtime versions")
    configure_dual_runtime(torch)
    clip.configure_determinism()
    updates = plan["budget"]["updates_per_sample"]
    fixed = {key: plan["scientific_config"][key] for key in CONFIG}
    config = parent.StatefulOursConfig(**fixed, outer_rounds=updates // 25)
    pr = paths["child"] / "parent_v17"
    manifest = pr / "01_公共数据集/runtime_v1/train_validation_manifest.csv"
    records = parent.training_records(manifest, args.source_prefix, args.source_root)
    record = record_for_sample(plan, sample, records)
    model_manifest = pr / "02_模型/model_manifest.json"
    car_path, mask_path = (pr / "01_公共数据集" / name for name in ("car_00.jpg", "car_00_mask.jpg"))
    assets, depth_assets, ready = clip.preflight_training_assets(parent, [record], model_manifest,
        car_path, mask_path, args.weights_root, 1.0)
    inventory_path = pr / "00_冻结协议/sd35_snapshot_inventory.json"
    snapshot_path = args.sd35_snapshot or Path(json.loads(inventory_path.read_text())["snapshot_path"])
    snapshot = parent.validate_snapshot(snapshot_path, inventory_path, inputs.SD_MANIFEST_SHA)
    if not ready or snapshot["status"] != "file_hashes_verified":
        raise RuntimeError("sample/road/car/victim/full SD3.5 assets failed verification")
    if sha(record.source_path) != sample["source_sha256"]:
        raise RuntimeError("actual selected image SHA differs from PLAN")
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    car, mask = parent.load_car(car_path, mask_path, device)
    context = parent.build_context(record, car, mask, device, set())
    initial_small = parent.extract_road_patch(context.background, context.road_quad)
    import torch.nn.functional as F
    initial_patch = F.interpolate(initial_small, size=(512, 512), mode="bilinear", align_corners=False)
    adapters, depth_objective = clip.build_training_adapters(parent, device, model_manifest, args.weights_root, 1.0)
    from depth_hinge_objective import DepthObjective
    depth_objective = DepthObjective(parent, depth_objective.adapters,
        beta=plan["scientific_config"]["depth_beta"], depth_target=plan["scientific_config"]["switches"]["depth_target"],
        epsilon=plan["scientific_config"]["controller_config"]["epsilon"])
    pipeline = parent.build_pipeline(snapshot_path, device, dtype)
    prompt_pack = parent.encode_prompt(pipeline, config)
    # Creation is delayed until all source, input, weight and runtime gates pass.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        summary = run_sample_optimization(parent=parent, config=config, output=args.output,
            pipeline=pipeline, prompt_pack=prompt_pack, initial_patch=initial_patch,
            initial_small=initial_small, sample=sample, context=context, adapters=adapters,
            depth_objective=depth_objective, device=device, dtype=dtype,
            eot_config=plan["scientific_config"]["eot"], kernels=kernels,
            plan_sha256=plan["_plan_sha256"], source_bindings=plan["source_bindings"],
            scientific_config=plan["scientific_config"])
        write_json(args.output / "runtime_provenance.json", {"identity": IDENTITY,
            "hostname": socket.gethostname(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "diffusers": diffusers.__version__,
            "gpu_identity": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader"], text=True).strip(),
            "source_path": str(record.source_path), "source_sha256": sha(record.source_path),
            "original_plan_source_path": sample["source_path"],
            "source_path_mapping": {"source_prefix": str(args.source_prefix) if args.source_prefix else None,
                "source_root": str(args.source_root) if args.source_root else None},
            "source_verification": paths["verified"], "assets": assets, "depth_assets": depth_assets,
            "sd35_snapshot": snapshot, "training_manifest_sha256": sha(manifest),
            "model_manifest_sha256": sha(model_manifest), "plan_sha256": plan["_plan_sha256"],
            "sample_id": sample["sample_id"], "executed_updates": summary["executed_updates"],
            "dtype": str(dtype), "real_cuda_run": True,
            "dual_backward_runtime": summary["dual_backward_runtime"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")})
    except BaseException as error:
        write_json(args.output / "failure.json", {"identity": IDENTITY, "sample_id": sample["sample_id"],
            "error": repr(error), "traceback": traceback.format_exc(), "test_data_opened": False})
        raise
    return summary


if __name__ == "__main__":
    main()
