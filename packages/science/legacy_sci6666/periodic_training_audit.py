"""CPU-only audit of the independent periodic candidate's saved training.

No victim models, source images, optimizations, or VJPs are executed. Controller
replay and evidence ownership are exact. The two saved task-gradient probes are
recombined on CPU with explicitly bounded FP32 device-reduction tolerances;
recorded FP32 clip scale is used for exact BF16 element arithmetic. This is not
a zero-tolerance CPU/CUDA claim or an attack-effect acceptance criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

import torch

from artifact_io import load_torch_lossless, state_fingerprint
from periodic_controller import PeriodicConfig, PeriodicController
from task_gradient_balance import combine_task_gradients

ROOT = Path(__file__).resolve().parent
IDENTITY = "ours_periodic_foa_v1"
TENSOR_RTOL = 1e-5
TENSOR_ATOL = 1e-12
NORM_RTOL = 1e-5
COSINE_ATOL = 8 * torch.finfo(torch.float32).eps
DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.float64, torch.bfloat16)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "SHA256 required")
    return value


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "plain evidence path required")
    return path


def read_json(path):
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value
    return json.loads(Path(path).read_text(encoding="utf8"), parse_float=finite,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON " + value)))


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if not line:
            continue
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON " + value)))
        require(isinstance(row, dict), "JSONL rows must be objects")
        rows.append(row)
    return rows


def checked_file(root, relative, binding=None):
    root = plain(root)
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts, "unsafe evidence member")
    path = plain(root / relative)
    require(path.is_file(), "missing evidence member: " + str(relative))
    if binding is not None:
        require(path.stat().st_size == binding["bytes"] and file_sha(path) == digest(binding["sha256"]),
                "evidence bytes/SHA differs: " + str(relative))
    return path


def _native():
    evaluation = importlib.import_module("sample_evaluate")
    require(Path(evaluation.__file__).resolve() == ROOT / "sample_evaluate.py", "native evaluator module shadowed")
    return evaluation


def _controller(config):
    c, switches = config["controller_config"], config["switches"]
    return PeriodicController(PeriodicConfig(**{key: c[key] for key in
        ("model_mu", "task_mu", "model_temperature", "task_temperature", "ema_decay", "ratio_delta", "ratio_min", "ratio_max")},
        period=switches["period"], model_adaptive=switches["model_adaptive"],
        task_adaptive=switches["task_adaptive"], ema=switches["ema"]))


def quantized_pair(values, dtype):
    require(dtype in DTYPES, "unreviewed task-loss dtype")
    require(isinstance(values, (list, tuple)) and len(values) == 2, "two weights required")
    result = torch.tensor(values, dtype=DTYPES[dtype])
    require(bool(torch.isfinite(result).all()), "nonfinite weight")
    return result.tolist()


def replay_case_trace(trace, cycles, config, sample, updates):
    """Independently reset/replay every outer; no prior-case state is accepted."""
    require(type(updates) is int and updates in (25, 500) and len(trace) == updates, "incomplete fixed update trace")
    require(config["outer_rounds"] == updates // 25, "outer budget differs")
    controller = _controller(config)
    expected_cycles = []
    for index, row in enumerate(trace):
        step, outer, inner = index + 1, index // 25, index % 25 + 1
        if inner == 1:
            controller.reset_outer(outer)
        expected = controller.weights()
        require(row.get("global_step") == step and row.get("outer_round") == outer + 1 and row.get("inner_step") == inner,
                "trace step/outer order differs")
        require(all(row.get(key) == sample[source] for key, source in
                    (("sample_id", "sample_id"), ("case_id", "case_id"), ("dataset", "dataset"), ("split", "source_split"))),
                "trace belongs to another sample")
        require(row.get("switches") == config["switches"] and row.get("controller_config") == config["controller_config"],
                "trace controller configuration differs")
        require(row.get("current_period_weights") == expected, "period weights differ or changed before the next cycle")
        dtype = row.get("task_loss_dtype")
        require(row.get("detection_model_weights") == quantized_pair(expected["detector"], dtype) and
                row.get("adaptive_weights") == quantized_pair(expected["detector"], dtype) and
                row.get("depth_model_weights") == quantized_pair(expected["depth"], dtype) and
                row.get("task_weights") == expected["task"], "logged model/task weights differ from current period")
        require(row.get("autograd_calls_this_step") == 2 and
                row.get("weighted_total_loss_role") == "detached_diagnostic_not_third_backward", "two-VJP branch differs")
        require(row.get("noise_seed") == config["seed"] + 1000 * outer and
                row.get("eot_seed") == config["seed"] + 1000 * step, "noise/EOT seed schedule differs")
        det, depth = row["detection_deficit_details"], row["depth_deficit_details"]
        require(depth.get("detached") is True and depth.get("extra_clipping") is False,
                "depth control must be detached and un-clipped")
        actual_cycle = controller.observe(inner, det["deficits"], depth["deficits"])
        require(row.get("cycle_summary") == actual_cycle, "cycle summary/member/current/next history differs")
        if actual_cycle is not None:
            expected_cycles.append(actual_cycle)
    require(cycles == expected_cycles, "cycle_trace differs from replay or omits a cycle")
    return {"updates": updates, "statistical_cycles": len(expected_cycles),
            "weight_refreshes_used": sum(int(c["applied_next"]) for c in expected_cycles),
            "controller_replay_exact": True, "outer_reset_verified": True,
            "period_weights_fixed_until_next_cycle": True, "cross_case_controller_reused": False}


def validate_payload_metadata(payload, plan, sample):
    metadata = payload.get("metadata", {})
    expected = {key: sample[key] for key in ("sample_id", "case_id", "dataset", "source_split", "source_sha256")}
    expected.update(identity=IDENTITY, plan_sha256=plan["_plan_sha256"], source_bindings=plan["source_bindings"], test_data_opened=False)
    for key, value in expected.items():
        require(type(metadata.get(key)) is type(value) and metadata.get(key) == value, "retained state metadata differs: " + key)


def _tensor(value, label):
    require(isinstance(value, torch.Tensor) and value.numel() and value.is_floating_point() and
            value.device.type == "cpu" and bool(torch.isfinite(value).all()), "finite CPU tensor required: " + label)
    return value


def tensor_close(actual, expected, label):
    _tensor(actual, label); _tensor(expected, label)
    require(actual.shape == expected.shape and actual.dtype == expected.dtype, "tensor shape/dtype differs: " + label)
    require(torch.allclose(actual, expected, rtol=TENSOR_RTOL, atol=TENSOR_ATOL), "CPU/device tensor recomputation differs: " + label)
    if not bool(torch.count_nonzero(expected)):
        require(torch.equal(actual, expected), "exact-zero scientific branch differs: " + label)


def number_close(actual, expected, label, *, atol=0.):
    require(type(actual) in (int, float) and type(expected) in (int, float) and
            math.isfinite(actual) and math.isfinite(expected), "finite numeric diagnostic required: " + label)
    require(abs(actual - expected) <= atol + NORM_RTOL * abs(expected), "CPU/device scalar recomputation differs: " + label)


def details_close(actual, expected, label="task_allocation"):
    """Branches/keys/types exact; only recorded FP32 arithmetic has tolerance."""
    require(type(actual) is type(expected), "diagnostic type differs: " + label)
    if isinstance(actual, dict):
        require(set(actual) == set(expected), "diagnostic fields differ: " + label)
        for key in actual:
            details_close(actual[key], expected[key], label + "." + key)
    elif isinstance(actual, list):
        require(len(actual) == len(expected), "diagnostic length differs: " + label)
        for index, (a, b) in enumerate(zip(actual, expected)):
            details_close(a, b, label + "[" + str(index) + "]")
    elif type(actual) is float:
        atol = COSINE_ATOL if "cosine" in label or "first_order_margins" in label else 0.
        number_close(actual, expected, label, atol=atol)
    else:
        require(actual == expected, "scientific branch differs: " + label)


def audit_math_state(state, row, config):
    """Recombine already saved VJPs and audit arithmetic; never calls autograd."""
    gd = _tensor(state["detection_task_gradient"], "detection_task_gradient")
    gp = _tensor(state["depth_task_gradient"], "depth_task_gradient")
    saved = _tensor(state["joint_gradient"], "joint_gradient")
    z = _tensor(state["latent_variable"], "latent_variable")
    require(state.get("switches") == config["switches"] and
            state.get("current_period_weights") == row["current_period_weights"] and
            state.get("requested_task_priorities") == row["task_weights"], "fixed arithmetic state controller differs")
    for key, detail in (("detection_deficits", "detection_deficit_details"), ("depth_deficits", "depth_deficit_details")):
        value = _tensor(state[key], key)
        require(value.shape == (2,) and value.dtype == torch.float32 and value.tolist() == row[detail]["deficits"],
                "retained control deficits differ from trace")
    switches = config["switches"]
    rebuilt, allocation = combine_task_gradients(gd, gp, state["requested_task_priorities"],
        task_balance=switches["task_balance"], conflict_guard=switches["conflict_guard"],
        norm_match=switches["norm_match"], n_min=config["controller_config"]["n_min"])
    tensor_close(rebuilt, saved, "joint_gradient")
    details_close(allocation, state["task_allocation"])
    require(state["task_allocation"] == row["task_allocation"], "state/trace task diagnostics differ")
    update = row["update_statistics"]
    require(update.get("coefficient") == config["gradient_coefficient"] == 100000. and
            update.get("lr") == config["latent_lr"] == .05 and
            update.get("max_norm") == config["clip_max_norm"] == 120., "frozen clip/update settings differ")
    require(update.get("clip_stage") == "after_coefficient_before_lr" and
            update.get("norm_scope") == "all_elements_of_combined_amplified_latent_gradient", "clip stage/scope differs")
    with torch.no_grad():
        cast = saved.detach().to(z)
        amplified = config["gradient_coefficient"] * cast
        for name, value in (("cast_gradient", cast), ("amplified_gradient", amplified)):
            require(torch.equal(value, _tensor(state[name], name)), "exact element arithmetic differs: " + name)
        before = float(torch.linalg.vector_norm(amplified.float().reshape(-1)))
        number_close(before, update["norm_before"], "clip.norm_before")
        clipped = update["norm_before"] > 120.
        require(type(update.get("clipped")) is bool and update["clipped"] == clipped, "recorded clip branch differs from original FP32 norm")
        original_norm = torch.tensor(update["norm_before"], dtype=torch.float32)
        scale = torch.tensor(120., dtype=torch.float32) / original_norm if clipped else torch.tensor(1., dtype=torch.float32)
        require(float(scale) == update["clip_scale"], "original FP32 clip-scale arithmetic differs")
        used = (amplified.float() * scale).to(z.dtype) if clipped else amplified
        proposed = config["latent_lr"] * used
        candidate = (z.detach() - proposed).detach()
        realized = z.detach().float() - candidate.float()
        for name, value in (("clipped_gradient", used), ("proposed_update", proposed),
                            ("candidate_latent", candidate), ("realized_delta_float32", realized)):
            require(torch.equal(value, _tensor(state[name], name)), "exact element arithmetic differs: " + name)
        number_close(float(torch.linalg.vector_norm(used.float().reshape(-1))), update["norm_after"], "clip.norm_after")
        rms = float(torch.linalg.vector_norm(realized.reshape(-1))) / math.sqrt(z.numel())
        number_close(rms, update["realized_update_rms"], "clip.realized_update_rms")
    return {"global_step": row["global_step"], "two_saved_vjps_recombined_without_new_vjp": True,
            "gradient_and_task_diagnostics_within_declared_tolerance": True,
            "clip_element_arithmetic_exact_using_recorded_fp32_scale": True,
            "cpu_clip_norms_within_declared_tolerance": True}


def audit_training(runs_root, plan):
    from sample_contract import validate_plan, planned_budget
    validate_plan(plan)
    digest(plan["_plan_sha256"])
    root = plain(runs_root)
    native = _native()
    observed = {}
    def observe(path):
        path = plain(path)
        binding = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}
        observed[str(path)] = binding
        return binding
    batch_path = checked_file(root, "BATCH_RESULT.json")
    batch_binding = observe(batch_path)
    batch = read_json(batch_path)
    require(batch.get("identity") == IDENTITY and batch.get("status") == "batch_training_complete" and
            batch.get("plan_sha256") == plan["_plan_sha256"] and batch.get("source_bindings") == plan["source_bindings"] and
            batch.get("budget") == plan["budget"] and batch.get("executed_updates") == plan["budget"]["total_updates"] and
            batch.get("test_data_opened") is False and batch.get("resume_enabled") is False,
            "bound complete batch/plan/budget differs")
    members = batch.get("completed_cases", [])
    require(len(members) == len(plan["samples"]) and [m.get("sample_id") for m in members] ==
            [s["sample_id"] for s in plan["samples"]], "batch omits/reorders/duplicates a prescribed sample")
    reports = []
    updates = plan["budget"]["updates_per_sample"]
    for index, (sample, member) in enumerate(zip(plan["samples"], members), 1):
        require(member.get("case_id") == sample["case_id"] and member.get("dataset") == sample["dataset"] and
                member.get("source_split") == sample["source_split"] and member.get("executed_updates") == updates,
                "batch case identity/budget differs")
        expected_relative = "samples/" + sample["case_id"] + "/train/summary.json"
        require(member.get("summary_path") == expected_relative, "summary routing is not its own sample")
        summary_path = checked_file(root, expected_relative)
        require(observe(summary_path)["sha256"] == digest(member["summary_sha256"]), "batch summary SHA differs")
        summary = native.validate_summary(plan, sample, read_json(summary_path))
        require(summary.get("switches") == plan["scientific_config"]["switches"] and summary.get("resume_enabled") is False and
                summary.get("fresh_sample_initialization") is True and summary.get("no_cross_sample_state") is True,
                "case switches/fresh initialization differs")
        run = summary_path.parent
        receipt_path = checked_file(root, "CASE_COMPLETED_%04d.json" % index)
        observe(receipt_path)
        receipt = read_json(receipt_path)
        require(receipt.get("identity") == IDENTITY and receipt.get("status") == "case_completed" and
                receipt.get("plan_sha256") == plan["_plan_sha256"] and receipt.get("completed_cases") == index and
                receipt.get("planned_cases") == len(members) and receipt.get("completed_updates") == index * updates and
                receipt.get("case") == member, "case completion receipt detached from batch")
        runtime_path = checked_file(run, "runtime_provenance.json")
        require(observe(runtime_path)["sha256"] == digest(member["runtime_provenance_sha256"]), "runtime provenance SHA differs")
        runtime = read_json(runtime_path)
        require(runtime.get("identity") == IDENTITY and runtime.get("sample_id") == sample["sample_id"] and
                runtime.get("case_id") == sample["case_id"] and runtime.get("source_sha256") == sample["source_sha256"] and
                runtime.get("source_split") == sample["source_split"] and runtime.get("executed_updates") == updates and
                runtime.get("trajectory_reset") == "fresh_latent_scheduler_periodic_controller_rng_per_sample", "case runtime owner/reset differs")
        manifest_path = checked_file(run, "state_evidence/MANIFEST.json")
        require(observe(manifest_path)["sha256"] == digest(summary["state_evidence_manifest_sha256"]), "state manifest SHA differs")
        manifest = native.validate_state_manifest(plan, sample, read_json(manifest_path))
        trace_path, cycle_path = checked_file(run, "attack_trace.jsonl"), checked_file(run, "cycle_trace.jsonl")
        require(observe(trace_path)["sha256"] == digest(summary["trace_sha256"]) == digest(member["trace_sha256"]), "trace SHA differs")
        require(observe(cycle_path)["sha256"] == digest(summary["cycle_trace_sha256"]), "cycle trace SHA differs")
        trace, cycles = read_jsonl(trace_path), read_jsonl(cycle_path)
        replay = replay_case_trace(trace, cycles, plan["scientific_config"], sample, updates)
        require(summary.get("statistical_cycles") == replay["statistical_cycles"] and
                summary.get("weight_refreshes_used") == replay["weight_refreshes_used"], "summary cycle counts differ")
        phase_index = {(e["phase"], e["global_step"]): e for e in manifest["entries"]}
        for row in trace:
            require(set(row.get("state_bindings", {})) == {"before_denoise", "pre_backward", "pre_update", "post_update"}, "trace phase hash set differs")
            for phase, bound in row["state_bindings"].items():
                require(bound == phase_index[phase, row["global_step"]]["state_fingerprint"], "trace/manifest phase hash differs")
        math_reports = []
        for entry in [*manifest["outer_boundary_checkpoints"], *manifest["fixed_update_math_checks"]]:
            path = checked_file(run, entry["path"], entry)
            observe(path)
            payload = load_torch_lossless(path)
            validate_payload_metadata(payload, plan, sample)
            require(payload.get("phase") == entry["phase"] and payload.get("global_step") == entry["global_step"] and
                    state_fingerprint(payload.get("state")) == entry["state_fingerprint"], "retained state fingerprint/phase differs")
            if entry["retention_kind"] == "fixed_update_math_probe":
                math_reports.append(audit_math_state(payload["state"], trace[entry["global_step"] - 1], plan["scientific_config"]))
        reports.append({"sample_id": sample["sample_id"], "case_id": sample["case_id"],
                        "summary_sha256": member["summary_sha256"], "runtime_provenance_sha256": member["runtime_provenance_sha256"],
                        "call_budget_per_sample": planned_budget(updates), "phase_fingerprints": len(manifest["entries"]),
                        **replay, "fixed_update_math_checks": math_reports})
    for binding in observed.values():
        checked_file(Path(binding["path"]).parent, Path(binding["path"]).name, binding)
    return {"schema": "ours-periodic-training-audit-v1", "identity": IDENTITY, "status": "periodic_training_audit_passed",
            "plan_sha256": plan["_plan_sha256"], "source_bindings": plan["source_bindings"],
            "batch_result_binding": batch_binding, "sample_count": len(reports), "executed_updates": sum(r["updates"] for r in reports),
            "cases": reports, "artifact_bindings": list(observed.values()),
            "source_code_sha256": {n: file_sha(ROOT / n) for n in ("periodic_training_audit.py", "sample_evaluate.py", "sample_contract.py", "periodic_controller.py", "task_gradient_balance.py", "artifact_io.py")},
            "tolerances": {"controller_and_sha": 0, "gradient_tensor_rtol": TENSOR_RTOL, "gradient_tensor_atol": TENSOR_ATOL,
                           "norm_relative": NORM_RTOL, "norm_absolute": 0., "cosine_margin_absolute": COSINE_ATOL},
            "clip_arithmetic_uses_recorded_original_fp32_norm_scale": True,
            "cross_cpu_cuda_zero_tolerance_claim": False, "native_evaluation_raw_metric_audit_replaced": False,
            "all_intermediate_gradients_recovered_or_recomputed": False, "models_loaded_or_called": False,
            "source_images_opened": False, "new_vjp_calls": 0, "new_optimization": False,
            "test_data_opened": False, "historical_results_modified": False,
            "fixed_final_selection": True, "efficacy_or_full_gpu_acceptance_claim": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "runs-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("child", "r2-source", "v2-source", "clip-source"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    from sample_contract import load_plan
    plan = load_plan(args.plan)
    sources = (args.child, args.r2_source, args.v2_source, args.clip_source)
    require(all(sources) or not any(sources), "all four explicit source paths required together")
    if all(sources):
        from source_bindings import verify_sources
        paths = verify_sources(plan, child=args.child, r2_source=args.r2_source, v2_source=args.v2_source, clip_source=args.clip_source)
        sys.path.insert(0, str(paths["r2_source"]))
    output = plain(args.output)
    require(output.parent.is_dir() and not output.exists(), "exclusive new audit output required")
    report = audit_training(args.runs_root, plan)
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": file_sha(output),
                      "sample_count": report["sample_count"], "executed_updates": report["executed_updates"]}), flush=True)
    return report


if __name__ == "__main__":
    main()
