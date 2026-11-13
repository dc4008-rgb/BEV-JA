#!/usr/bin/env python3
"""Persistent model loading, with independent sequential sample attack chains.

One invocation runs only its frozen20-case shard or the two-case diagnostic.
Victim models and SD3.5 weights load once. Per-case core resets latent, scheduler,
periodic controller and RNG; no previous case's patch is an initialization source. The output
root is new and batch retry/resume is deliberately not automatic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from sample_contract import IDENTITY, load_plan, record_for_sample
from source_bindings import verify_sources, verify_tree
from sample_training import (sha, write_json, reset_sample_rng,
                             load_verified_helpers, run_sample_optimization, configure_dual_backward_runtime)


def run_batch_cases(*, plan, output_root, records, parent, config, pipeline,
                    prompt_pack, adapters, depth_objective, device, dtype,
                    kernels, context_for, initial_for, runtime_provenance,
                    core=run_sample_optimization):
    """Use the same loaded models, but a fresh own-input trajectory for each case."""
    output_root = Path(output_root)
    if not output_root.is_dir() or (output_root / "samples").exists() or (output_root / "BATCH_RESULT.json").exists():
        raise FileExistsError("batch output is not a newly initialized run root")
    samples_root = output_root / "samples"
    samples_root.mkdir(exist_ok=False)
    started = time.monotonic()
    completed = []
    for index, sample in enumerate(plan["samples"]):
        record = record_for_sample(plan, sample, records)
        # A shared model object is deliberately retained; scientific trajectory
        # state is reset before context construction and again by the core.
        reset_sample_rng(plan["scientific_config"]["seed"])
        context = context_for(record)
        initial_small, initial_patch = initial_for(context)
        case_root = samples_root / sample["case_id"] / "train"
        case_root.mkdir(parents=True, exist_ok=False)
        try:
            result = core(parent=parent, config=config, output=case_root,
                pipeline=pipeline, prompt_pack=prompt_pack, initial_patch=initial_patch,
                initial_small=initial_small, sample=sample, context=context,
                adapters=adapters, depth_objective=depth_objective, device=device,
                dtype=dtype, eot_config=plan["scientific_config"]["eot"], kernels=kernels,
                plan_sha256=plan["_plan_sha256"], source_bindings=plan["source_bindings"],
                scientific_config=plan["scientific_config"])
            if (result.get("identity") != IDENTITY or result.get("status") != "train_optimization_complete"
                or result.get("executed_updates") != plan["budget"]["updates_per_sample"]
                or result.get("sample_id") != sample["sample_id"]
                or result.get("plan_sha256") != plan["_plan_sha256"]):
                raise ValueError("case did not complete its own frozen budget and identity")
            write_json(case_root / "runtime_provenance.json", {**runtime_provenance,
                "sample_id": sample["sample_id"], "case_id": sample["case_id"],
                "source_path": str(record.source_path), "original_plan_source_path": sample["source_path"],
                "source_sha256": sample["source_sha256"], "source_split": sample["source_split"],
                "executed_updates": result["executed_updates"],
                "models_loaded_once_for_batch": True,
                "trajectory_reset": "fresh_latent_scheduler_periodic_controller_rng_per_sample"})
            row = {"sample_id": sample["sample_id"], "case_id": sample["case_id"],
                "dataset": sample["dataset"], "source_split": sample["source_split"],
                "executed_updates": result["executed_updates"],
                "summary_path": str((case_root / "summary.json").relative_to(output_root)),
                "summary_sha256": sha(case_root / "summary.json"),
                "trace_sha256": result["trace_sha256"],
                "runtime_provenance_sha256": sha(case_root / "runtime_provenance.json"),
                "final_patch_binding": result["patch_bindings"]["final_patch.pt"]}
            completed.append(row)
            write_json(output_root / f"CASE_COMPLETED_{index + 1:04d}.json", {
                "identity": IDENTITY, "status": "case_completed",
                "completed_cases": len(completed), "planned_cases": len(plan["samples"]),
                "completed_updates": sum(c["executed_updates"] for c in completed),
                "plan_sha256": plan["_plan_sha256"], "case": row,
                "elapsed_seconds": time.monotonic() - started})
        except BaseException as error:
            failure = {"identity": IDENTITY, "sample_id": sample["sample_id"],
                "plan_sha256": plan["_plan_sha256"], "error": repr(error),
                "traceback": traceback.format_exc(), "test_data_opened": False}
            if not (case_root / "failure.json").exists():
                write_json(case_root / "failure.json", failure)
            write_json(output_root / "BATCH_FAILURE.json", {**failure,
                "completed_cases": completed, "automatic_retry": False})
            raise
        del context, initial_small, initial_patch, result
    total = sum(case["executed_updates"] for case in completed)
    if len(completed) != plan["budget"]["sample_count"] or total != plan["budget"]["total_updates"]:
        raise RuntimeError("batch aggregate budget did not complete exactly")
    payload = {"identity": IDENTITY, "status": "batch_training_complete",
        "plan_sha256": plan["_plan_sha256"], "source_bindings": plan["source_bindings"],
        "shard_id": plan.get("shard_id"), "budget": plan["budget"],
        "completed_cases": completed, "executed_updates": total,
        "models_loaded_once_for_batch": True, "test_data_opened": False,
        "optimization_on_original_validation": plan["optimization_on_original_validation"],
        "held_out_image_generalization": False,
        "elapsed_seconds": time.monotonic() - started, "resume_enabled": False}
    write_json(output_root / "BATCH_RESULT.json", payload)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan", "output-root", "child", "r2-source", "v2-source", "clip-source"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--sd35-snapshot", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = load_plan(args.plan)
    if args.output_root.exists() or any(p.is_symlink() for p in (args.output_root, *args.output_root.absolute().parents)):
        raise FileExistsError("new nonsymlink output root required; no retry/resume/overwrite")
    package = verify_tree(Path(__file__).resolve().parent, args.package_manifest_sha256)
    paths = verify_sources(plan, child=args.child, r2_source=args.r2_source,
        v2_source=args.v2_source, clip_source=args.clip_source)
    hostname = socket.gethostname()
    gpu_identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader"], text=True).strip()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
        "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU already has a compute process: " + active)
    existing = args.output_root.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if plan["status"] == "frozen_formal_shard":
        admission = plan["admission_binding"]["document"]
        if (admission["package_sha256"] != args.package_manifest_sha256
            or admission["checked_hostname"] != hostname or admission["gpu_uuid"] not in gpu_identity
            or admission["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            or admission["driver"] not in gpu_identity):
            raise ValueError("formal admission differs from live package/host/GPU")
        if free < admission["required_disk_bytes"]:
            raise RuntimeError("live free disk is below the formal admission requirement")
    inputs, parent, clip, kernels = load_verified_helpers(paths)
    import torch
    import torchvision
    import diffusers
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("real batch training requires CUDA")
    if (torch.__version__.split("+")[0], torchvision.__version__.split("+")[0], diffusers.__version__.split("+")[0]) != ("2.8.0", "0.23.0", "0.38.0"):
        raise RuntimeError("unexpected frozen runtime versions")
    clip.configure_determinism()
    dual_runtime = configure_dual_backward_runtime()
    config = parent.StatefulOursConfig(**{key: plan["scientific_config"][key] for key in clip.CONFIG if key != "outer_rounds"},
        outer_rounds=plan["scientific_config"]["outer_rounds"])
    pr = paths["child"] / "parent_v17"
    manifest = pr / "01_公共数据集/runtime_v1/train_validation_manifest.csv"
    records = parent.training_records(manifest, args.source_prefix, args.source_root)
    selected = [record_for_sample(plan, sample, records) for sample in plan["samples"]]
    model_manifest = pr / "02_模型/model_manifest.json"
    car_path, mask_path = (pr / "01_公共数据集" / n for n in ("car_00.jpg", "car_00_mask.jpg"))
    assets, depth_assets, ready = clip.preflight_training_assets(parent, selected,
        model_manifest, car_path, mask_path, args.weights_root, 1.0)
    inventory_path = pr / "00_冻结协议/sd35_snapshot_inventory.json"
    snapshot_path = args.sd35_snapshot or Path(json.loads(inventory_path.read_text())["snapshot_path"])
    snapshot = parent.validate_snapshot(snapshot_path, inventory_path, inputs.SD_MANIFEST_SHA)
    if not ready or snapshot["status"] != "file_hashes_verified":
        raise RuntimeError("full shard sources/weights/car/SD3.5 failed asset verification")
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    car, mask = parent.load_car(car_path, mask_path, device)
    # These constructors run exactly once; the per-case loop never calls them.
    adapters, depth_objective = clip.build_training_adapters(parent, device,
        model_manifest, args.weights_root, 1.0)
    from depth_hinge_objective import DepthObjective
    depth_objective = DepthObjective(parent, depth_objective.adapters,
        beta=plan["scientific_config"]["depth_beta"], depth_target=plan["scientific_config"]["switches"]["depth_target"],
        epsilon=plan["scientific_config"]["controller_config"]["epsilon"])
    pipeline = parent.build_pipeline(snapshot_path, device, dtype)
    prompt_pack = parent.encode_prompt(pipeline, config)
    verified = set()
    context_for = lambda record: parent.build_context(record, car, mask, device, verified)

    def initial_for(context):
        small = parent.extract_road_patch(context.background, context.road_quad)
        return small, F.interpolate(small, size=(512, 512), mode="bilinear", align_corners=False)

    runtime = {"identity": IDENTITY, "hostname": hostname, "gpu_identity": gpu_identity,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "dual_backward_runtime": dual_runtime, "torch": torch.__version__, "torchvision": torchvision.__version__, "diffusers": diffusers.__version__,
        "package_verification": package, "source_verification": paths["verified"],
        "training_manifest_sha256": sha(manifest), "model_manifest_sha256": sha(model_manifest),
        "plan_sha256": plan["_plan_sha256"], "dtype": str(dtype), "real_cuda_run": True,
        "source_path_mapping": {"source_prefix": str(args.source_prefix) if args.source_prefix else None,
            "source_root": str(args.source_root) if args.source_root else None},
        "assets": assets, "depth_assets": depth_assets, "sd35_snapshot": snapshot,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / "BATCH_RUNTIME.json", runtime)
    write_json(args.output_root / "PLAN_COPY.json", {k: v for k, v in plan.items() if k != "_plan_sha256"})
    return run_batch_cases(plan=plan, output_root=args.output_root, records=records,
        parent=parent, config=config, pipeline=pipeline, prompt_pack=prompt_pack,
        adapters=adapters, depth_objective=depth_objective, device=device, dtype=dtype,
        kernels=kernels, context_for=context_for, initial_for=initial_for, runtime_provenance=runtime)


if __name__ == "__main__":
    main()
