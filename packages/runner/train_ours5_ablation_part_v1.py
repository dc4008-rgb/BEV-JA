#!/usr/bin/env python3
"""Declared Ours5 switch-only ablations, one original50 part per invocation.

The original SD3.5 numerical engine and all model/gradient math are imported
unchanged. The external owner admits current identity/resources and sequences
four successful original parts under one task deadline. This leaf records
completed case prefixes and retains partial failures without automatic retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
import traceback
from types import ModuleType

IDENTITY = "ours5-b3-full-nuscenes100-existing-switch-ablation-v1"
REQUEST_SCHEMA = "ours5-b3-full-nuscenes100-original50-training-request-v1"
RESUME_REQUEST_SCHEMA = "ours5-first2-original50-prefix-resume-training-request-v1"
CONDITIONING_MODE = "two_clip_no_t5_explicit"
PROTOCOL_SHA = "accee048bc48f068cd4f99684b0d3dfb99134d23afbf6a8f3d605d5d52ff0dc7"
PLAN_SHA = "661b32e3f309a2a5924edd270f31f79bbe9c4e25c044aac97b3f77ecb9c4eb1b"
SCI_SHA = "1940406257965a181371af89634a7378433392147719a4b16ae28d7f33c9081d"
LIBRARY_SHA = "fa7a76392b2218b70a9c46b1e187d4e7c9b01bda4f5b65a7fff9addf50879e1b"
ENGINE_SHA = "e7b6b31edb3050e7d9496b276d3fde21ffa68748b60efa7966059741642ca2ad"
KERNEL_SHA = "946c415b14c8119e70b389f3ec3c03444e51dd854742878cd0cf191495c263cb"
INVENTORY_SHA = "0c21d98782b4a88f524b9377d779d7143d51fea72f390f068f479caeed36b1dc"
OUTPUT_NAMESPACE = "ours5_b3_full_nuscenes100_20261105_001"
STAGES = {"formal200_part_training": "explicit_budget1_3_5_and_typed_previous_phase"}
HELPERS = {"child", "r2_source", "v2_source", "clip_source"}
SOURCE_FILENAMES = {"connectivity_library": "budget_gpu_component_v1.py",
    "kernel": "sd35_conditional_two_clip_v1.py", "derived_engine": "derived_budget_training_engine_v2.py"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute() and ".." not in Path(value).parts,
            "explicit absolute path without traversal required")
    return Path(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    path = Path(path)
    return {"path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": sha(path)}


def validate_descriptor(d):
    require(isinstance(d, dict) and set(d) == {"path", "bytes", "sha256"}, "closed whole-file descriptor required")
    absolute(d["path"])
    require(type(d["bytes"]) is int and d["bytes"] > 0, "nonempty file descriptor required")
    require(isinstance(d["sha256"], str) and re.fullmatch("[0-9a-f]{64}", d["sha256"]), "whole SHA256 required")


def checked_file(d):
    validate_descriptor(d)
    path = absolute(d["path"])
    require(path.is_file() and not path.is_symlink(), "regular nonsymlink control/source file required")
    before = path.stat()
    data = path.read_bytes()
    with path.open("rb") as f:
        os.fsync(f.fileno())
    again = path.read_bytes()
    after = path.stat()
    require(data == again and len(data) == d["bytes"] and hashlib.sha256(data).hexdigest() == d["sha256"],
            "bound file bytes changed")
    require(all(getattr(before, key) == getattr(after, key) for key in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "control file unstable")
    return data


def write_new(path, document):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return descriptor(path)


def load_bound_source(name, d):
    require(name not in sys.modules, "bound module name already exists")
    data = checked_file(d)
    module = ModuleType(name)
    module.__file__ = d["path"]
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(data, d["path"], "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def validate_request(q):
    keys = {"schema", "identity", "stage", "method_id", "node_id", "part_id", "budget_outer_rounds",
            "expected_hostname", "expected_boot_id", "deadline_unix", "protocol", "master_reference",
            "reference_plan", "science_package", "snapshot", "helper_roots", "weights_root", "source_mapping",
            "output_base", "output_directory", "source_files", "operator_admission", "task_scope", "storage_profile"}
    require(isinstance(q, dict), "ablation TRAIN request object required")
    resumed = q.get("schema") == RESUME_REQUEST_SCHEMA
    require(not resumed, "new b3-full-nuscenes100 task is fresh200, no prior case/prefix allowed")
    if resumed:
        keys.add("accepted_prefix_CPU")
    require(set(q) == keys, "closed ablation TRAIN request required")
    require(q["schema"] in (REQUEST_SCHEMA, RESUME_REQUEST_SCHEMA) and q["identity"] == IDENTITY,
            "current ablation request identity required")
    if resumed:
        require(q["part_id"] == 1, "only the observed original part1 prefix may be reused")
        validate_descriptor(q["accepted_prefix_CPU"])
    from ablation_contract_v1 import METHODS, PART_SHA, MASTER_SHA
    require(q["method_id"] in METHODS and q["node_id"] == METHODS[q["method_id"]]["node_id"], "declared method and node required")
    require(type(q["part_id"]) is int and q["part_id"] in range(4), "original part 0..3 required")
    require(q["budget_outer_rounds"] == 5 and q["stage"] == "formal200_part_training", "fresh five-round TRAIN only")
    require(re.fullmatch(r"autodl-container-[a-z0-9-]+", q["expected_hostname"]), "actual full hostname required")
    require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", q["expected_boot_id"]), "actual boot UUID required")
    require(type(q["deadline_unix"]) in (int, float) and math.isfinite(q["deadline_unix"]), "finite deadline required")
    for name in ("protocol", "master_reference", "reference_plan", "operator_admission", "task_scope"):
        validate_descriptor(q[name])
    require(q["protocol"]["sha256"] == PROTOCOL_SHA and q["master_reference"]["sha256"] == MASTER_SHA
            and q["reference_plan"]["sha256"] == PART_SHA[q["part_id"]], "frozen inputs required")
    require(set(q["science_package"]) == {"root", "manifest"}, "original SCI package required")
    absolute(q["science_package"]["root"]); validate_descriptor(q["science_package"]["manifest"])
    require(q["science_package"]["manifest"]["sha256"] == SCI_SHA, "original SCI1940 required")
    require(set(q["helper_roots"]) == HELPERS, "four original helper roots required")
    for path in q["helper_roots"].values(): absolute(path)
    require(set(q["snapshot"]) == {"root", "inventory"}, "original snapshot required")
    absolute(q["snapshot"]["root"]); validate_descriptor(q["snapshot"]["inventory"])
    require(q["snapshot"]["inventory"]["sha256"] == INVENTORY_SHA, "original SD35 snapshot required")
    require(q["source_mapping"] is None or set(q["source_mapping"]) == {"prefix", "root"}, "explicit source mapping required")
    if q["source_mapping"] is not None:
        for path in q["source_mapping"].values(): absolute(path)
    if q["weights_root"] is not None: absolute(q["weights_root"])
    base, output = absolute(q["output_base"]), absolute(q["output_directory"])
    require(OUTPUT_NAMESPACE in base.parts and output.parent == base and output.name == "formal200_part_training", "new exclusive part output required")
    require(set(q["source_files"]) == {"probe", "formal_contract", "connectivity_library", "kernel", "derived_engine"}, "five bound source files required")
    for source in q["source_files"].values(): validate_descriptor(source)
    for name, digest in (("connectivity_library", LIBRARY_SHA), ("kernel", KERNEL_SHA), ("derived_engine", ENGINE_SHA)):
        require(q["source_files"][name]["sha256"] == digest, "unchanged scientific source required: " + name)
    require(isinstance(q["storage_profile"], dict), "finite storage profile required")
    return q



def time_and_identity(q):
    require(socket.gethostname() == q["expected_hostname"], "fresh hostname differs")
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == q["expected_boot_id"], "fresh boot differs")
    require(0 < q["deadline_unix"] - time.time() <= 86400, "expired or unbounded operator deadline")


def expected_call_budget(updates):
    require(type(updates) is int and updates in (25, 75, 125), "new explicit25/75/125 budget only")
    outer = updates // 25
    return {"detector_forwards": 4 * updates + 2, "autograd_calls": 2 * updates,
        "transformer_forward_calls": updates, "vae_encode_calls": 1 + outer,
        "vae_decode_calls": 1 + 2 * updates + outer, "noise_draws": outer,
        "depth_clean_forwards": 4 * updates, "depth_adv_forwards": 4 * updates}


def stage_configuration(protocol, reference_plan, stage, scientific_config_for):
    require(stage == "formal200_part_training", "declared training stage required")
    from ablation_contract_v1 import validate_scientific_factory
    actual = validate_scientific_factory(protocol["method_id"], protocol["original_complete_scientific_config"], scientific_config_for)
    require(actual == protocol["scientific_config"], "declared ablation configuration changed")
    return 125, actual



def require_root_stage_binding(q, operator_ready):
    import ablation_contract_v1 as contract
    contract.validate_ready(q, operator_ready)



def load_science(q, plan, library):
    root = absolute(q["science_package"]["root"])
    manifest = json.loads(checked_file(q["science_package"]["manifest"]))
    require(absolute(q["science_package"]["manifest"]["path"]) == root / "PACKAGE_MANIFEST.json", "science root/manifest mismatch")
    for item in manifest["files"]:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe science member")
        checked_file({"path": str(root / relative), "bytes": item["bytes"], "sha256": item["sha256"]})
    sys.path.insert(0, str(root))
    for name in ("benchmark_mode", "source_bindings", "sample_contract", "geometry_eot"):
        require(name not in sys.modules, "scientific module collision: " + name)
    import benchmark_mode
    benchmark_mode.configure("C1")
    import source_bindings
    paths = source_bindings.verify_sources(plan, **{name: Path(path) for name, path in q["helper_roots"].items()})
    import sample_contract
    from b3_overrides import validate_plan as _b3_validate_plan
    _b3_validate_plan(plan, sample_contract)
    old_module = load_bound_source("_conditional_stage_original_sample_training", descriptor(root / "sample_training.py"))
    inputs, parent, clip, old_kernels = old_module.load_verified_helpers(paths)
    paths["legacy_ledger_source_observation"] = library.observe_legacy_ledger_source(old_kernels)
    paths["original_encode_decode_source_observation"] = {
        name: {"qualified_name": getattr(parent, name).__module__ + "." + getattr(parent, name).__qualname__,
            "source_file": descriptor(inspect.getsourcefile(getattr(parent, name)))}
        for name in ("encode_patch", "decode_patch")}
    return inputs, parent, clip, old_kernels, paths, sample_contract


def server_only_index(output):
    rows = [descriptor(path) for path in sorted(output.rglob("*")) if path.is_file()]
    require(not any(path.is_symlink() for path in output.rglob("*")), "new server-only case contains symlink")
    index = write_new(output / "SERVER_ONLY_ARTIFACT_INDEX.json", {"scope": "server-only full case artifact inventory; do not export body or tensor files to LOCAL/USB",
        "files": rows, "complete_scientific_or_final_acceptance": False})
    return {"server_only_inventory": index, "artifact_count_before_index": len(rows),
        "tensor_file_count": sum(Path(row["path"]).suffix == ".pt" for row in rows)}


def run(q, request_d):
    time_and_identity(q)
    for name, item in q["source_files"].items():
        path = Path(__file__) if name == "probe" else Path(__file__).with_name("ablation_contract_v1.py" if name == "formal_contract" else SOURCE_FILENAMES[name])
        require(absolute(item["path"]).resolve() == path.resolve(), "new source path mismatch: " + name)
        checked_file(item)
    library = load_bound_source("_conditional_training_bound_connectivity_library", q["source_files"]["connectivity_library"])
    kernel_source = load_bound_source("_conditional_training_bound_two_clip_kernel", q["source_files"]["kernel"])
    fc = load_bound_source("_actual_bound_ablation_contract", q["source_files"]["formal_contract"])
    require(fc.IDENTITY == IDENTITY and fc.PROTOCOL_SHA == PROTOCOL_SHA, "new formal contract identity/source protocol")
    protocol, master_reference, reference_plan = fc.load_inputs(q)
    prior_rows, prefix_proof = fc.accepted_prefix_rows(q)
    reused_count = len(prior_rows)
    operator_ready = json.loads(checked_file(q["operator_admission"]))
    library.validate_operator_GPU_query(operator_ready)
    require_root_stage_binding(q, operator_ready)
    storage_before = fc.storage_guard(q, operator_ready)
    snapshot = library.check_snapshot(q["snapshot"])
    inputs, parent, clip, old_kernels, paths, contract = load_science(q, reference_plan, library)
    fc.b3_overrides.apply()
    engine = load_bound_source("_conditional_training_bound_derived_engine", q["source_files"]["derived_engine"])
    updates, scientific_config = stage_configuration(protocol, reference_plan, q["stage"], contract.scientific_config_for)
    require(engine.planned_budget(updates) == expected_call_budget(updates), "new engine call budget differs")
    import torch
    import torchvision
    import diffusers
    import transformers
    require(torch.cuda.is_available(), "actual stage requires CUDA")
    require((torch.__version__.split("+")[0], torchvision.__version__.split("+")[0], diffusers.__version__.split("+")[0]) ==
        ("2.8.0", "0.23.0", "0.38.0"), "frozen runtime versions required")
    clip.configure_determinism()
    dual_runtime = engine.configure_dual_runtime(torch)
    pr = paths["child"] / "parent_v17"
    mapping = q["source_mapping"]
    records = parent.training_records(pr / "01_公共数据集/runtime_v1/train_validation_manifest.csv",
        Path(mapping["prefix"]) if mapping else None, Path(mapping["root"]) if mapping else None)
    selected = [contract.record_for_sample(reference_plan, sample, records) for sample in reference_plan["samples"]]
    for sample, record in zip(reference_plan["samples"], selected):
        require(sha(record.source_path) == sample["source_sha256"], "actual input wholeSHA differs")
    car_file, mask_file = (pr / "01_公共数据集" / name for name in ("car_00.jpg", "car_00_mask.jpg"))
    model_manifest = pr / "02_模型/model_manifest.json"
    assets, depth_assets, ready = clip.preflight_training_assets(parent, selected, model_manifest,
        car_file, mask_file, Path(q["weights_root"]) if q["weights_root"] else None, 1.0)
    require(ready, "original input/car/four-source assets unavailable")
    output, base = absolute(q["output_directory"]), absolute(q["output_base"])
    require(base.is_dir() and not output.exists() and not any(path.is_symlink() for path in (output, *output.parents)),
            "new exclusive stage output under existing base required")
    old_roots = [Path(path).resolve() for path in q["helper_roots"].values()] + [Path(q["snapshot"]["root"]).resolve(), Path(q["science_package"]["root"]).resolve()]
    require(not any(output.resolve().is_relative_to(root) for root in old_roots), "output cannot be inside original source/input tree")
    time_and_identity(q)
    output.mkdir()
    execution_plan = fc.make_execution_plan(q, protocol, master_reference, reference_plan)
    execution_plan_d = write_new(output / "ABLATION_PART_EXECUTION_PLAN.json", execution_plan)
    stage_started = time.monotonic()
    write_new(output / "STAGE_INTENT.json", {"identity": IDENTITY, "request": request_d, "stage": q["stage"],
        "scope": "one original50 part, fresh125, declared existing-switch ablation",
        "task_scope": q["task_scope"], "method_id": q["method_id"], "operator_admission": q["operator_admission"],
        "hostname": socket.gethostname(), "boot_id": q["expected_boot_id"], "pid": os.getpid(), "deadline_unix": q["deadline_unix"]})
    callback = None
    gpu_identity = None
    try:
        gpu_identity = library.cuda_identity_before_model(torch, operator_ready)
        from diffusers import StableDiffusion3Pipeline
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        torch.cuda.reset_peak_memory_stats(device)
        pipeline = StableDiffusion3Pipeline.from_pretrained(Path(q["snapshot"]["root"]), torch_dtype=dtype,
            local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True,
            text_encoder_3=None, tokenizer_3=None).to(device)
        # The original parent decoder, including decoded.float() postprocessing,
        # remains the actual attack decoder; no .decode_leaf or VAEOnly builder.
        car, mask = parent.load_car(car_file, mask_file, device)
        adapters, original_depth = clip.build_training_adapters(parent, device, model_manifest,
            Path(q["weights_root"]) if q["weights_root"] else None, 1.0)
        from depth_hinge_objective import DepthObjective
        depth_objective = DepthObjective(parent, original_depth.adapters,
            beta=scientific_config["depth_beta"], depth_target=scientific_config["switches"]["depth_target"],
            epsilon=scientific_config["controller_config"]["epsilon"])
        config = parent.StatefulOursConfig(**{key: scientific_config[key] for key in engine.CONFIG}, outer_rounds=updates // 25)
        rows = deepcopy(prior_rows)
        if reused_count:
            publish_completed_prefix(output, q, request_d, rows)
        for sample, record in zip(reference_plan["samples"][reused_count:], selected[reused_count:]):
            time_and_identity(q)
            case_storage_before=fc.storage_guard(q, operator_ready)
            case_started_wall,case_started_cpu=time.monotonic(),time.process_time()
            engine.reset_sample_rng(config.seed)
            case = output / "samples" / sample["case_id"]
            case.mkdir(parents=True, exist_ok=False)
            train = case / "train"
            context = parent.build_context(record, car, mask, device, set())
            initial_small = parent.extract_road_patch(context.background, context.road_quad)
            initial_patch = torch.nn.functional.interpolate(initial_small, size=(512, 512), mode="bilinear", align_corners=False)
            callback = engine.ConditionalTrainingDenoise(pipeline, kernel_source, deadline_unix=q["deadline_unix"])
            prompt_pack, condition_metadata = callback.encode_condition(device)
            kernels = engine.with_real_conditioning(old_kernels, callback)
            train.mkdir(exist_ok=False)
            torch.cuda.reset_peak_memory_stats(device)
            write_new(case / "CASE_CALL_INTENT.json", {"request": request_d, "stage": q["stage"], "sample": sample,
                "actual_original_source_sha256": sha(record.source_path), "fresh_empty_train_directory": str(train),
                "updates_requested": updates, "real_condition": condition_metadata})
            started_wall, started_cpu = time.monotonic(), time.process_time()
            summary = engine.run_sample_optimization(parent=parent, config=config, output=train, pipeline=pipeline,
                prompt_pack=prompt_pack, initial_patch=initial_patch, initial_small=initial_small, sample=sample,
                context=context, adapters=adapters, depth_objective=depth_objective, device=device, dtype=dtype,
                eot_config=scientific_config["eot"], kernels=kernels, plan_sha256=execution_plan_d["sha256"],
                source_bindings=reference_plan["source_bindings"], scientific_config=scientific_config)
            torch.cuda.synchronize(device)
            returned_wall, returned_cpu = time.monotonic(), time.process_time()
            observation = callback.observation()
            require(summary["status"] == "train_optimization_complete" and summary["executed_updates"] == updates,
                    "new core did not return all declared updates")
            engine.validate_ledger_snapshot(summary["budget"], expected_call_budget(updates))
            require(observation["callback_completion_checks_passed"] is True and observation["completed_generation_steps"] == updates,
                    "actual generation callback did not complete")
            generation_d = write_new(case / "CONDITIONAL_GENERATION_OBSERVATION.json", observation)
            measurement_d = write_new(case / "CASE_ACTUAL_MEASUREMENT.json", {"scope": "actual core function call plus its immediate CUDA completion synchronization; intent/observation/model load/preflight excluded",
                "core_call_and_sync_elapsed_seconds": returned_wall - started_wall,
                "process_CPU_seconds_delta_for_core_call_and_sync": returned_cpu - started_cpu,
                "original_inner_core_elapsed_seconds": summary["elapsed_seconds"],
                "process_cumulative_ru_maxrss_Linux_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_cuda_allocated_bytes_after_reset_including_resident_models": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes_after_reset_including_resident_models": torch.cuda.max_memory_reserved(device)})
            returned_d = write_new(case / "CASE_RETURNED.json", {"schema": "ours5-ablation-case-returned-v1",
                "identity": IDENTITY, "method_id": q["method_id"], "ablation_variant": scientific_config["ablation_variant"],
                "task_scope": q["task_scope"], "part_id": q["part_id"], "executed_updates": updates,
                "sample": sample, "patch_bindings": summary["patch_bindings"],
                "native_metadata_expected": native_metadata_expected(summary),
                "stage": q["stage"], "sample_id": sample["sample_id"],
                "actual_function_return_observed": "bound derived engine run_sample_optimization returned normally",
                "summary": descriptor(train / "summary.json"), "actual_budget_snapshot": summary["budget"],
                "generation_observation": generation_d, "actual_measurement": measurement_d,
                "not_a_process_END_or_independent_scientific_acceptance": True})
            callback.close(); callback = None
            custody = server_only_index(train)
            case_storage_after=fc.storage_guard(q,operator_ready)
            case_returned_wall,case_returned_cpu=time.monotonic(),time.process_time()
            total_measurement_d=write_new(case / "CASE_TOTAL_ACTUAL_MEASUREMENT.json", {
                "scope":"fresh case RNG/context/initial patch/condition plus core and immediate CUDA synchronization plus original tiny/RETURN/native-custody retention and post-case disk guard; phase model load/preflight and this measurement write excluded",
                "budget_outer_rounds":q["budget_outer_rounds"],"updates":updates,
                "case_start_monotonic":case_started_wall,"case_return_monotonic":case_returned_wall,
                "case_end_to_end_elapsed_seconds":case_returned_wall-case_started_wall,
                "preprocessing_condition_and_intent_elapsed_seconds":started_wall-case_started_wall,
                "core_call_and_sync_elapsed_seconds":returned_wall-started_wall,
                "post_core_observation_RETURN_and_custody_elapsed_seconds":case_returned_wall-returned_wall,
                "case_process_CPU_seconds_delta":case_returned_cpu-case_started_cpu,
                "actual_core_measurement":measurement_d,"actual_CASE_RETURNED":returned_d,
                "actual_DATA_storage_before_case":case_storage_before,"actual_DATA_storage_after_case":case_storage_after,
                "model_load_apportioned_to_case":False,"checkpoint_from_another_budget_loaded":False})
            rows.append({"sample": sample, "actual_source_path": str(record.source_path), "source_sha256": sha(record.source_path),
                "returned": returned_d, "summary": descriptor(train / "summary.json"), "generation_observation": generation_d,
                "actual_measurement": measurement_d, "actual_total_measurement":total_measurement_d, "attack_trace": descriptor(train / "attack_trace.jsonl"),
                "cycle_trace": descriptor(train / "cycle_trace.jsonl"), "state_manifest": descriptor(train / "state_evidence/MANIFEST.json"),
                "executed_updates": summary["executed_updates"], "actual_budget_snapshot": summary["budget"],
                "origin_kind": "NEW_CONDITIONAL_BUDGET", "actual_training_plan": execution_plan_d,
                "actual_new_attack_updates": updates, "new_function_calls": 1, **custody})
            publish_completed_prefix(output, q, request_d, rows)
            del context, initial_small, initial_patch, prompt_pack, kernels, summary
        time_and_identity(q)
        fc.storage_guard(q, operator_ready)
        result = {"schema": "ours5-ablation-original50-training-returned-v1", "method_id": q["method_id"], "identity": IDENTITY, "stage": q["stage"],
            "status": "FORMAL50_BUDGET_TRAINING_RETURNED_NOT_INDEPENDENT_ACCEPTANCE", "request": request_d,
            "protocol": q["protocol"], "original_input_plan": q["reference_plan"], "master_reference": q["master_reference"],
            "actual_execution_plan": execution_plan_d, "part_id": q["part_id"], "budget_outer_rounds":q["budget_outer_rounds"], "source_files": q["source_files"],
            "operator_admission": q["operator_admission"], "task_scope": q["task_scope"],
            "conditioning_mode": CONDITIONING_MODE, "T5_loaded_or_executed": False,
            "scientific_config": scientific_config, "ablation_variant": scientific_config["ablation_variant"],
            "old500_INPUT_reference_unmodified_and_real_budget_execution_plan": True,
            "fresh_initialization_for_this_budget":True,"checkpoint_from_another_budget_loaded":False,
            "full_snapshot": snapshot, "source_package_verification": paths["verified"],
            "legacy_ledger_source_observation": paths["legacy_ledger_source_observation"],
            "original_encode_decode_source_observation": paths["original_encode_decode_source_observation"],
            "source_assets": assets, "depth_weight_assets": depth_assets, "dual_backward_runtime": dual_runtime,
            "runtime": {"hostname": socket.gethostname(), "boot_id": q["expected_boot_id"], "pid": os.getpid(),
                "torch": torch.__version__, "torchvision": torchvision.__version__, "diffusers": diffusers.__version__,
                "transformers": transformers.__version__, "actual_CUDA_identity_before_model_load": gpu_identity,
                "elapsed_seconds_including_model_load_and_part_cases": time.monotonic() - stage_started},
            "rows": rows, "effective_case_count": len(rows), "effective_updates": updates*len(rows),
            "new_case_count": len(rows) - reused_count,
            "reused_case_count": reused_count,
            "actual_new_attack_updates": sum(r["actual_new_attack_updates"] for r in rows[reused_count:]),
            "actual_task_scope":q["task_scope"],
            "process_END_receipt_owned_by_external_actual_waiter": True, "independent_engineering_or_scientific_acceptance": False,
            "eight_model_final_only_evaluation_executed": False, "protocol_diagnostic500_full_acceptance": False,
            "original200_four_part_scope_not_TEST_not_heldout": True, "validation40_are_also_optimized": True, "test_data_opened": False,
            "server_only_tensors_and_full_inventory_not_for_LOCAL_or_USB": True}
        if reused_count:
            result.update(schema=fc.RESUME_RESULT_SCHEMA,
                          status="FORMAL50_PREFIX_REUSED_SUFFIX_RETURNED_NOT_INDEPENDENT_ACCEPTANCE",
                          accepted_prefix_CPU=q["accepted_prefix_CPU"],
                          original_failed_request=prefix_proof["old_request"],
                          original_failed_END=prefix_proof["old_END"],
                          prior_complete_rows_reused_without_model_calls=reused_count,
                          unfinished_original_next_case_not_loaded=True)
        result_d = write_new(output / "ABLATION_PART_TRAINING_RETURNED.json", result)
        print(json.dumps({"result": result_d, "status": result["status"], "stage": q["stage"],
            "effective_updates": result["effective_updates"], "actual_new_attack_updates": result["actual_new_attack_updates"]}, ensure_ascii=False))
    except BaseException as error:
        if isinstance(error, library.GPUIdentityBindingError):
            gpu_identity = error.observation
        write_new(output / "STAGE_FAILURE.json", {"identity": IDENTITY, "stage": q["stage"], "request": request_d,
            "error": repr(error), "traceback": traceback.format_exc(), "no_retry": True,
            "actual_CUDA_identity_before_model_load": gpu_identity,
            "partial_conditional_observation": callback.observation() if callback is not None else None,
            "elapsed_seconds": time.monotonic() - stage_started})
        raise
    finally:
        if callback is not None:
            callback.close()


def native_metadata_expected(summary):
    keys = ("identity", "sample_id", "case_id", "dataset", "source_split", "source_sha256",
            "plan_sha256", "source_bindings", "input_role", "test_data_opened", "dual_backward_runtime")
    shared = {key: summary[key] for key in keys}
    return {"zero": {**shared, "role": "sample_specific_initial_road_crop", "updates": 0},
            "final": {**shared, "role": "sample_specific_final_patch", "updates": 125,
                      "checkpoint_selection": "fixed_final"}}


def publish_completed_prefix(output, q, request_d, rows):
    value = {"schema": "ours5-ablation-completed-case-prefix-v1", "method_id": q["method_id"],
             "request": request_d, "task_scope": q["task_scope"], "part_id": q["part_id"],
             "completed_cases": len(rows), "completed_updates": 125 * len(rows), "rows": rows}
    target = Path(output) / "CASE_COMPLETED_PREFIX.json"
    temporary = target.with_name("CASE_COMPLETED_PREFIX.next.json")
    write_new(temporary, value)
    os.replace(temporary, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true", help="requires actual root owned lifecycle and previous-stage admission")
    args = parser.parse_args()
    require(re.fullmatch("[0-9a-f]{64}", args.request_sha256), "explicit request wholeSHA required")
    request_d = descriptor(args.request)
    require(request_d["sha256"] == args.request_sha256, "actual request wholeSHA differs")
    q = validate_request(json.loads(checked_file(request_d)))
    if not args.execute:
        print(json.dumps({"status": "STAGE_REQUEST_METADATA_ONLY_NO_MODEL_NO_ADMISSION", "request": request_d,
            "stage": q["stage"], "scientific_acceptance": False}, ensure_ascii=False))
        return
    run(q, request_d)


if __name__ == "__main__":
    main()
