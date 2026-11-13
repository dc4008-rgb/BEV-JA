"""Two declared Ours5 ablations over the unchanged scientific implementation.

This module validates metadata. Importing it performs no model or SSH operation.
The process owner supplies current identity, resource admission and one deadline.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

import b3_overrides

IDENTITY = "ours5-b3-full-nuscenes100-existing-switch-ablation-v1"
PROTOCOL_SHA = "accee048bc48f068cd4f99684b0d3dfb99134d23afbf6a8f3d605d5d52ff0dc7"
MASTER_SHA = "eb4fa9bf9d394d90ed2b764de3d5ca54f26f2d0d3072050e8f9086ee522458f8"
MASTER_SAMPLES_SHA = "78fc617bcc0534c95288b6f1a91af5eb68707af49d2bcaa9b0a631788dc6a96a"
PART_SHA = (
    "b1cba8eb660d190641ef8d105f15573f48bc3e32231693f6ac6685c0297df5bc",
    "173ace34d822c9214b069155297b99ef5a75dabb8fc13d514a1ee60466838f78",
)
METHODS = {"ours5_b3_full_nuscenes100": {"variant": "custom", "node_id": "b3_m1", "switch_changes": {}}}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path).absolute()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def checked(binding):
    need(isinstance(binding, dict) and set(binding) == {"path", "bytes", "sha256"}, "closed descriptor required")
    path = Path(binding["path"])
    need(path.is_absolute() and ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "literal nonsymlink path required")
    need(type(binding["bytes"]) is int and 0 < binding["bytes"] <= 8 * 1024**2, "bounded metadata required")
    data = path.read_bytes()
    need(len(data) == binding["bytes"] and hashlib.sha256(data).hexdigest() == binding["sha256"], "bound metadata changed")
    return data


def read(binding):
    return json.loads(checked(binding), parse_constant=lambda value: need(False, "nonfinite JSON"))


def scientific_configuration(method_id, original_complete):
    need(method_id in METHODS, "only the two authorized ablations are admitted")
    need(original_complete["ablation_variant"] == "complete" and original_complete["outer_rounds"] == 5, "original full five-round configuration required")
    selected = deepcopy(original_complete)
    selected["ablation_variant"] = METHODS[method_id]["variant"]
    selected["switches"].update(METHODS[method_id]["switch_changes"])
    if b3_overrides.CONTROLLER_OVERRIDES:
        selected["controller_config"].update(b3_overrides.CONTROLLER_OVERRIDES)
    if b3_overrides.DEPTH_BETA is not None:
        selected["depth_beta"] = float(b3_overrides.DEPTH_BETA)
    return selected


def validate_scientific_factory(method_id, original_complete, factory):
    selected = scientific_configuration(method_id, original_complete)
    actual = factory("custom", 125, switches=selected["switches"])
    need(actual == selected, "original named variant differs from the declared switch-only change")
    return actual


def validate_task_scope(q):
    scope = read(q["task_scope"])
    need(scope.get("schema") == "ours5-b3-full-nuscenes100-authorized-scope-v1", "current ablation task scope required")
    need(scope.get("outer_rounds") == 5 and scope.get("inner_steps") == 25 and scope.get("sample_count_per_method") == 100, "five by twenty-five and nuScenes one hundred required")
    need(scope.get("fresh_initialization") is True, "fresh per-image initialization required")
    assignments = scope["assignments"]
    need(set(assignments) == set(METHODS), "both explicit method assignments required")
    for method_id, selection in METHODS.items():
        row = assignments[method_id]
        need(row["node_id"] == selection["node_id"] and row["part_ids"] == [0, 1], "each method uses the two nuScenes-only parts")
        need(re.fullmatch(r"autodl-container-[a-z0-9-]+", row["hostname"]) and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", row["boot_id"]), "observed complete host and boot required")
    selected = assignments[q["method_id"]]
    need(selected["node_id"] == q["node_id"] and selected["hostname"] == q["expected_hostname"] and selected["boot_id"] == q["expected_boot_id"], "actual request differs from assigned node")
    for key in ("started_unix", "absolute_deadline_unix"):
        need(type(scope[key]) in (int, float) and math.isfinite(scope[key]), "finite task deadline required")
    need(0 < scope["absolute_deadline_unix"] - scope["started_unix"] <= 129600 and scope["started_unix"] <= time.time() < q["deadline_unix"] <= scope["absolute_deadline_unix"], "current finite whole-task clock required")
    need(len(assignments) == 1 and set(assignments) == {"ours5_b3_full_nuscenes100"}, "one newly authorized b3-full-nuscenes100 task only")
    return scope


def load_inputs(q):
    need(q["protocol"]["sha256"] == PROTOCOL_SHA and q["master_reference"]["sha256"] == MASTER_SHA and q["reference_plan"]["sha256"] == PART_SHA[q["part_id"]], "literal original protocol, master and part required")
    protocol, master, part = (read(q[key]) for key in ("protocol", "master_reference", "reference_plan"))
    original = deepcopy(protocol["budget_configurations"]["5"]["scientific_config"])
    original_from_part = deepcopy(part["scientific_config"])
    original_from_part["outer_rounds"] = 5
    need(original == original_from_part, "same original complete scientific configuration required")
    need(master["master_samples_sha256"] == MASTER_SAMPLES_SHA and part["master_samples"] == master["master_samples"] and part["source_bindings"] == master["source_bindings"], "original cohort and source bindings required")
    ordered_ids = master["partition"]["shard_members"][str(q["part_id"]) ]
    by_id = {row["sample_id"]: row for row in master["master_samples"]}
    need(len(by_id) == 200 and len(ordered_ids) == 50 and len(set(ordered_ids)) == 50 and part["samples"] == [by_id[sample_id] for sample_id in ordered_ids], "exact original ordered fifty required")
    need(sum(row["dataset"] == "kitti" for row in part["samples"]) == 0 and sum(row["dataset"] == "nuscenes" for row in part["samples"]) == 50, "nuScenes-only fifty required")
    selected_protocol = deepcopy(protocol)
    selected_protocol["original_complete_scientific_config"] = original
    selected_protocol["scientific_config"] = scientific_configuration(q["method_id"], original)
    selected_protocol["selected_outer_rounds"] = 5
    selected_protocol["method_id"] = q["method_id"]
    return selected_protocol, master, part


def expected_call_budget(updates=125):
    need(updates == 125, "exact 125 updates per image required")
    return {"detector_forwards": 502, "autograd_calls": 250,
            "transformer_forward_calls": 125, "vae_encode_calls": 6,
            "vae_decode_calls": 256, "noise_draws": 5,
            "depth_clean_forwards": 500, "depth_adv_forwards": 500}


def make_execution_plan(q, protocol, master, part):
    return {"schema": "ours5-ablation-original50-execution-plan-v1", "identity": IDENTITY,
            "method_id": q["method_id"], "part_id": q["part_id"], "samples": deepcopy(part["samples"]),
            "source_bindings": deepcopy(part["source_bindings"]), "scientific_config": deepcopy(protocol["scientific_config"]),
            "budget": {"sample_count": 50, "updates_per_sample": 125, "total_updates": 6250},
            "original_input_plan": q["reference_plan"], "task_scope": q["task_scope"],
            "per_case_budget": expected_call_budget(), "fresh_initialization": True}


def validate_ready(q, ready):
    validate_task_scope(q)
    need(ready.get("ablation_method_id") == q["method_id"] and ready.get("task_scope") == q["task_scope"], "GPU admission bound to actual ablation scope required")
    need(ready.get("actual_identity") == {"hostname": q["expected_hostname"], "boot_id": q["expected_boot_id"]} and ready.get("deadline_unix") == q["deadline_unix"], "current host, boot and deadline admission required")
    need(ready.get("part_id") == q["part_id"], "admission for current part required")
    need(read(ready["base_request_without_admission"]) == {key: value for key, value in q.items() if key != "operator_admission"}, "exact preadmission request binding required")


def storage_guard(q, ready):
    profile = q["storage_profile"]
    need(set(profile) == {"namespace", "maximum_allocated_bytes", "reserve_bytes", "maximum_entries"}, "finite task storage profile required")
    root = Path(profile["namespace"])
    need(root.is_dir() and not root.is_symlink() and Path(q["output_base"]).parent == root, "owned task DATA namespace required")
    need(all(type(profile[key]) is int for key in ("maximum_allocated_bytes", "reserve_bytes", "maximum_entries")) and profile["maximum_allocated_bytes"] > 0 and profile["reserve_bytes"] >= 1024**3 and profile["maximum_entries"] > 0, "positive finite caps and reserve required")
    allocated = logical = entries = 0
    for path in root.rglob("*"):
        entries += 1
        need(entries <= profile["maximum_entries"], "owned DATA entry cap exceeded")
        current = path.lstat()
        need(not stat.S_ISLNK(current.st_mode) and current.st_dev == root.stat().st_dev, "ordinary same-device DATA entries required")
        if stat.S_ISDIR(current.st_mode):
            continue
        need(stat.S_ISREG(current.st_mode) and current.st_nlink == 1, "ordinary private DATA file required")
        allocated += current.st_blocks * 512
        logical += current.st_size
    fs = os.statvfs(root)
    free = fs.f_bavail * fs.f_frsize
    need(allocated <= profile["maximum_allocated_bytes"] and free >= profile["maximum_allocated_bytes"] - allocated + profile["reserve_bytes"], "DATA cap or free reserve exceeded")
    return {"namespace": str(root), "allocated_bytes": allocated, "logical_bytes": logical, "entries": entries, "free_bytes": free, "profile": profile}


def validate_completed_prefix(q, prefix):
    """Read completed case originals without loading tensors or accepting a partial case."""
    need(prefix["schema"] == "ours5-ablation-completed-case-prefix-v1" and prefix["method_id"] == q["method_id"]
         and prefix["task_scope"] == q["task_scope"] and prefix["part_id"] == q["part_id"], "same actual prefix provenance required")
    need(read(prefix["request"]) == q, "actual prefix request differs")
    protocol, master, part = load_inputs(q)
    rows = prefix["rows"]
    need(0 <= len(rows) <= 50 and prefix["completed_cases"] == len(rows) and prefix["completed_updates"] == 125 * len(rows), "complete prefix counters differ")
    need([row["sample"] for row in rows] == part["samples"][:len(rows)], "ordered original complete prefix required")
    for row in rows:
        returned = read(row["returned"])
        summary = read(row["summary"])
        generation = read(row["generation_observation"])
        plan = read(row["actual_training_plan"])
        need(returned["method_id"] == q["method_id"] and returned["sample"] == row["sample"]
             and returned["task_scope"] == q["task_scope"] and returned["part_id"] == q["part_id"]
             and returned["ablation_variant"] == METHODS[q["method_id"]]["variant"]
             and returned["executed_updates"] == 125 and returned["summary"] == row["summary"]
             and returned["generation_observation"] == row["generation_observation"], "actual case RETURN differs")
        need(plan["method_id"] == q["method_id"] and plan["task_scope"] == q["task_scope"]
             and plan["scientific_config"] == protocol["scientific_config"]
             and summary["plan_sha256"] == row["actual_training_plan"]["sha256"]
             and summary["source_sha256"] == row["sample"]["source_sha256"], "actual scientific input provenance differs")
        need(summary["status"] == "train_optimization_complete" and summary["executed_updates"] == 125
             and summary["scientific_config"] == protocol["scientific_config"] and summary["fresh_sample_initialization"] is True, "full original per-case scientific completion required")
        for key in ("planned", "attempted", "completed"):
            need(summary["budget"][key] == expected_call_budget(), "actual full call counts required")
        need(generation["callback_completion_checks_passed"] is True and generation["completed_generation_steps"] == 125, "true conditional generation completion required")
        for key in ("zero_step_patch.pt", "final_patch.pt"):
            binding = summary["patch_bindings"][key]
            need(isinstance(binding, dict) and set(binding) == {"bytes", "sha256"},
                 "original closed patch binding required")
            path = Path(row["summary"]["path"]).parent / key
            checked(dict(binding, path=str(path)))
        for key in ("actual_measurement", "actual_total_measurement"):
            read(row[key])
    return rows


def validate_result(q, result):
    need(result["schema"] == "ours5-ablation-original50-training-returned-v1" and result["method_id"] == q["method_id"]
         and result["part_id"] == q["part_id"] and result["task_scope"] == q["task_scope"], "current complete part RETURN required")
    need(result["effective_case_count"] == result["new_case_count"] == 50 and result["reused_case_count"] == 0
         and result["actual_new_attack_updates"] == 6250 and result["effective_updates"] == 6250, "all original fifty fresh125 cases required")
    prefix = {"schema": "ours5-ablation-completed-case-prefix-v1", "method_id": q["method_id"],
              "request": result["request"], "task_scope": q["task_scope"], "part_id": q["part_id"], "completed_cases": 50,
              "completed_updates": 6250, "rows": result["rows"]}
    validate_completed_prefix(q, prefix)
    return {"method_id": q["method_id"], "part_id": q["part_id"], "completed_cases": 50, "completed_updates": 6250}


def build_training_collection(request_bindings, result_bindings):
    """Join four real part returns in original master order for the evaluation caller."""
    need(len(request_bindings) == len(result_bindings) == 2, "two actual part requests and returns required")
    ordered = {}
    requests = [read(binding) for binding in request_bindings]
    method_id = requests[0]["method_id"]
    for q, request_binding, result_binding in zip(requests, request_bindings, result_bindings):
        need(q["method_id"] == method_id and q["task_scope"] == requests[0]["task_scope"] and q["part_id"] not in ordered, "one method and two unique parts required")
        result = read(result_binding)
        need(result["request"] == request_binding, "part RETURN request provenance differs")
        validate_result(q, result)
        ordered[q["part_id"]] = result
    need(set(ordered) == {0, 1}, "exact two nuScenes-only parts required")
    master = read(requests[0]["master_reference"])
    rows = [row for part_id in range(2) for row in ordered[part_id]["rows"]]
    by_id = {row["sample"]["sample_id"]: row for row in rows}
    nu_ids = {sample["sample_id"] for sample in master["master_samples"] if sample["dataset"] == "nuscenes"}
    need(len(rows) == len(by_id) == 100 and set(by_id) == nu_ids, "exact unique nuScenes100 cohort required")
    rows = [by_id[sample["sample_id"]] for sample in master["master_samples"] if sample["dataset"] == "nuscenes"]
    return {"schema": "ours5-b3-nuscenes100-training-collection-v1", "identity": IDENTITY,
            "method_id": method_id, "ablation_variant": METHODS[method_id]["variant"],
            "task_scope": requests[0]["task_scope"], "master_reference": requests[0]["master_reference"],
            "request_bindings": request_bindings, "result_bindings": result_bindings,
            "sample_count": 100, "updates_per_sample": 125, "effective_training_updates": 12500,
            "rows": rows, "tensor_loading_or_model_calls": 0}


def accepted_prefix_rows(q):
    need("accepted_prefix_CPU" not in q, "fresh b3-full-nuscenes100 task cannot reuse any old case")
    return [], None
