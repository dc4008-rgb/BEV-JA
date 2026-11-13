"""Independent user-specified periodic FOA contract; no historical results reused."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from source_bindings import SOURCE_MANIFESTS, DATA_MANIFEST_SHA256, PAPER_SHA256, REFERENCE_MANIFEST_SHA256, SPECIFICATION_SHA256

IDENTITY = "ours_complete_eot1_parallel200_v1"
SCHEMA = "ours-complete-eot1-parallel200-plan-v1"
ROOT = Path(__file__).resolve().parent
OLD_MASTER_SAMPLES_SHA256 = "0538c8ab84521abb088e780be6706e7c340f3b47eddabaf482023d4d6b76c23b"
SHARD_COUNT = 4
MASTER_SAMPLE_COUNT = 200
DIAGNOSTIC_RECEIPT_SCHEMA = "ours-complete-eot1-parallel200-diagnostic-acceptance-v1"
ADMISSION_RECEIPT_SCHEMA = "ours-complete-eot1-parallel200-admission-v1"
DIAGNOSTIC_IDS = ('kitti_000046', 'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404')
CONTROLLER_CONFIG = {
    "theta": 0.3, "chi": 0.5, "epsilon": 1e-8, "beta_eval": 0.1,
    "dmax": 3.0, "model_mu": 0.5, "model_temperature": 1.0,
    "task_mu": 0.8, "task_temperature": 1.0, "ema_decay": 0.7,
    "ratio_delta": 1e-3, "ratio_min": 0.5, "ratio_max": 2.0,
    "n_min": 1e-8,
    "uncertified_reference_policy": "unit_deficit_preserve_scene",
    "vehicle_labels": [3, 6, 8],
}
VARIANTS = ("old_target_equal", "hinge_target_equal", "fixed_task_balance",
            "periodic_adaptive", "periodic_ema", "complete")
SWITCHES = {"depth_target": "at_least_beta", "model_adaptive": True,
    "task_adaptive": True, "task_balance": "unit_direction", "ema": True,
    "conflict_guard": True, "norm_match": True, "period": 5}
VARIANT_SWITCHES = {
    "old_target_equal": {**SWITCHES, "depth_target": "exact_beta", "model_adaptive": False,
        "task_adaptive": False, "task_balance": "raw_sum", "ema": False,
        "conflict_guard": False, "norm_match": False},
    "hinge_target_equal": {**SWITCHES, "model_adaptive": False,
        "task_adaptive": False, "task_balance": "raw_sum", "ema": False,
        "conflict_guard": False, "norm_match": False},
    "fixed_task_balance": {**SWITCHES, "model_adaptive": False,
        "task_adaptive": False, "ema": False, "conflict_guard": False},
    "periodic_adaptive": {**SWITCHES, "ema": False, "conflict_guard": False},
    "periodic_ema": {**SWITCHES, "conflict_guard": False},
    "complete": dict(SWITCHES),
}

def validate_switches(switches):
    if not isinstance(switches, dict) or set(switches) != set(SWITCHES):
        raise ValueError("all eight explicit independent switches required")
    for key in ("model_adaptive", "task_adaptive", "ema", "conflict_guard", "norm_match"):
        if type(switches[key]) is not bool:raise ValueError(key + " must be boolean")
    if switches["depth_target"] not in ("exact_beta", "at_least_beta"):
        raise ValueError("invalid depth target")
    if switches["task_balance"] not in ("raw_sum", "unit_direction"):
        raise ValueError("invalid task balance")
    if type(switches["period"]) is not int or switches["period"] not in (1, 5):
        raise ValueError("period must be1 or5")
    return switches

BASE_CONFIG = {
    "num_inference_steps": 100, "strength": 0.25, "latent_lr": 0.05,
    "step_feedback_mode": "latent_chain", "guidance_scale": 4.5,
    "seed": 20260919,
    "prompt": "top-down aerial view of clean asphalt road surface",
    "negative_prompt": "vehicle, car, obstacle",
    "depth_weight": 1.0, "depth_beta": 0.1,
    "gradient_coefficient": 100000.0, "clip_max_norm": 120.0,
    "detector_models": ["fasterrcnn", "ssd300"],
    "depth_models": ["monodepth2", "dpt_large"],
    "weighting_rule": "periodic_gap_gated_foa_v1",
    "gradient_combination": "two_task_fp32_periodic_balance_v1",
    "controller_config": copy.deepcopy(CONTROLLER_CONFIG),
    "eot": {"crop_side_fraction": [0.9, 1.0], "scale_range": [0.9, 1.1],
            "translation_fraction": [-0.05, 0.05], "rotation_degrees": [-5.0, 5.0],
            "padding_mode": "border", "random_views": 3, "max_attempts": 50},
}


from benchmark_mode import configure as configure_benchmark, total_views as benchmark_total_views, random_views as benchmark_random_views


def scientific_config_for(variant, updates=25, *, switches=None):
    if variant not in VARIANTS and variant != "custom":raise ValueError("unknown periodic variant")
    chosen = copy.deepcopy(VARIANT_SWITCHES[variant]) if switches is None else copy.deepcopy(validate_switches(switches))
    if variant != "custom" and chosen != VARIANT_SWITCHES[variant]:raise ValueError("named variant switches differ")
    base = copy.deepcopy(BASE_CONFIG)
    base["eot"]["random_views"] = benchmark_random_views()
    return {**base, "outer_rounds": updates // 25,
            "switches": chosen, "ablation_variant": variant}


def planned_budget(updates, variant=None):
    """Actual calls, including two clean identity detectors for each fresh case."""
    if type(updates) is not int or updates not in (25, 500):
        raise ValueError("only fixed25 diagnostic or full500 formal budgets are allowed")
    outer = updates // 25
    return {"detector_forwards": 2 * benchmark_total_views() * updates + 2, "autograd_calls": 2 * updates,
        "transformer_forward_calls": updates, "vae_encode_calls": 1 + outer,
        "vae_decode_calls": 1 + 2 * updates + outer, "noise_draws": outer,
        "depth_clean_forwards": 2 * benchmark_total_views() * updates, "depth_adv_forwards": 2 * benchmark_total_views() * updates}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document_sha256(document):
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def partition_samples(master_samples):
    """Four fixed shards, each old5+new20 per dataset, then interleave domains."""
    groups = {}
    for domain in ('kitti', 'nuscenes'):
        for role in ('retained_validation40', 'added_train160'):
            groups[domain, role] = sorted((s for s in master_samples if s['dataset'] == domain and
                s['selection_role'] == role), key=lambda s: s['sample_id'])
    shards = {}
    for shard_id in range(SHARD_COUNT):
        per_domain = [groups[domain, 'retained_validation40'][shard_id::SHARD_COUNT] +
            groups[domain, 'added_train160'][shard_id::SHARD_COUNT] for domain in ('kitti', 'nuscenes')]
        shards[str(shard_id)] = [s for pair in zip(*per_domain) for s in pair]
    return shards


def partition_contract(master):
    shards = partition_samples(master)
    return {'rule': 'dataset_and_original_role_sorted_alternating_four', 'shard_count': SHARD_COUNT,
        'case_order': 'interleave_kitti_nuscenes_within_shard_old5_then_new20_per_domain',
        'shard_members': {key: [s['sample_id'] for s in rows] for key, rows in shards.items()}}


def _validate_master(plan):
    # Original200 selection/QA is already sealed. Open its exact new-package
    # metadata copies rather than historical LOCAL source paths; actual source
    # CSV/image hashes remain mandatory in the unchanged runtime preflight.
    from formal_execution_contract import validate_master_plan_reference
    return validate_master_plan_reference(plan)


def _positive_number(value, name):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def validate_diagnostic_receipt(document, *, diagnostic_plan_sha256):
    if (document.get("schema") != DIAGNOSTIC_RECEIPT_SCHEMA or document.get("identity") != IDENTITY
        or document.get("status") != "diagnostic_accepted"):
        raise ValueError("completed diagnostic acceptance receipt required")
    if document.get("diagnostic_plan_sha256") != diagnostic_plan_sha256 or not re.fullmatch(r"[0-9a-f]{64}", diagnostic_plan_sha256):
        raise ValueError("diagnostic plan SHA differs")
    if document.get("source_bindings") != SOURCE_MANIFESTS or document.get("real_cuda_run") is not True or document.get("test_data_opened") is not False:
        raise ValueError("real CUDA and frozen sources/no-test diagnostic required")
    if (document.get("specification_sha256") != SPECIFICATION_SHA256
        or document.get("controller_config") != CONTROLLER_CONFIG):
        raise ValueError("diagnostic must bind this exact user specification and periodic controller configuration")
    diagnostic_config = scientific_config_for(document.get("ablation_variant"), 25,
        switches=document.get("switches"))
    if document.get("switches") != diagnostic_config["switches"]:
        raise ValueError("diagnostic must explicitly bind all eight switches")
    if document.get("source_package_sha256") is None or not re.fullmatch(r"[0-9a-f]{64}", document["source_package_sha256"]):
        raise ValueError("diagnostic must bind this independent scientific package")
    if any(not isinstance(document.get(key), str) or not document[key] for key in
           ("checked_hostname", "boot_id", "gpu_uuid", "driver")):
        raise ValueError("diagnostic must record actual live host/boot/GPU/driver")
    cases = document.get("cases", [])
    if tuple(c.get("sample_id") for c in cases) != DIAGNOSTIC_IDS or len(cases) != 2:
        raise ValueError("both fixed25-step diagnostic cases required")
    for case in cases:
        if case.get("executed_updates") != 25 or case.get("source_split") != "train":
            raise ValueError("complete25 train-only diagnostic required")
        if case.get("finite_nonzero_joint_gradient") is not True or case.get("state_reset_verified") is not True:
            raise ValueError("diagnostic gradient and independent reset must be verified")
        if case.get("budget") != planned_budget(25):
            raise ValueError("diagnostic must verify actual two-VJP and clean-reference call budget")
        if case.get("periodic_controller_verified") is not True:
            raise ValueError("actual periodic model/task weights and guarded gradient combination must be verified")
        for key in ("summary_sha256", "trace_sha256", "runtime_provenance_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", case.get(key, "")):
                raise ValueError("bound diagnostic artifact SHA required")
    _positive_number(document.get("elapsed_seconds"), "diagnostic elapsed_seconds")
    return document


def validate_admission_receipt(document, *, shard_id, diagnostic_receipt_sha256):
    if (document.get("schema") != ADMISSION_RECEIPT_SCHEMA or document.get("identity") != IDENTITY
        or document.get("status") != "admitted" or document.get("shard_id") != shard_id):
        raise ValueError("this shard requires a new resource admission receipt")
    if document.get("diagnostic_receipt_sha256") != diagnostic_receipt_sha256:
        raise ValueError("admission diagnostic binding differs")
    if document.get("budget") != {"sample_count": 50, "updates_per_sample": 500, "total_updates": 25000}:
        raise ValueError("resource admission must cover cumulative50x500; fresh40 and retained10 remain separate")
    for key in ("package_sha256",):
        if not re.fullmatch(r"[0-9a-f]{64}", document.get(key, "")):
            raise ValueError("admission package SHA required")
    if not document.get("checked_hostname") or not document.get("boot_id") or not document.get("gpu_uuid") or not document.get("driver") or document.get("all_assets_verified") is not True:
        raise ValueError("live host/boot/GPU/driver and full asset verification required")
    if (document.get("specification_sha256") != SPECIFICATION_SHA256
        or document.get("controller_config") != CONTROLLER_CONFIG):
        raise ValueError("admission must bind this user specification and controller configuration")
    free = _positive_number(document.get("free_disk_bytes"), "free_disk_bytes")
    required = _positive_number(document.get("required_disk_bytes"), "required_disk_bytes")
    estimated = _positive_number(document.get("estimated_seconds"), "estimated_seconds")
    if free < required:
        raise ValueError("insufficient disk for formal shard")
    if "max_runtime_seconds" not in document:
        raise ValueError("time admission must explicitly declare its limit or null")
    maximum = document["max_runtime_seconds"]
    if maximum is not None and estimated > _positive_number(maximum, "max_runtime_seconds"):
        raise ValueError("estimated formal shard exceeds admitted time")
    return document


def _validate_formal_bindings(plan):
    diagnostic = plan.get("diagnostic_binding", {})
    admission = plan.get("admission_binding", {})
    for binding in (diagnostic, admission):
        if not isinstance(binding.get("document"), dict) or not re.fullmatch(r"[0-9a-f]{64}", binding.get("sha256", "")):
            raise ValueError("formal plans require both explicit bound receipts")
        # The original file SHA and canonical parsed-document SHA are separate;
        # the latter ensures an embedded receipt cannot silently be edited.
        if document_sha256(binding["document"]) != binding.get("document_sha256"):
            raise ValueError("embedded acceptance/admission receipt changed")
    validate_diagnostic_receipt(diagnostic["document"], diagnostic_plan_sha256=plan.get("diagnostic_plan_sha256", ""))
    validate_admission_receipt(admission["document"], shard_id=plan["shard_id"],
        diagnostic_receipt_sha256=diagnostic["sha256"])
    if diagnostic["document"]["source_package_sha256"] != admission["document"]["package_sha256"]:
        raise ValueError("diagnostic and formal admission scientific packages must be identical")
    if (diagnostic["document"]["ablation_variant"] != plan["ablation_variant"]
        or diagnostic["document"]["switches"] != plan["scientific_config"]["switches"]):
        raise ValueError("formal switches must match their completed diagnostic")
    if any(diagnostic["document"][key] != admission["document"][key]
           for key in ("checked_hostname", "boot_id", "gpu_uuid", "driver")):
        raise ValueError("this formal admission requires its own current runtime GPU diagnostic")


def validate_plan(plan, *, allow_draft=False):
    if plan.get("schema") != SCHEMA or plan.get("identity") != IDENTITY:
        raise ValueError("sample-specific plan identity differs")
    executable = plan.get("status") == "frozen_diagnostic_plan"
    formal = plan.get("status") == "frozen_formal_shard"
    if not executable and not formal and not (allow_draft and plan.get("status") == "draft_not_executable"):
        raise ValueError("this plan is not executable; GPU diagnostic and formal budget are pending")
    if plan.get("specification_sha256") != SPECIFICATION_SHA256:
        raise ValueError("active user specification differs")
    if plan.get("source_bindings") != SOURCE_MANIFESTS:
        raise ValueError("source bindings differ")
    if plan.get("paper_sha256") != PAPER_SHA256 or plan.get("accepted_reference_package_sha256") != REFERENCE_MANIFEST_SHA256:
        raise ValueError("legacy paper provenance or read-only reference identity differs")
    if plan.get("data_manifest_sha256") != DATA_MANIFEST_SHA256:
        raise ValueError("data manifest identity differs")
    if plan.get("protocol_sha256") != sha256(ROOT / "PROTOCOL.md"):
        raise ValueError("sample-specific protocol changed")
    updates = plan.get("budget", {}).get("updates_per_sample")
    if type(updates) is not int or updates not in (25, 500):
        raise ValueError("fixed budget must be two-train25 diagnostic or full500 formal")
    if plan.get("call_budget_per_sample") != planned_budget(updates):
        raise ValueError("periodic two-VJP exact call budget differs")
    if plan.get("clean_reference_detector_forwards_per_sample") != 2:
        raise ValueError("two clean identity detector references must be disclosed")
    variant = plan.get("ablation_variant")
    switches = plan.get("scientific_config", {}).get("switches")
    expected = scientific_config_for(variant, updates, switches=switches)
    if plan.get("scientific_config") != expected:
        raise ValueError("scientific configuration differs")
    if plan.get("initialization") != "own_unattacked_scene_road_crop":
        raise ValueError("each sample must initialize from its own unpatched road crop")
    if plan.get("checkpoint_selection") != "fixed_final":
        raise ValueError("only the fixed final checkpoint is eligible")
    if plan.get("reset_policy") != "fresh_latent_scheduler_periodic_controller_rng_per_sample":
        raise ValueError("cross-sample state reuse is forbidden")
    if plan.get("uncertified_reference_handling") != "unit_deficit_preserve_scene_not_success":
        raise ValueError("all input scenes must remain, with neutral uncertified-reference handling")
    samples = plan.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("nonempty attack inputs are required")
    ids, cases = set(), set()
    for sample in samples:
        sample_id, case_id = sample.get("sample_id"), sample.get("case_id")
        if not isinstance(sample_id, str) or sample_id in ids:
            raise ValueError("duplicate or missing sample ID")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", case_id) or case_id in cases:
            raise ValueError("duplicate or unsafe case directory")
        ids.add(sample_id); cases.add(case_id)
        if sample.get("dataset") not in ("kitti", "nuscenes"):
            raise ValueError("unsupported attack dataset")
        if sample.get("source_split") not in ("train", "validation"):
            raise ValueError("test or unknown inputs cannot be optimized")
        if not re.fullmatch(r"[0-9a-f]{64}", sample.get("source_sha256", "")):
            raise ValueError("source image SHA is required")
        if not Path(sample.get("source_path", "")).is_absolute():
            raise ValueError("original source path must be recorded")
        quad = json.loads(sample["road_quad_norm"])
        import math
        if not isinstance(quad, list) or len(quad) != 4 or any(not isinstance(point, list) or len(point) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in point)
            for point in quad):
            raise ValueError("source road geometry is required")
        if any(type(sample.get(key)) is not int or sample[key] <= 0 for key in ("width", "height")):
            raise ValueError("positive original source dimensions required")
    if plan["budget"].get("sample_count") != len(samples) or plan["budget"].get("total_updates") != updates * len(samples):
        raise ValueError("aggregate budget must include every sample")
    if plan.get("evaluation_policy") != "each_patch_only_on_its_own_attack_input":
        raise ValueError("sample-specific evaluation mapping is required")
    if plan.get("test_data_opened") is not False:
        raise ValueError("test access is forbidden")
    if plan.get("optimization_on_original_validation") is not any(s["source_split"] == "validation" for s in samples):
        raise ValueError("optimization on original validation must be disclosed accurately")
    if plan.get("held_out_image_generalization") is not False or plan.get("gpu_validation_complete") is not False:
        raise ValueError("no held-out image or GPU validation claim is established by this prepared plan")
    if executable:
        if plan.get("mode") != "diagnostic" or updates != 25 or len(samples) != 2:
            raise ValueError("only the fixed two-sample GPU diagnostic is currently executable")
        if any(s["source_split"] != "train" for s in samples):
            raise ValueError("diagnostic inputs must come from train")
        if {s["dataset"] for s in samples} != {"kitti", "nuscenes"}:
            raise ValueError("diagnostic requires one input from each dataset")
        if tuple(s["sample_id"] for s in samples) != DIAGNOSTIC_IDS:
            raise ValueError("diagnostic sample IDs and order must remain fixed")
    else:
        master, shards = _validate_master(plan)
        if formal:
            if plan.get("mode") != "sample_specific_development_shard" or updates != 500 or plan.get("shard_id") not in range(SHARD_COUNT):
                raise ValueError("formal shard requires50 owned samplesx500 with declared shard identity")
            if samples != shards[str(plan["shard_id"])]:
                raise ValueError("formal samples differ from fixed shard partition")
            if plan.get("gpu_diagnostic_complete") is not True:
                raise ValueError("formal execution requires an accepted real diagnostic")
            _validate_formal_bindings(plan)
        elif samples != master:
            raise ValueError("formal master draft must preserve frozen200 rows")
    return plan


def load_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(plan)
    plan["_plan_sha256"] = sha256(path)
    return plan


def select_sample(plan, sample_id):
    matches = [s for s in plan["samples"] if s["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError("sample ID is missing or ambiguous in the fixed plan")
    return copy.deepcopy(matches[0])


def record_for_sample(plan, sample, records):
    if select_sample(plan, sample["sample_id"]) != sample:
        raise ValueError("sample selection was changed")
    matches = [r for r in records if r.sample_id == sample["sample_id"]]
    if len(matches) != 1:
        raise ValueError("sample is missing or duplicate in the source manifest")
    record = matches[0]
    expected = (sample["dataset"], sample["source_split"], sample["source_sha256"], sample["road_quad_norm"])
    if (record.dataset, record.split, record.source_sha256, record.road_quad_norm) != expected:
        raise ValueError("sample/source provenance differs")
    if record.split == "test":
        raise ValueError("test samples cannot be optimized")
    return record
