"""Generated B3 controller/depth_beta override; do not edit by hand."""
CONTROLLER_OVERRIDES = {}
DEPTH_BETA = None

_APPLIED = False


def apply():
    """Patch the parent sample_contract lazily; call only after it is importable."""
    global _APPLIED
    if _APPLIED:
        return
    import sample_contract as _sample_contract
    if CONTROLLER_OVERRIDES:
        _sample_contract.CONTROLLER_CONFIG.update(CONTROLLER_OVERRIDES)
        _sample_contract.BASE_CONFIG["controller_config"].update(CONTROLLER_OVERRIDES)
    if DEPTH_BETA is not None:
        _sample_contract.BASE_CONFIG["depth_beta"] = float(DEPTH_BETA)
    _APPLIED = True


def validate_plan(plan, sample_contract):
    """B3 relaxed plan guard: same per-sample/science checks, but the derived
    nuScenes-only cohort need not match the parent's fixed 25/25 shard receipts."""
    import json
    import math
    import re
    from pathlib import Path
    if plan.get("schema") != sample_contract.SCHEMA or plan.get("identity") != sample_contract.IDENTITY:
        raise ValueError("sample-specific plan identity differs")
    if plan.get("status") != "frozen_formal_shard":
        raise ValueError("B3 derived plan must retain the accepted formal status")
    samples = plan.get("samples")
    if not isinstance(samples, list) or len(samples) != 50:
        raise ValueError("B3 plan must contain exactly 50 nuScenes samples")
    ids, cases = set(), set()
    for sample in samples:
        sample_id, case_id = sample.get("sample_id"), sample.get("case_id")
        if not isinstance(sample_id, str) or sample_id in ids:
            raise ValueError("duplicate or missing sample ID")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", case_id) or case_id in cases:
            raise ValueError("duplicate or unsafe case directory")
        ids.add(sample_id); cases.add(case_id)
        if sample.get("dataset") != "nuscenes":
            raise ValueError("B3 plan is nuScenes-only")
        if sample.get("source_split") not in ("train", "validation"):
            raise ValueError("test or unknown inputs cannot be optimized")
        if not re.fullmatch(r"[0-9a-f]{64}", sample.get("source_sha256", "")):
            raise ValueError("source image SHA is required")
        if not Path(sample.get("source_path", "")).is_absolute():
            raise ValueError("original source path must be recorded")
        quad = json.loads(sample["road_quad_norm"])
        if not isinstance(quad, list) or len(quad) != 4 or any(
                not isinstance(point, list) or len(point) != 2 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in point)
                for point in quad):
            raise ValueError("source road geometry is required")
        if any(type(sample.get(key)) is not int or sample[key] <= 0 for key in ("width", "height")):
            raise ValueError("positive original source dimensions required")
    if (plan["budget"].get("sample_count") != 50 or plan["budget"].get("updates_per_sample") != 500
            or plan["budget"].get("total_updates") != 25000):
        raise ValueError("B3 plan must retain the accepted 50x500 metadata")
    if plan.get("optimization_on_original_validation") is not any(s["source_split"] == "validation" for s in samples):
        raise ValueError("optimization on original validation must be disclosed accurately")
    if plan.get("evaluation_policy") != "each_patch_only_on_its_own_attack_input":
        raise ValueError("sample-specific evaluation mapping is required")
    if plan.get("test_data_opened") is not False:
        raise ValueError("test access is forbidden")
    if plan.get("scientific_config") != sample_contract.scientific_config_for(
            plan["ablation_variant"], 500, switches=plan["scientific_config"]["switches"]):
        raise ValueError("B3 plan scientific configuration differs")
    return True
