"""Strict provenance for the separately declared, user-selected cap120 follow-up.

No model imports, dataset reads, SSH, or experiment execution. An execution OFF
may have new host/driver provenance only after its full 2041 scientific states
match the immutable original joint500; cap120 then must match that OFF runtime.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN_SCHEMA = "r2-joint-clip120-followup-plan-v1"
PLAN_STATUS = "fixed_user_requested_post_evaluation_followup"
SELECTION_CONTEXT = "user_requested_120_after_six_group_development_results"
ENVIRONMENT_FIELDS = (
    "device", "python", "torch", "torchvision", "diffusers", "transformers", "numpy",
    "cuda", "cudnn_version", "deterministic_algorithms", "deterministic_warn_only",
    "cudnn_deterministic", "cudnn_benchmark", "cudnn_allow_tf32", "matmul_allow_tf32",
    "float32_matmul_precision", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
    "intraop_threads", "interop_threads", "scheduler_class", "scheduler_source_sha256",
    "latent_dtype", "new_eot_sampling_device", "global_spatial_override",
    "scoped_projection_override", "r2_source_manifest_sha256", "depth_weight",
    "depth_models", "depth_objective", "detector_foa_unchanged",
    "accepted_joint_v2_manifest_sha256", "fresh_start", "total_updates",
)
POLICY_FIXED = {"scope": "global_total_latent_gradient",
                "position": "after_100000_before_lr", "norm_dtype": "float32"}
REQUIRED_SCIENCE = {"updates": 500, "seed": 20260919, "latent_lr": .05,
                    "gradient_multiplier": 100000.0, "depth_weight": 1.0,
                    "normalizations": [], "update_clamp": None,
                    "fresh_start": True, "checkpoint_selection": "fixed final500 only"}
OFF_ORIGINS = ("pinned_historical_off",
               "independent_off500_same_original_package_full2041_exact")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA256 required")
    return value


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates,
                      parse_constant=nonfinite)


def followup_bindings():
    return read_json(ROOT / "FOLLOWUP_BINDINGS.json")


def reference_bindings():
    return read_json(ROOT / "REFERENCE_BINDINGS.json")


def bound_path(root, relative, expected_sha):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("unsafe evidence member")
    path = Path(root).absolute() / relative
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("evidence symlinks forbidden: " + str(path))
    if not path.is_file() or sha(path) != require_sha(expected_sha):
        raise ValueError("evidence SHA differs: " + str(path))
    return path


def require_historical_reference(reference_run):
    files = reference_bindings()["files"]
    for relative, digest in files.items():
        bound_path(reference_run, relative, digest)
    return dict(files)


def scientific_environment(runtime):
    missing = [name for name in ENVIRONMENT_FIELDS if name not in runtime]
    if missing:
        raise ValueError("OFF scientific runtime fields missing: " + repr(missing))
    values = {name: runtime[name] for name in ENVIRONMENT_FIELDS}
    gpu = runtime.get("gpu_identity")
    if not isinstance(gpu, str) or len(gpu.split(",")) != 3:
        raise ValueError("one explicit GPU name, UUID and driver tuple required")
    model, uuid, driver = [item.strip() for item in gpu.split(",")]
    if not model or not uuid.startswith("GPU-") or not driver:
        raise ValueError("invalid recorded GPU identity")
    values.update(gpu_model=model, gpu_driver=driver)
    return {"values": values, "sha256": json_fingerprint(values)}


def validate_environment_contract(contract):
    if not isinstance(contract, dict) or set(contract) != {"values", "sha256"}:
        raise ValueError("complete scientific environment contract required")
    values = contract["values"]
    if (not isinstance(values, dict)
            or set(values) != set(ENVIRONMENT_FIELDS) | {"gpu_model", "gpu_driver"}
            or json_fingerprint(values) != require_sha(contract["sha256"])):
        raise ValueError("scientific environment fingerprint differs")
    return contract


def validate_followup_plan(plan):
    bindings = followup_bindings()
    if (not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA
            or plan.get("status") != PLAN_STATUS
            or plan.get("selection_context") != SELECTION_CONTEXT
            or plan.get("exploratory_posthoc") is not True):
        raise ValueError("explicit post-evaluation cap120 follow-up plan required")
    if (plan.get("groups") != [{"id": "cap120", "aliases": ["cap120"], "max_norm": 120.0,
                               "executed_updates": 500, "fresh_start": True}]
            or type(plan["groups"][0]["max_norm"]) not in (int, float)
            or type(plan["groups"][0]["executed_updates"]) is not int
            or plan["groups"][0].get("fresh_start") is not True
            or plan.get("unique_run_count") != 1 or type(plan.get("unique_run_count")) is not int
            or plan.get("required_total_updates") != 500
            or type(plan.get("required_total_updates")) is not int):
        raise ValueError("only one literal cap120 group and exactly 500 updates are allowed")
    if json_fingerprint(plan.get("scientific_configuration")) != json_fingerprint(REQUIRED_SCIENCE):
        raise ValueError("frozen scientific configuration differs")
    if (plan.get("reference") != bindings["reference"]
            or plan.get("historical_off") != {"package_sha256": bindings["prior_package_sha256"],
                                              "files": bindings["off_files"]}
            or plan.get("source_evidence") != {
                "prior_grid_sha256": bindings["prior_grid_sha256"],
                "seen_comparison_sha256": bindings["seen_comparison_sha256"],
                "seen_points_sha256": bindings["seen_points_sha256"],
                "baselines_sha256": bindings["baselines_sha256"]}):
        raise ValueError("follow-up source evidence differs from pinned completed six-group evidence")
    expected_selection = {
        "threshold_literal": 120.0, "selection_basis": "explicit_user_request",
        "previous_development_results_seen": True, "train_quantile": False,
        "test_used_for_selection": False, "automatic_refinement": False,
        "promotion_to_default": False, "training_opens_validation_or_test": False,
    }
    if plan.get("selection") != expected_selection:
        raise ValueError("follow-up must disclose post-evaluation user selection and no test use")
    control = plan.get("execution_off")
    if (not isinstance(control, dict) or control.get("origin") not in OFF_ORIGINS
            or control.get("package_sha256") != bindings["prior_package_sha256"]
            or control.get("grid_sha256") != bindings["prior_grid_sha256"]
            or control.get("compared_states") != 2041 or type(control.get("compared_states")) is not int
            or control.get("tolerance") != 0 or type(control.get("tolerance")) is not int
            or control.get("state_comparison") != "all_original_joint500_scientific_fingerprints_exact"):
        raise ValueError("execution OFF must establish the original complete 2041-state exact gate")
    if set(control.get("files", {})) != set(bindings["off_files"]):
        raise ValueError("execution OFF evidence file set differs")
    for digest in control["files"].values():
        require_sha(digest)
    if control["files"]["CLIP_PLAN.json"] != bindings["prior_grid_sha256"]:
        raise ValueError("execution OFF must use the original sealed six-group plan")
    if (control["origin"] == OFF_ORIGINS[0]) != (control["files"] == bindings["off_files"]):
        raise ValueError("historical/new OFF origin label disagrees with file identities")
    validate_environment_contract(control.get("scientific_environment"))
    return plan


def _check_off_structure(root, reference_run):
    """Hash saved evidence and compare states, without loading any tensor."""
    bindings = followup_bindings()
    root = Path(root)
    summary = read_json(root / "summary.json")
    fixed = {
        "status": "train_optimization_complete", "identity": bindings["prior_identity"],
        "candidate_id": bindings["prior_candidate_id"], "run_role": bindings["prior_run_role"],
        "package_manifest_sha256": bindings["prior_package_sha256"], "executed_updates": 500,
        "depth_weight": 1.0, "seed": 20260919, "latent_lr": .05,
        "gradient_multiplier": 100000.0, "update_clamp": None, "normalizations": [],
        "test_data_opened": False, "validation_data_opened": False, "fresh_start": True,
        "resume_enabled": False, "checkpoint_selection": "fixed final500 only",
    }
    for key, expected in fixed.items():
        if summary.get(key) != expected or isinstance(summary.get(key), bool) != isinstance(expected, bool):
            raise ValueError("execution OFF contract differs: " + key)
    policy = dict(POLICY_FIXED, group_id="off", max_norm=None,
                  grid_sha256=bindings["prior_grid_sha256"])
    if summary.get("gradient_clipping") != policy:
        raise ValueError("execution OFF is not the original unclipped policy")
    fields = {
        "attack_trace.jsonl": "attack_trace_sha256", "schedule.json": "schedule_sha256",
        "state_evidence/MANIFEST.json": "state_evidence_manifest_sha256",
        "CLIPPING_COMPARISON_GATE.json": "clipping_comparison_gate_sha256",
        "runtime_provenance.json": "runtime_provenance_sha256",
    }
    files = {"summary.json": sha(root / "summary.json"),
             "CLIP_PLAN.json": bindings["prior_grid_sha256"]}
    for name, key in fields.items():
        files[name] = require_sha(summary.get(key))
    for name, digest in files.items():
        bound_path(root, name, digest)
    trace = [json.loads(line) for line in (root / "attack_trace.jsonl").read_text().splitlines()]
    if (len(trace) != 500 or any(row.get("split") != "train"
            or row.get("global_step") != index
            or type(row.get("global_step")) is not int
            or row.get("clipping_diagnostics", {}).get("clipped") is not False
            for index, row in enumerate(trace, 1))):
        raise ValueError("execution OFF trace must contain 500 unclipped train updates")
    gate = read_json(root / "CLIPPING_COMPARISON_GATE.json")
    ref_files = require_historical_reference(reference_run)
    original_summary = read_json(Path(reference_run) / "summary.json")
    for key in ("config", "budget", "training_models", "training_depth_models", "depth_beta",
                "accepted_r2_source_manifest_sha256", "accepted_joint_v2_manifest_sha256",
                "training_manifest_sha256", "model_manifest_sha256", "schedule_sha256"):
        if key not in original_summary or summary.get(key) != original_summary[key]:
            raise ValueError("execution OFF scientific metadata differs: " + key)
    reference = {"candidate_id": reference_bindings()["candidate_id"],
                 "summary_sha256": ref_files["summary.json"],
                 "state_manifest_sha256": ref_files["state_evidence/MANIFEST.json"]}
    expected_gate = {"status": "exact_full_trajectory", "compared_states": 2041,
                     "tolerance": 0, "reference": reference, "group_id": "off",
                     "max_norm": None, "grid_sha256": bindings["prior_grid_sha256"]}
    for key, value in expected_gate.items():
        if gate.get(key) != value or isinstance(gate.get(key), bool) != isinstance(value, bool):
            raise ValueError("execution OFF exact gate differs: " + key)
    manifest = read_json(root / "state_evidence/MANIFEST.json")
    original = read_json(Path(reference_run) / "state_evidence/MANIFEST.json")
    entries, reference_entries = manifest.get("entries"), original.get("entries")
    if (manifest.get("schema") != "r2-joint-clip500-lean-state-manifest-v1"
            or manifest.get("candidate_id") != bindings["prior_candidate_id"]
            or manifest.get("status") != "complete" or manifest.get("executed_updates") != 500
            or manifest.get("gradient_clipping") != policy
            or manifest.get("reference_binding") != reference
            or manifest.get("comparison_counts") != {"states_checked": 2041}
            or not isinstance(entries, list) or len(entries) != 2041
            or not isinstance(reference_entries, list) or len(reference_entries) != 2041):
        raise ValueError("execution OFF state manifest does not establish complete exact500")
    fields = ("index", "phase", "global_step", "updates_applied", "state_fingerprint")
    for index, (entry, expected) in enumerate(zip(entries, reference_entries), 1):
        if (entry.get("index") != index or entry.get("recording_preserved_rng") is not True
                or any(entry.get(key) != expected.get(key) for key in fields)):
            raise ValueError("execution OFF scientific state differs at " + str(index))
        stored = entry.get("stored_state")
        if stored is not None:
            path = bound_path(root, stored["path"], stored["sha256"])
            if path.stat().st_size != stored["bytes"]:
                raise ValueError("OFF stored-state size differs")
    ledger = bound_path(root, "state_evidence/ledger.jsonl", manifest["ledger_sha256"])
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if rows != entries:
        raise ValueError("OFF ledger differs from complete state manifest")
    runtime = read_json(root / "runtime_provenance.json")
    if runtime.get("package_manifest_sha256") != bindings["prior_package_sha256"]:
        raise ValueError("OFF runtime comes from a different package")
    environment = scientific_environment(runtime)
    return {"origin": OFF_ORIGINS[0] if files == bindings["off_files"] else OFF_ORIGINS[1],
            "package_sha256": bindings["prior_package_sha256"],
            "grid_sha256": bindings["prior_grid_sha256"],
            "files": files, "scientific_environment": environment,
            "compared_states": 2041, "tolerance": 0,
            "state_comparison": "all_original_joint500_scientific_fingerprints_exact"}


def verified_off(root, reference_run, expected_files=None):
    if expected_files is not None:
        for relative, digest in expected_files.items():
            bound_path(root, relative, digest)
    return _check_off_structure(root, reference_run)


def build_followup_plan(*, reference_run, historical_off_run, off_run,
                        comparison_file, points_file):
    bindings = followup_bindings()
    require_historical_reference(reference_run)
    verified_off(historical_off_run, reference_run, bindings["off_files"])
    execution_off = verified_off(off_run, reference_run)
    for path, expected in ((comparison_file, bindings["seen_comparison_sha256"]),
                           (points_file, bindings["seen_points_sha256"])):
        bound_path(Path(path).parent, Path(path).name, expected)
    plan = {
        "schema": PLAN_SCHEMA, "status": PLAN_STATUS,
        "selection_context": SELECTION_CONTEXT, "exploratory_posthoc": True,
        "selection": {"threshold_literal": 120.0, "selection_basis": "explicit_user_request",
            "previous_development_results_seen": True, "train_quantile": False,
            "test_used_for_selection": False, "automatic_refinement": False,
            "promotion_to_default": False, "training_opens_validation_or_test": False},
        "groups": [{"id": "cap120", "aliases": ["cap120"], "max_norm": 120.0, "executed_updates": 500, "fresh_start": True}],
        "unique_run_count": 1, "required_total_updates": 500,
        "scientific_configuration": dict(REQUIRED_SCIENCE),
        "reference": bindings["reference"],
        "historical_off": {"package_sha256": bindings["prior_package_sha256"],
                           "files": bindings["off_files"]},
        "execution_off": execution_off,
        "source_evidence": {key: bindings[key] for key in (
            "prior_grid_sha256", "seen_comparison_sha256", "seen_points_sha256", "baselines_sha256")},
    }
    return validate_followup_plan(plan)


def load_plan(path, expected_sha, group_id, reference_run):
    require_historical_reference(reference_run)
    bound_path(Path(path).parent, Path(path).name, expected_sha)
    plan = validate_followup_plan(read_json(path))
    if group_id != "cap120":
        raise ValueError("this follow-up accepts only --clip-group cap120")
    return plan, dict(POLICY_FIXED, group_id="cap120", max_norm=120.0,
                      grid_sha256=expected_sha)


def require_completed_off(off_run, policy, expected_package_sha, *, plan, reference_run):
    """Strictly bind historical or newly reverified OFF; never bypass its gate."""
    validate_followup_plan(plan)
    require_sha(expected_package_sha)
    if (policy != dict(POLICY_FIXED, group_id="cap120", max_norm=120.0,
                       grid_sha256=policy.get("grid_sha256"))):
        raise ValueError("cap120 clipping policy required")
    require_sha(policy["grid_sha256"])
    if off_run is None:
        raise ValueError("a complete bound execution OFF control is required")
    expected = plan["execution_off"]
    observed = verified_off(off_run, reference_run, expected["files"])
    if observed != expected:
        raise ValueError("execution OFF environment or evidence differs from frozen follow-up")
    return {"run": str(Path(off_run).resolve()), "control_kind": expected["origin"],
            "prior_package_sha256": expected["package_sha256"],
            "followup_package_sha256": expected_package_sha,
            "files": expected["files"], "scientific_environment": expected["scientific_environment"],
            "compared_states": 2041, "tolerance": 0}


def require_matching_environment(runtime, off_binding):
    observed = scientific_environment(runtime)
    expected = validate_environment_contract(off_binding["scientific_environment"])
    if observed != expected:
        changed = sorted(k for k in expected["values"] if observed["values"].get(k) != expected["values"][k])
        raise ValueError("cap120 runtime differs from verified execution OFF: " + repr(changed))
    return {"status": "exact_scientific_environment", "sha256": observed["sha256"],
            "excluded_ephemeral_fields": ["hostname", "gpu_uuid", "python_executable",
                                         "output_paths", "package_manifest_sha256", "protocol_sha256"],
            "driver_exemption": False}


def verify_scientific_kernels(root=ROOT):
    """Bind exact helper source text; AST node schemas vary across Python versions."""
    bindings = followup_bindings()
    root = Path(root)
    for name, digest in bindings["scientific_source_files"].items():
        bound_path(root, name, digest)
    expected = bindings["training_scientific_function_source_sha256"]
    source = (root / "clip_training.py").read_text(encoding="utf-8")
    observed = {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode("utf-8")).hexdigest()
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name not in ("parse_args", "main")}
    if observed != expected:
        raise ValueError("follow-up scientific training helpers differ from original six-group implementation")
    return {"status": "unchanged_scientific_functions_and_update_sources",
            "function_count": len(expected), "source_training_sha256": bindings["source_training_sha256"],
            "comparison": "exact_utf8_function_source_text_excluding_main_and_parse_args"}
