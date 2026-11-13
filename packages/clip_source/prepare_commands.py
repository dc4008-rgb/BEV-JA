#!/usr/bin/env python3
"""Write an explicit command matrix; never execute a process, SSH, or GPU call.

Run this on the prepared host after freeze_followup_plan.py, with live paths in a
config JSON. It checks the plan against the historical training evidence and the
new source package against its sealed SHA. It does not assert runtime/asset/GPU
readiness, create experiment directories, run any command, or power off a host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = "r2-joint-clip120-command-matrix-v1"
INPUT_SCHEMA = "r2-joint-clip120-collection-inputs-v1"
REQUIRED_CONFIG = {"python", "hostname", "child", "r2_source", "v2_source", "reference_run", "off_run", "output_root"}
OPTIONAL_CONFIG = {"sd35_snapshot", "weights_root", "source_prefix", "source_root"}
GROUP_ORDER = ("cap120",)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_sha(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "explicit lowercase SHA256 is required")
    return value


def absolute_path(value, label):
    require(isinstance(value, str) and value and not any(ord(c) < 32 for c in value), label + " must be a path string")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, label + " must be an absolute path without '..'")
    return path


def validate_config(config):
    require(isinstance(config, dict), "config must be a JSON object")
    require(REQUIRED_CONFIG <= set(config), "missing config keys: " + repr(sorted(REQUIRED_CONFIG - set(config))))
    require(not (set(config) - REQUIRED_CONFIG - OPTIONAL_CONFIG), "unknown config keys")
    hostname = config["hostname"]
    require(isinstance(hostname, str) and hostname and not any(c.isspace() or ord(c) < 32 for c in hostname),
            "hostname must be the explicit live hostname")
    require((config.get("source_prefix") is None) == (config.get("source_root") is None),
            "source remapping needs both source_prefix and source_root")
    for name, value in config.items():
        if name != "hostname" and value is not None:
            absolute_path(value, name)
    for name in REQUIRED_CONFIG - {"hostname"}:
        require(config[name] is not None, name + " cannot be null")
    output_root = Path(config["output_root"])
    for name in ("child", "r2_source", "v2_source", "reference_run", "off_run", "sd35_snapshot", "weights_root", "source_root"):
        if config.get(name):
            protected = Path(config[name])
            require(output_root != protected and protected not in output_root.parents,
                    "output_root must be outside source, asset, data, and historical reference trees")
    return dict(config)


def validate_groups(plan):
    require(plan.get("schema") == "r2-joint-clip120-followup-plan-v1"
            and plan.get("status") == "fixed_user_requested_post_evaluation_followup", "fixed cap120 followup plan required")
    require(plan.get("selection_context") == "user_requested_120_after_six_group_development_results"
            and plan.get("exploratory_posthoc") is True, "post-evaluation context required")
    groups = plan.get("groups")
    require(isinstance(groups, list) and len(groups) == 1, "exactly one cap120 group required")
    group = groups[0]
    require(group.get("id") == "cap120" and type(group.get("max_norm")) in (float, int)
            and group["max_norm"] == 120.0 and type(group.get("executed_updates")) is int
            and group["executed_updates"] == 500 and group.get("fresh_start") is True, "only fresh cap120 and fixed500 are accepted")
    return groups


def build_commands(config, plan, *, plan_path, plan_sha, package_sha, baselines_sha, control_root,
                   package_root=ROOT):
    """Pure argv construction. main also verifies the historical/OFF plan and sealed package."""
    config = validate_config(config)
    validate_groups(plan)
    plan_path = absolute_path(str(plan_path), "plan")
    package_root = absolute_path(str(package_root), "package_root")
    control_root = absolute_path(str(control_root), "control_root")
    output_root = Path(config["output_root"])
    require(output_root != package_root and package_root not in output_root.parents,
            "output_root must be outside the sealed new source package")
    for digest in (plan_sha, package_sha, baselines_sha):
        require_sha(digest)
    inputs_path = control_root / "COLLECTION_INPUTS.json"
    comparison_path = output_root / "CAP120_COMPARISON.json"
    baselines_path = package_root / "BASELINE_REFERENCES.json"
    base = output_root / "cap120"
    paths = {"id": "cap120", "run": str(base / "run"), "evaluation": str(base / "evaluation"),
             "audit_depth": str(base / "audit_depth"), "audit_detection": str(base / "audit_detection")}
    inputs = {"schema": INPUT_SCHEMA, "grid_sha256": plan_sha, "baselines_sha256": baselines_sha,
              "training_source": str(package_root), "training_package_manifest_sha256": package_sha,
              "groups": [paths]}
    commands = []
    def add(identifier, stage, argv):
        commands.append({"id": identifier, "stage": stage, "group_id": "cap120", "argv": argv,
            "cwd": str(package_root), "environment": {"PYTHONHASHSEED": "20260919", "PYTHONDONTWRITEBYTECODE": "1"},
            "depends_on": [commands[-1]["id"]] if commands else [], "required_exit_code": 0,
            "automatic_retry": False, "execution_mode": "argv_without_shell", "action_performed_by_generator": False})
    def optional(argv, names):
        for name in names:
            if config.get(name) is not None:
                argv.extend(["--" + name.replace("_", "-"), config[name]])
    train = [config["python"], str(package_root / "clip_training.py"),
        "--child", config["child"], "--r2-source", config["r2_source"], "--v2-source", config["v2_source"],
        "--reference-run", config["reference_run"], "--off-run", config["off_run"],
        "--clip-plan", str(plan_path), "--clip-plan-sha256", plan_sha, "--clip-group", "cap120",
        "--depth-weight", "1.0", "--output", paths["run"], "--expected-hostname", config["hostname"],
        "--package-manifest-sha256", package_sha]
    optional(train, ("sd35_snapshot", "source_prefix", "source_root", "weights_root"))
    add("train_cap120", "training", train)
    common = ["--r2-source", config["r2_source"], "--child", config["child"],
        "--expected-hostname", config["hostname"], "--training-source", str(package_root),
        "--training-package-manifest-sha256", package_sha]
    evaluation = [config["python"], str(package_root / "clip_evaluate.py"), *common,
        "--run", paths["run"], "--output", paths["evaluation"]]
    optional(evaluation, ("weights_root", "source_prefix", "source_root"))
    add("evaluate_cap120", "evaluation", evaluation)
    for task in ("depth", "detection"):
        argv = [config["python"], str(package_root / ("clip_audit_" + task + ".py")), *common,
                "--run", paths["evaluation"], "--output", paths["audit_" + task]]
        if task == "detection":
            argv.extend(["--training-run", paths["run"], "--evaluation-source", str(package_root / "clip_evaluate.py")])
        add("audit_" + task + "_cap120", "raw_audit", argv)
    add("collect_cap120", "collection", [config["python"], str(package_root / "collect_cap120.py"),
        "--plan", str(plan_path), "--baselines", str(baselines_path), "--inputs", str(inputs_path),
        "--output", str(comparison_path)])
    matrix = {"schema": SCHEMA, "status": "commands_prepared_not_executed", "plan_path": str(plan_path),
        "plan_sha256": plan_sha, "package_root": str(package_root), "package_manifest_sha256": package_sha,
        "baselines_sha256": baselines_sha, "collector_inputs_path": str(inputs_path), "runtime_config": config,
        "canonical_group_count": 1, "total_training_updates": 500, "commands": commands,
        "required_output_parent_directories": [str(output_root), str(base)],
        "output_parent_policy": "Create only these fresh parents; all stage output directories must remain absent.",
        "execution_policy": "Run all five fixed commands in order; stop on any nonzero exit; no retry, checkpoint selection or additional threshold selection.",
        "all_training_precedes_new_evaluation": True, "selection_context": plan["selection_context"],
        "exploratory_posthoc": True, "preregistered_grid_point": False,
        "runtime_assets_or_gpu_ready": False, "commands_executed": False, "ssh_executed": False,
        "power_action_scheduled": False,
        "limitations": ["One user-requested threshold after viewing six-group development results.",
            "This argv manifest does not execute training or establish a critical threshold.",
            "Historical OFF/environment verification must pass before the new cap120 training.",
            "No test access or automatic promotion; all comparisons remain exploratory and descriptive."]}
    return matrix, inputs


def _write_exclusive(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from clip_evidence import load_plan, require_completed_off
    from reference_long_training import verify_tree
    config_raw = args.config.read_bytes()
    config = validate_config(json.loads(config_raw))
    package_verification = verify_tree(ROOT, require_sha(args.package_manifest_sha256))
    plan, policy = load_plan(args.plan, require_sha(args.plan_sha256), "cap120", Path(config["reference_run"]))
    off_verification = require_completed_off(Path(config["off_run"]), policy, args.package_manifest_sha256,
        plan=plan, reference_run=Path(config["reference_run"]))
    output_dir = args.output_dir.absolute()
    for path in (output_dir, *output_dir.parents):
        require(not path.is_symlink(), "output-dir symlinks forbidden")
    if output_dir.exists():
        raise FileExistsError("output-dir exists; never overwrite a prepared command matrix")
    require(output_dir.parent.is_dir(), "output-dir parent must already exist")
    require(not Path(config["output_root"]).exists(), "experiment output_root must be fresh before command preparation")
    for protected in (ROOT, Path(config["reference_run"]), Path(config["child"]), Path(config["r2_source"]), Path(config["v2_source"]), Path(config["off_run"])):
        require(output_dir != protected and protected not in output_dir.parents,
                "command evidence must be outside sealed source and historical trees")
    baselines_sha = sha((ROOT / "BASELINE_REFERENCES.json").read_bytes())
    matrix, inputs = build_commands(config, plan, plan_path=args.plan.resolve(), plan_sha=args.plan_sha256,
        package_sha=args.package_manifest_sha256, baselines_sha=baselines_sha, control_root=output_dir)
    matrix["verification"] = {"fixed_user_followup_plan_and_references_verified": True,
        "source_package": package_verification, "runtime_config_sha256": sha(config_raw),
        "execution_off": off_verification}
    output_dir.mkdir(exist_ok=False)
    inputs_sha = _write_exclusive(output_dir / "COLLECTION_INPUTS.json", inputs)
    matrix["collector_inputs_sha256"] = inputs_sha
    matrix_sha = _write_exclusive(output_dir / "COMMANDS.json", matrix)
    print(json.dumps({"status": matrix["status"], "command_count": len(matrix["commands"]),
        "commands_sha256": matrix_sha, "collector_inputs_sha256": inputs_sha,
        "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
