#!/usr/bin/env python3
"""Freeze an exploratory clipping grid from the original joint500 TRAIN trace.

No model, evaluation file, validation metric, or test data is loaded. This is a
plan builder, not a training launcher. The inputs must be the original completed
joint500 ``run/summary.json`` and ``run/attack_trace.jsonl``. Their authenticity
must also be checked against the sealed historical retention manifest at launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "r2-joint-clip-train-grid-v1"
IDENTITY = "r2_joint_depth_long500_20261001"
TRAINING_MANIFEST_SHA = "38ae14801d2e330236bd7b408f08ef29c5f93b4e802c6a8f2414a5a352ceb23f"
MODEL_MANIFEST_SHA = "d994315183884c678c35e1b4cdb03c0b459b74ba27fd16bc95e3c17717c0b5dc"
PACKAGE_MANIFEST_SHA = "b1a9086bc9a2595cc9a32a0692e884047efb12d63ffd18c67a8026a96dd4b70f"
CHILD_MANIFEST_SHA = "a4ee2aab2ab40fe9c122c16bc05fd5a32cbd06286413384f4ad4086c26b863cb"
N_UPDATES = 500
SEED = 20260919
CONSISTENCY_REL_TOL = 1e-12
CONSISTENCY_ABS_TOL = 1e-12
QUANTILES = (("p90", .90), ("p50", .50), ("p10", .10), ("p1", .01))
CONFIG = {"num_inference_steps": 100, "strength": .25, "outer_rounds": 20,
          "latent_lr": .05, "step_feedback_mode": "latent_chain", "guidance_scale": 4.5,
          "seed": SEED, "prompt": "top-down aerial view of clean asphalt road surface",
          "negative_prompt": "vehicle, car, obstacle", "active_steps_per_round": 25,
          "adversarial_updates": 500, "denoise_steps": 500, "noise_samples": 20}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _load_json(data):
    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    _finite_tree(value)
    return value


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite number in input")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite real number")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(name + " must be finite" + (" and nonnegative" if nonnegative else ""))
    return float(value)


def _integer(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError(name + " must be a " + ("positive" if positive else "nonnegative") + " integer")
    return value


def _expected(mapping, name, value):
    if (name not in mapping or mapping[name] != value
            or (type(mapping[name]) is bool) != (type(value) is bool)):
        raise ValueError("unexpected " + name + "; expected " + repr(value))


def validate_summary(summary):
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON object")
    expected = {"status": "train_optimization_complete", "identity": IDENTITY,
        "candidate_id": IDENTITY + "-two-detector-foa-plus-fixed-depth",
        "run_role": "fresh_joint500_training", "executed_updates": N_UPDATES,
        "seed": SEED, "depth_weight": 1.0, "latent_lr": .05,
        "gradient_multiplier": 100000.0, "normalizations": [], "update_clamp": None,
        "training_models": ["fasterrcnn", "ssd300"],
        "training_depth_models": ["monodepth2", "dpt_large"],
        "training_tasks": ["detection", "depth"], "depth_disabled_path": False,
        "training_manifest_sha256": TRAINING_MANIFEST_SHA,
        "model_manifest_sha256": MODEL_MANIFEST_SHA,
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA,
        "historical_source_manifest_sha256": CHILD_MANIFEST_SHA,
        "fresh_start": True, "test_data_opened": False, "validation_data_opened": False,
        "checkpoint_selection": "fixed final500 only"}
    for name, value in expected.items():
        _expected(summary, name, value)
    if not isinstance(summary.get("config"), dict):
        raise ValueError("missing scientific config")
    for name, value in CONFIG.items():
        _expected(summary["config"], name, value)
    for name in ("attack_trace_sha256", "schedule_sha256", "state_evidence_manifest_sha256"):
        value = summary.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError("invalid SHA256: " + name)


def _tensor_norm(statistics, name):
    """Read the actual components.tensor_statistics schema, never RMS as L2."""
    if not isinstance(statistics, dict):
        raise ValueError("missing tensor statistics: " + name)
    count = _integer(statistics.get("numel"), name + ".numel", positive=True)
    _expected(statistics, "all_finite", True)
    _expected(statistics, "finite_count", count)
    for key in ("nan_count", "positive_inf_count", "negative_inf_count"):
        _expected(statistics, key, 0)
    shape = statistics.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError("invalid tensor shape: " + name)
    for size in shape:
        _integer(size, name + ".shape", positive=True)
    if math.prod(shape) != count:
        raise ValueError("tensor shape and numel disagree: " + name)
    stats = statistics.get("finite_component_statistics")
    if not isinstance(stats, dict):
        raise ValueError("missing finite_component_statistics: " + name)
    if "l2" not in stats and "rms" not in stats:
        raise ValueError("need l2 or rms with numel: " + name)
    l2 = _number(stats["l2"], name + ".l2", nonnegative=True) if "l2" in stats else None
    rms = _number(stats["rms"], name + ".rms", nonnegative=True) if "rms" in stats else None
    derived = rms * math.sqrt(count) if rms is not None else None
    if derived is not None and not math.isfinite(derived):
        raise ValueError("derived L2 is nonfinite")
    if l2 is not None and derived is not None and not math.isclose(
            l2, derived, rel_tol=CONSISTENCY_REL_TOL, abs_tol=CONSISTENCY_ABS_TOL):
        raise ValueError("L2 and RMS*sqrt(numel) disagree: " + name)
    if l2 is None:
        return derived, "derived_rms_times_sqrt_numel", count, shape
    return l2, "reported_l2", count, shape


def validate_trace(data, summary):
    if _sha(data) != summary["attack_trace_sha256"]:
        raise ValueError("attack trace SHA256 differs from the completed summary")
    lines = data.splitlines()
    if len(lines) != N_UPDATES or any(not line.strip() for line in lines):
        raise ValueError("expected exactly 500 nonblank training trace rows")
    norms, sources, canonical_shape = [], [], None
    for step, line in enumerate(lines, 1):
        row = _load_json(line)
        if not isinstance(row, dict):
            raise ValueError("trace row must be an object")
        for key, value in {"global_step": step, "split": "train", "depth_weight": 1.0,
            "training_models": ["fasterrcnn", "ssd300"],
            "training_depth_models": ["monodepth2", "dpt_large"],
            "training_tasks": ["detection", "depth"],
            "mathematical_objective": "R2_detector_FOA_plus_fixed_depth_semantic_target_far"}.items():
            _expected(row, key, value)
        updates = row.get("update_statistics")
        if not isinstance(updates, dict):
            raise ValueError("missing per-step update statistics")
        for key, value in {"coefficient": 100000.0, "lr": .05,
            "normalization": "none", "clamp": "none", "inputs_modified": False}.items():
            _expected(updates, key, value)
        tensors = row.get("tensor_statistics")
        if not isinstance(tensors, dict):
            raise ValueError("missing tensor_statistics")
        for key in ("latent", "weighted_loss_gradient", "amplified_gradient", "proposed_update",
                    "candidate_latent", "realized_delta_float32"):
            _tensor_norm(tensors.get(key), key)
        norm, source, count, shape = _tensor_norm(tensors["amplified_gradient"], "amplified_gradient")
        if canonical_shape is None:
            canonical_shape = shape
        if shape != canonical_shape:
            raise ValueError("amplified-gradient shape varies within the original run")
        _expected(updates, "numel", count)
        _expected(updates, "shape", shape)
        norms.append(norm)
        sources.append(source)
    return norms, sources, canonical_shape


def linear_quantile(values, probability):
    """NumPy/PyTorch linear interpolation convention: index=(n-1)*q."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    # The convex form avoids subtracting two potentially very large finite values.
    value = ordered[lo] * (1.0 - (index - lo)) + ordered[hi] * (index - lo)
    if not math.isfinite(value):
        raise ValueError("quantile is nonfinite")
    return value


def build_plan(summary_path, trace_path):
    summary_path, trace_path = Path(summary_path), Path(trace_path)
    summary_bytes, trace_bytes = summary_path.read_bytes(), trace_path.read_bytes()
    summary = _load_json(summary_bytes)
    validate_summary(summary)
    norms, sources, shape = validate_trace(trace_bytes, summary)
    groups = []
    requested = [("off", None)] + [(name, linear_quantile(norms, q)) for name, q in QUANTILES] + [("zero", 0.0)]
    for alias, threshold in requested:
        same = next((x for x in groups if x["max_norm"] == threshold), None)
        if same is not None:
            same["aliases"].append(alias)
            continue
        exceeded = sum(value > threshold for value in norms) if threshold is not None else 0
        groups.append({"id": alias, "aliases": [alias], "max_norm": threshold,
            "executed_updates": N_UPDATES, "seed": SEED, "fresh_start": True,
            "reference_counterfactual_exceed_count": exceeded,
            "reference_counterfactual_exceed_fraction": exceeded / N_UPDATES})
    return {"schema": SCHEMA, "status": "thresholds_frozen_from_train",
        "candidate_id": "r2_joint_clip_sweep_20261003", "seed": SEED,
        "scope": "single_seed_exploratory_fixed_grid", "reference": {
            "identity": IDENTITY, "summary_path": str(summary_path.absolute()),
            "trace_path": str(trace_path.absolute()), "summary_sha256": _sha(summary_bytes),
            "trace_sha256": _sha(trace_bytes), "training_manifest_sha256": TRAINING_MANIFEST_SHA,
            "model_manifest_sha256": MODEL_MANIFEST_SHA, "package_manifest_sha256": PACKAGE_MANIFEST_SHA,
            "schedule_sha256": summary["schedule_sha256"],
            "state_evidence_manifest_sha256": summary["state_evidence_manifest_sha256"],
            "trace_rows": N_UPDATES, "declared_split": "train",
            "launch_requirement": "also verify summary and trace hashes against the historical sealed retention manifest"},
        "clipping": {"norm_type": "L2", "position": "after-amplification-before-lr",
            "gradient_multiplier": 100000.0, "latent_lr": .05,
            "depth_weight": 1.0, "tensor": "combined_detector_plus_depth_latent_gradient",
            "norm_measurement": "CPU float64 statistics of the original amplified tensor"},
        "threshold_rule": {"requested_groups": [name for name, _ in requested],
            "quantiles": {name: q for name, q in QUANTILES},
            "interpolation": "linear: index=(n-1)*q",
            "equal_thresholds": "deduplicate exact numeric equality and retain all aliases",
            "validation_or_test_used": False, "post_evaluation_refinement_allowed": False},
        "reference_gradient_scale": {"shape": shape, "numel": math.prod(shape),
            "minimum_l2": min(norms), "maximum_l2": max(norms),
            "reported_l2_steps": sources.count("reported_l2"),
            "derived_l2_steps": sources.count("derived_rms_times_sqrt_numel"),
            "derivation": "rms*sqrt(numel), only when l2 is absent",
            "l2_rms_consistency_rel_tol": CONSISTENCY_REL_TOL,
            "l2_rms_consistency_abs_tol": CONSISTENCY_ABS_TOL},
        "groups": groups, "unique_run_count": len(groups),
        "required_updates_per_group": N_UPDATES, "required_total_updates": len(groups) * N_UPDATES,
        "measurement_notes": [
            "Exceedance fractions describe the original un-clipped train trajectory only; they are not measured clipping rates of new runs.",
            "New clipped trajectories can have different gradients and exceedance rates; record actual rates during each full run.",
            "The zero threshold removes adversarial latent updates but preserves diffusion and its random draws; it is not the clean/zero patch evaluation state.",
            "No validation evaluation or test result participates in threshold selection, checkpoint selection, stopping, or grid refinement.",
            "This plan does not establish a unique monotonic critical threshold, statistical significance, or equal final RGB perturbation budgets."]}


def write_plan(plan, output):
    output = Path(output).absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError("output symlinks forbidden")
    if output.exists():
        raise FileExistsError("output already exists; do not overwrite a frozen plan")
    encoded = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Parent must already exist: a typo must not create an unexpected output tree.
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(args.summary, args.trace)
    digest = write_plan(plan, args.output)
    print(json.dumps({"status": plan["status"], "schema": SCHEMA,
        "groups": len(plan["groups"]), "output": str(args.output), "sha256": digest}))


if __name__ == "__main__":
    main()
