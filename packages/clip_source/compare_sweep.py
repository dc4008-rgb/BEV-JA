"""Describe all fixed-grid baseline crossings; never choose or refine a threshold.

Read audited numerical points only: no model, dataset, training, or inference
imports. Off is a separate reference, never the endpoint of a finite interval.
An opposite-sign adjacent pair is an observed bracket, not a claim of continuity
or a numerically interpolated critical value. These are exploratory comparisons
of prediction change, not ground-truth error, significance, or causal gains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re


PLAN_SCHEMA = "r2-joint-clip-train-grid-v1"
BASELINES_SCHEMA = "r2-joint-clip-baselines-v1"
POINTS_SCHEMA = "r2-joint-clip-evaluated-points-v1"
OUTPUT_SCHEMA = "r2-joint-clip-exploratory-comparison-v1"
DIRECTIONS = {
    "larger_is_more_prediction_deviation": (
        "Higher is a larger clean-reference prediction deviation; it does not establish "
        "a farther-direction attack or greater error against ground truth."
    ),
    "larger_is_more_AP50_drop": (
        "Higher is a larger clean-reference relative AP50 drop; the difference between "
        "percentage-valued TNR scores is in percentage points, not relative-to-Ours decline."
    ),
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _load_json(payload):
    value = json.loads(payload, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _list(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    return value


def _groups(plan):
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "thresholds_frozen_from_train":
        raise ValueError("plan must be a frozen train-derived clipping grid")
    groups, seen_thresholds = {}, set()
    off_count = 0
    for group in _list(plan.get("groups"), "plan.groups"):
        if not isinstance(group, dict):
            raise ValueError("plan group must be an object")
        group_id = _text(group.get("id"), "group.id")
        if group_id in groups or "max_norm" not in group:
            raise ValueError("plan requires unique canonical group ids and explicit max_norm")
        cap = group["max_norm"]
        if cap is None:
            off_count += 1
            if group_id != "off":
                raise ValueError("the unbounded reference must have canonical id off")
        else:
            cap = _number(cap, f"{group_id}.max_norm")
            if cap < 0 or group_id == "off" or cap in seen_thresholds:
                raise ValueError("finite canonical thresholds must be nonnegative and unique")
            seen_thresholds.add(cap)
        groups[group_id] = cap
    if off_count != 1 or not seen_thresholds:
        raise ValueError("plan needs exactly one off reference and at least one finite threshold")
    if "unique_run_count" in plan and plan["unique_run_count"] != len(groups):
        raise ValueError("plan unique_run_count mismatch")
    rule = plan.get("threshold_rule", {})
    if not isinstance(rule, dict):
        raise ValueError("plan threshold_rule must be an object")
    if rule.get("validation_or_test_used", False) is not False or rule.get("post_evaluation_refinement_allowed", False) is not False:
        raise ValueError("validation/test selection or post-evaluation refinement is forbidden")
    return groups


def _baselines(document):
    if document.get("schema") != BASELINES_SCHEMA:
        raise ValueError("wrong baseline schema")
    comparisons = {}
    for item in _list(document.get("comparisons"), "baselines.comparisons"):
        if not isinstance(item, dict):
            raise ValueError("baseline comparison must be an object")
        comparison_id = _text(item.get("id"), "comparison.id")
        if comparison_id in comparisons:
            raise ValueError("duplicate baseline comparison id")
        for field in ("task", "model", "metric", "baseline_method", "unit"):
            _text(item.get(field), f"baseline.{field}")
        if item.get("direction") not in DIRECTIONS:
            raise ValueError("unsupported metric direction: do not infer attack direction from a signed output metric")
        normalized = dict(item)
        normalized["value"] = _number(item.get("value"), "baseline.value")
        comparisons[comparison_id] = normalized
    return comparisons


def _sign(value):
    return "higher" if value > 0 else "lower" if value < 0 else "equal"


def _range_status(points, strict_brackets):
    signs = {point["relation"] for point in points}
    if len(signs) == 1:
        return f"all_finite_points_{next(iter(signs))}"
    if strict_brackets:
        return "observed_multiple_sign_change_brackets" if len(strict_brackets) > 1 else "observed_sign_change_bracket"
    if "higher" in signs and "lower" in signs:
        return "higher_and_lower_points_separated_by_exact_grid_ties"
    if "higher" in signs:
        return "higher_or_equal_at_sampled_finite_points_no_strict_crossing_observed"
    return "lower_or_equal_at_sampled_finite_points_no_strict_crossing_observed"


def summarize(plan, points, baselines, *, grid_sha256):
    """Validate the complete canonical Cartesian product and describe every crossing.

    The CLI supplies the SHA of the plan's exact file bytes. ``audit_verified``
    is an upstream audit attestation; this numeric-only program does not rerun
    tensor/artifact audits and does not upgrade that attestation into a new one.
    """
    if not isinstance(grid_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", grid_sha256) is None:
        raise ValueError("grid_sha256 must be a lowercase SHA256")
    groups = _groups(plan)
    references = _baselines(baselines)
    if points.get("schema") != POINTS_SCHEMA or points.get("grid_sha256") != grid_sha256:
        raise ValueError("evaluated points must bind the exact frozen plan SHA")
    rows = {}
    for row in _list(points.get("rows"), "points.rows"):
        if not isinstance(row, dict):
            raise ValueError("point must be an object")
        group_id = _text(row.get("group_id"), "point.group_id")
        comparison_id = _text(row.get("comparison_id"), "point.comparison_id")
        key = (group_id, comparison_id)
        if group_id not in groups or comparison_id not in references:
            raise ValueError("unknown or noncanonical group/comparison id")
        if key in rows:
            raise ValueError("duplicate group/comparison point")
        if row.get("audit_verified") is not True:
            raise ValueError("every point requires an explicit true audit_verified attestation")
        expected_cap = groups[group_id]
        if "max_norm" not in row:
            raise ValueError("point max_norm missing")
        actual_cap = row["max_norm"]
        if expected_cap is None:
            if actual_cap is not None:
                raise ValueError("off point max_norm must be null")
        elif actual_cap is None or _number(actual_cap, "point.max_norm") != expected_cap:
            raise ValueError("point threshold differs from frozen canonical group")
        baseline = references[comparison_id]
        for field in ("task", "model", "metric", "unit"):
            if row.get(field) != baseline[field]:
                raise ValueError(f"point {field} differs from frozen baseline")
        baseline_value = _number(row.get("baseline_value"), "point.baseline_value")
        if baseline_value != baseline["value"]:
            raise ValueError("point baseline value differs from frozen baseline")
        ours_value = _number(row.get("ours_value"), "point.ours_value")
        declared_delta = _number(row.get("delta"), "point.delta")
        calculated_delta = ours_value - baseline_value
        if not math.isfinite(calculated_delta) or declared_delta != calculated_delta:
            raise ValueError("delta must equal finite ours_value minus frozen baseline_value exactly")
        rows[key] = {
            "group_id": group_id, "max_norm": expected_cap,
            "ours_value": ours_value, "baseline_value": baseline_value,
            "delta": calculated_delta, "relation": _sign(calculated_delta),
            "audit_verified": True,
        }
    expected_keys = {(group_id, comparison_id) for group_id in groups for comparison_id in references}
    if set(rows) != expected_keys:
        missing = sorted(expected_keys - set(rows))
        raise ValueError(f"incomplete canonical groups × baseline comparisons; missing={missing}")

    results = []
    finite_order = sorted(((group_id, cap) for group_id, cap in groups.items() if cap is not None), key=lambda pair: pair[1])
    for comparison_id, baseline in references.items():
        finite = [rows[(group_id, comparison_id)] for group_id, _ in finite_order]
        ties = [dict(point) for point in finite if point["relation"] == "equal"]
        changes, brackets = [], []
        for lower, upper in zip(finite, finite[1:]):
            if lower["relation"] == upper["relation"]:
                continue
            strict = lower["relation"] != "equal" and upper["relation"] != "equal"
            item = {
                "kind": "opposite_sign_bracket" if strict else "adjacent_to_exact_grid_tie",
                "lower": dict(lower), "upper": dict(upper),
                "transition_as_threshold_increases": f"{lower['relation']}_to_{upper['relation']}",
                "interpolated_critical_value": None,
            }
            changes.append(item)
            if strict:
                brackets.append(item)
        intersections = [{"kind": "exact_grid_tie", "point": point} for point in ties] + [dict(item) for item in brackets]
        intersections.sort(key=lambda item: item["point"]["max_norm"] if item["kind"] == "exact_grid_tie" else item["lower"]["max_norm"])
        results.append({
            "comparison_id": comparison_id,
            "task": baseline["task"], "model": baseline["model"], "metric": baseline["metric"],
            "baseline_method": baseline["baseline_method"], "baseline_value": baseline["value"],
            "unit": baseline["unit"],
            "delta_definition": "ours_value - frozen_baseline_value",
            "delta_unit_note": "Difference is in percentage points when values are percentages; otherwise in the stated metric unit.",
            "direction": baseline["direction"], "direction_interpretation": DIRECTIONS[baseline["direction"]],
            "finite_points": finite, "off_reference": rows[("off", comparison_id)],
            "adjacent_sign_changes": changes, "crossing_brackets": brackets,
            "exact_ties": ties, "all_intersections": intersections,
            "range_status": _range_status(finite, brackets),
        })
    return {
        "schema": OUTPUT_SCHEMA, "status": "complete_grid_descriptive_comparison",
        "grid_sha256": grid_sha256, "canonical_group_count": len(groups),
        "comparison_count": len(references), "verified_point_count": len(rows),
        "finite_threshold_range": [finite_order[0][1], finite_order[-1][1]],
        "comparisons": results,
        "claim_scope": "single_seed_exploratory_fixed_grid; relations refer only to recorded numerical point estimates",
        "audit_scope": "Checks supplied audit_verified attestations and numeric contracts; does not rerun raw tensor or artifact audits.",
        "policy": {
            "off_connected_to_finite_grid": False, "monotonicity_assumed": False,
            "critical_value_interpolation": False, "threshold_selection": False,
            "post_evaluation_grid_refinement": False, "statistical_significance_claim": False,
            "ground_truth_error_claim": False,
        },
        "interpretation_notes": [
            "All adjacent opposite-sign finite brackets and exact sampled ties are reported; multiple crossings are retained.",
            "A bracket reports opposite signs at its sampled endpoints only; it does not prove continuity or locate a unique critical threshold.",
            "Exact equality uses stored full-precision numbers without a significance tolerance; it is not a statistical equivalence test.",
            "Off is reported independently and is not joined to the largest finite threshold.",
            "No crossing in the sampled range does not establish a bound outside that range or between unobserved points.",
            "Results must not be used to refine the grid on development validation, select hyperparameters, or tune on test.",
        ],
    }


def compare_files(plan_path, points_path, baselines_path):
    paths = {"plan": Path(plan_path), "points": Path(points_path), "baselines": Path(baselines_path)}
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    documents = {key: _load_json(payload) for key, payload in payloads.items()}
    result = summarize(documents["plan"], documents["points"], documents["baselines"], grid_sha256=_sha(payloads["plan"]))
    optional_baselines_sha = documents["points"].get("baselines_sha256")
    if optional_baselines_sha is not None and optional_baselines_sha != _sha(payloads["baselines"]):
        raise ValueError("evaluated points baseline SHA differs from supplied baseline file")
    result["input_bindings"] = {
        key: {"path": str(paths[key].absolute()), "sha256": _sha(payload), "bytes": len(payload)}
        for key, payload in payloads.items()
    }
    return result


def write_result(result, output):
    output = Path(output).absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError("output symlinks forbidden")
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_files(args.plan, args.points, args.baselines)
    digest = write_result(result, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": digest,
                      "comparison_count": result["comparison_count"], "point_count": result["verified_point_count"]}))


if __name__ == "__main__":
    main()
