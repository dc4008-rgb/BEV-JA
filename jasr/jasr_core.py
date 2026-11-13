"""New CPU JASR supplement; no model calls. Diagnostic scope is exactly two inputs."""
from __future__ import annotations

import math
import numpy as np

STATUS = "DIAGNOSTIC_NOT_FORMAL"
NUMERIC_POLICY = "float64_from_native_outputs_clean_native_dtype_epsilon_v1"
SEMANTICS = {
    "monodepth2": ("mono_scale_ambiguous_depth", 1),
    "dpt_large": ("relative_inverse_depth", -1),
    "depth_anything_v1": ("relative_inverse_depth", -1),
    "zoedepth": ("metric_depth", 1),
}
DTYPE_EPS = {
    "float16": 2.0 ** -10, "float32": 2.0 ** -23,
    "float64": 2.0 ** -52, "bfloat16": 2.0 ** -7,
}
THRESHOLD_KEYS = {"confidence_threshold", "iou_threshold", "beta_eval", "rho_min", "target_labels"}


class ContractError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ContractError(message)


def finite_number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be an explicit finite number")
    return float(value)


def validate_config(c):
    require(isinstance(c, dict) and c.get("schema") == "jasr-two-case-diagnostic-config-v1", "wrong configuration schema")
    require(c.get("status") == STATUS and c.get("formal_200_enabled") is False,
            "this tool refuses formal/200-sample execution")
    require(c.get("data_kind") in ("SYNTHETIC_TEST_ONLY", "SERVER_SAVED_OUTPUTS_DIAGNOSTIC"), "explicit data kind required")
    require(c.get("numeric_policy") == NUMERIC_POLICY, "numeric policy must be explicitly frozen")
    ids = c.get("sample_ids")
    require(isinstance(ids, list) and len(ids) == 2 and all(isinstance(s, str) and s for s in ids)
            and len(set(ids)) == 2, "exactly two distinct explicitly chosen sample IDs required")
    t = c.get("thresholds")
    require(isinstance(t, dict) and set(t) == THRESHOLD_KEYS, "all five evaluation threshold/class fields required")
    theta = finite_number(t["confidence_threshold"], "confidence_threshold")
    chi = finite_number(t["iou_threshold"], "iou_threshold")
    beta = finite_number(t["beta_eval"], "beta_eval")
    rho = finite_number(t["rho_min"], "rho_min")
    require(0 <= theta <= 1 and 0 < chi <= 1 and beta > 0 and 0 <= rho <= 1, "invalid evaluation threshold range")
    labels = t["target_labels"]
    require(isinstance(labels, list) and labels and all(type(x) is int and x >= 0 for x in labels)
            and len(labels) == len(set(labels)), "explicit unique target label IDs required; no default from training")
    require(c.get("clean_match_rule") == "score_ge_theta_and_iou_ge_chi_and_allowed_label",
            "clean and final use the manuscript >= boundary; training monitor > is not substituted")
    require(c.get("roi_rule") == "fixed_saved_object_mask_no_resize" and
            c.get("q_rule") == "clean_abs_mean_over_entire_native_output_domain",
            "fixed target ROI/full-output clean scale required")
    require(c.get("depth_validity_policy") == "saved_mask_finite_nonnegative_clean_zero_included",
            "explicit new JASR validity policy required; old clean-positive metric filtering is not silently reused")
    pairs = c.get("model_pairs")
    require(isinstance(pairs, list) and pairs, "explicit detector-depth pairs required")
    seen = set()
    for pair in pairs:
        require(isinstance(pair, dict) and set(pair) == {"detector", "depth"}, "closed model pair required")
        d, z = pair["detector"], pair["depth"]
        require(isinstance(d, str) and d and z in SEMANTICS and (d, z) not in seen, "unknown/duplicate pair")
        seen.add((d, z))
    for key in ("protocol_binding", "thresholds_binding", "server_export_proof"):
        binding = c.get(key)
        if c["data_kind"] == "SYNTHETIC_TEST_ONLY":
            require(binding == {"kind": "SYNTHETIC_TEST_ONLY"}, "synthetic inputs cannot claim actual evidence")
        else:
            require(isinstance(binding, dict) and set(binding) == {"path", "bytes", "sha256"}, key + " actual descriptor required")
    return c


def exact_binary_mask(value):
    """Exact conversion of the saved 0/1 float mask, never thresholding or resizing."""
    a = np.asarray(value)
    require(a.dtype.kind in "bf" and np.isfinite(a).all() and ((a == 0) | (a == 1)).all(),
            "saved mask must be exactly binary boolean/float; soft values cannot be repaired")
    return a.astype(np.bool_)


def matrix(value, name, *, mask=False):
    a = np.asarray(value)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    require(a.ndim == 2 and a.size > 0, name + " must be one native 2D canvas (singleton leading dimensions allowed)")
    require(a.size <= 16_777_216, "bounded diagnostic canvas required")
    if mask:
        require(a.dtype == np.bool_, name + " must be boolean, not a soft/interpolated mask")
        return a
    require(a.dtype.kind in "fiu" and np.isfinite(a).all() and (a >= 0).all(), name + " must be finite nonnegative native evaluation_output")
    return a.astype(np.float64)


def predictions(p, name):
    require(isinstance(p, dict) and set(p) == {"boxes", "scores", "labels"}, name + " requires boxes/scores/labels")
    b, s, lab = np.asarray(p["boxes"], dtype=np.float64), np.asarray(p["scores"], dtype=np.float64), np.asarray(p["labels"])
    if b.size == 0:
        b = b.reshape((0, 4))
    require(b.ndim == 2 and b.shape[1] == 4 and s.shape == (len(b),) and lab.shape == (len(b),), name + " shape mismatch")
    require(lab.dtype.kind in "iu" or (len(lab) == 0 and lab.dtype.kind == "f"), name + " labels must be integers")
    require((lab >= 0).all(), name + " labels must be nonnegative")
    require(np.isfinite(b).all() and np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all(), name + " nonfinite boxes or invalid confidence")
    require(((b[:, 2:] - b[:, :2]) >= 0).all(), name + " invalid xyxy box")
    return b, s, lab


def matched(p, target_box, thresholds):
    b, scores, labels = predictions(p, "saved detection")
    target = np.asarray(target_box, dtype=np.float64)
    require(target.shape == (4,) and np.isfinite(target).all() and (target[2:] > target[:2]).all(), "fixed positive-area target_box required")
    inter_sides = np.maximum(0.0, np.minimum(b[:, 2:], target[2:]) - np.maximum(b[:, :2], target[:2]))
    inter = inter_sides.prod(axis=1)
    area = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area + np.prod(target[2:] - target[:2]) - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    ok = np.isin(labels, thresholds["target_labels"]) & (scores >= thresholds["confidence_threshold"]) & (iou >= thresholds["iou_threshold"])
    return {"matched": bool(ok.any()), "matching_count": int(ok.sum()),
            "max_matching_score": float(scores[ok].max()) if ok.any() else None}


def depth_result(name, d, thresholds):
    require(isinstance(d, dict), "saved depth object required")
    require(d.get("semantics") == SEMANTICS[name][0], "native evaluation semantics mismatch for " + name)
    dtype = d.get("native_dtype")
    require(dtype in DTYPE_EPS, "explicit supported native floating dtype required")
    clean, final = matrix(d["clean"], "clean depth"), matrix(d["final"], "final depth")
    roi = matrix(d["object_mask"], "fixed object_mask", mask=True)
    require(clean.shape == final.shape == roi.shape, "clean/final/fixed target ROI shapes differ; resizing is forbidden")
    # q uses ALL native output pixels, not the object ROI or the final prediction.
    q = max(float(np.mean(np.abs(clean), dtype=np.float64)), DTYPE_EPS[dtype])
    require(math.isfinite(q) and q > 0, "finite clean-derived scale required")
    kappa = SEMANTICS[name][1]
    delta = kappa * (np.arcsinh(final / q) - np.arcsinh(clean / q))
    require(np.isfinite(delta).all(), "far-directed shift is nonfinite")
    count = int(roi.sum())
    if count == 0:
        return {"clean_depth_valid": False, "invalid_reason": "empty_fixed_object_mask", "roi_pixels": 0,
                "kappa": kappa, "q_clean": q, "native_dtype": dtype, "FarShift": None, "rho": None, "depth_success": None}
    values = delta[roi]
    far, rho = float(values.mean(dtype=np.float64)), float((values > 0).mean(dtype=np.float64))
    return {"clean_depth_valid": True, "invalid_reason": None, "roi_pixels": count,
            "kappa": kappa, "q_clean": q, "native_dtype": dtype, "FarShift": far, "rho": rho,
            "depth_success": bool(far >= thresholds["beta_eval"] and rho >= thresholds["rho_min"])}


def calculate(config, data):
    c = validate_config(config)
    require(data.get("schema") == "jasr-neutral-two-case-input-v1" and data.get("status") == STATUS
            and data.get("data_kind") == c["data_kind"], "input kind/status/schema mismatch")
    cases = data.get("cases")
    require(isinstance(cases, list) and [x.get("sample_id") for x in cases] == c["sample_ids"], "exact ordered two-case coverage required")
    rows, summaries = [], []
    for pair in c["model_pairs"]:
        pair_rows = []
        for case in cases:
            require(isinstance(case.get("binding"), dict), "same-case input binding required")
            require(isinstance(case.get("source_payloads"), list), "source payload bindings required")
            if c["data_kind"] != "SYNTHETIC_TEST_ONLY":
                require(3 <= len(case["source_payloads"]) <= 9 and case["binding"].get("sample_id") == case["sample_id"], "actual det/depth/mask and same-car identity bindings required")
                for desc in case["source_payloads"]:
                    require(isinstance(desc, dict) and set(desc) == {"path", "bytes", "sha256"}, "actual source descriptor required")
            detector, dep = pair["detector"], pair["depth"]
            require(detector in case["detectors"] and dep in case["depths"], "missing explicitly requested model output")
            saved = case["detectors"][detector]
            clean = matched(saved["clean"], case["target_box"], c["thresholds"])
            final = matched(saved["final"], case["target_box"], c["thresholds"])
            dr = depth_result(dep, case["depths"][dep], c["thresholds"])
            eligible = clean["matched"] and dr["clean_depth_valid"]
            det_success = not final["matched"]
            row = {"sample_id": case["sample_id"], "detector": detector, "depth_model": dep,
                   "clean_detected": clean["matched"], "clean_match": clean, "final_match": final,
                   "eligible_clean_pair": bool(eligible), "detection_success": det_success,
                   **dr, "joint_success": bool(det_success and dr["depth_success"]) if eligible else None,
                   "denominator_exclusion_reason": None if eligible else "clean_target_not_detected" if not clean["matched"] else dr["invalid_reason"]}
            rows.append(row)
            pair_rows.append(row)
        valid = [r for r in pair_rows if r["eligible_clean_pair"]]
        n = len(valid)
        counts = {"detection_success_count": sum(r["detection_success"] for r in valid),
                  "depth_success_count": sum(r["depth_success"] for r in valid),
                  "joint_success_count": sum(r["joint_success"] for r in valid)}
        summaries.append({"detector": pair["detector"], "depth_model": pair["depth"],
                          "eligible_sample_ids": [r["sample_id"] for r in valid], "eligible_count": n,
                          "excluded_count": len(pair_rows) - n, **counts,
                          "DetASR": counts["detection_success_count"] / n if n else None,
                          "DepASR": counts["depth_success_count"] / n if n else None,
                          "JASR": counts["joint_success_count"] / n if n else None,
                          "denominator_status": "VALID_CLEAN_PAIR_SET" if n else "NO_CLEAN_VALID_SAMPLES"})
    return {"schema": "jasr-two-case-diagnostic-result-v1", "status": STATUS, "data_kind": c["data_kind"],
            "formal_200_result": False, "numeric_policy": NUMERIC_POLICY, "thresholds": c["thresholds"],
            "depth_validity_policy": c["depth_validity_policy"],
            "rates_are_fractions_not_percent": True, "pair_rates_not_averaged_or_multiplied": True,
            "same_vehicle_joint_indicator": True, "sample_ids": c["sample_ids"], "per_sample_pair": rows, "per_model_pair": summaries}
