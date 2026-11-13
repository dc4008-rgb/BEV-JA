#!/usr/bin/env python3
"""Extract fixed descriptive reference values; no evaluation or model execution.

Only the five pinned local evidence files below are read. Historical evidence is
never modified. The output is a new file, and all numeric values retain the
source's precision (apart from the declared multiplication by 100 for percent).
These references must not be used to choose or refine clipping thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "r2-joint-clip-baselines-v1"
SOURCES = {
    "thys_nap": (
        "11_远程运维脚本/source_entry_long_20260929_readback_live/VERIFIED_RESULT_POINTER.json",
        "ced24c0c236287f8196a1d96dd37e118e454299ccfb9158348de4e0fc9611599"),
    "dpatch": (
        "11_远程运维脚本/dpatch_source_entry_20260929_live/FINAL_STATUS_20260930.json",
        "5277dba56fce463efd8fd7c932760f5416782dca6243bce49c7c051351a9be51"),
    "cap": (
        "11_远程运维脚本/cap_source_entry_20260930_live/FINAL_STATUS_V2.json",
        "235a97ac9a2237e9b197cd3948609e062ef9f72f27b47adf7964d2bc0fab4d6f"),
    "tsea": (
        "11_远程运维脚本/tsea_source_entry_20260930_live/EVALUATION_SUMMARY_RESUME_V1.json",
        "f86f50c539bc0e6e5fc266a9e790cd2984a3f0d7d2836bdade0cb2c3bb5d4a74"),
    "saam": (
        "11_远程运维脚本/joint_vs_saam_20261002/COMPARISON.json",
        "88db584c6761f86576bd820c7d48756f9304741207ce0527ebe57484ea6db39e"),
}
DETECTORS = ("fasterrcnn", "ssd300", "maskrcnn", "fcos")
DEPTH_MODELS = ("monodepth2", "dpt_large", "depth_anything_v1", "zoedepth")
DEPTH_METRICS = ("abs_rel_clean_ref", "rmse_clean_ref", "rmse_log_clean_ref")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value),
            "reference value must be finite")
    return value


def build_references(project_root):
    sources, data, comparisons, diagnostics = [], {}, [], []
    for key, (relative, expected_sha) in SOURCES.items():
        raw = (Path(project_root) / relative).read_bytes()
        require(sha(raw) == expected_sha, "source SHA mismatch: " + relative)
        data[key] = json.loads(raw)
        sources.append({"id": key, "path": relative, "sha256": expected_sha,
                        "bytes": len(raw), "access": "read_only"})

    def detector(method, model, raw_value, source_id, source_field, remote_sha, clean_ap, final_ap):
        require(model in DETECTORS, "unexpected detector")
        raw_value, clean_ap, final_ap = number(raw_value), number(clean_ap), number(final_ap)
        require(clean_ap > 0, "undefined TNR: clean AP50 is zero")
        recomputed = (clean_ap - final_ap) / clean_ap
        require(math.isclose(raw_value, recomputed, rel_tol=1e-12, abs_tol=1e-12), "TNR differs from AP50 pair")
        comparisons.append({"id": "detection_" + method + "_" + model + "_tnr_det",
            "task": "detection", "model": model, "metric": "tnr_det",
            "baseline_method": method, "value": raw_value * 100.0,
            "unit": "percent", "direction": "larger_is_more_AP50_drop",
            "dataset": "combined", "scene_count": 40,
            "reference_state": "clean", "evaluated_state": "final",
            "source_id": source_id, "source_field": source_field,
            "source_value": raw_value, "source_to_value_multiplier": 100.0,
            "remote_evaluation_summary_sha256": remote_sha,
            "clean_AP50": clean_ap, "final_AP50": final_ap})

    shared = data["thys_nap"]
    require(shared["retention_reverified"] is True and shared["result"]["scientific_complete"] is True,
            "Thys/NAP completion evidence missing")
    require(shared["report"]["status"] == "pass" and shared["report"]["test_data_used"] is False,
            "Thys/NAP report status invalid")
    for method in ("thys", "nap"):
        method_id = method + "-author-entry-bev-long-20260928-v1"
        result = shared["report"]["methods"][method_id]
        require(set(result["models"]) == set(DETECTORS), "Thys/NAP detector set changed")
        require(shared["result"]["methods"][method]["updates"] == 25000, "wrong Thys/NAP budget")
        for model in DETECTORS:
            row = result["models"][model]
            detector(method, model, row["clean_to_final_tnr"], "thys_nap",
                "/report/methods/" + method_id + "/models/" + model + "/clean_to_final_tnr",
                result["evaluation_summary_sha256"], row["clean_AP50"], row["final_AP50"])

    dpatch = data["dpatch"]
    require(dpatch["status"] == "complete_and_raw_audited" and dpatch["training"]["updates"] == 200000,
            "wrong DPatch completed run")
    require(dpatch["evaluation"]["scenes"] == 40 and dpatch["evaluation"]["test_data_used"] is False,
            "wrong DPatch evaluation scope")
    for index, row in enumerate(dpatch["evaluation"]["combined"]):
        detector("dpatch", row["model"], row["clean_to_final_tnr"], "dpatch",
            "/evaluation/combined/" + str(index) + "/clean_to_final_tnr",
            dpatch["evaluation"]["summary_sha256"], row["clean_ap50"], row["final_ap50"])

    cap = data["cap"]
    require(cap["status"] == "completed_and_audited" and cap["training"]["updates"] == 700,
            "wrong CAP completed run")
    require(cap["training"]["counts"] == {"isp": 360, "patch": 340}, "CAP alternating budget changed")
    require(cap["audit"]["status"] == "pass" and cap["training"]["test_data_used"] is False,
            "CAP audit or split invalid")
    for index, row in enumerate(cap["audit"]["combined"]):
        detector("cap", row["model"], row["clean_to_final_tnr"], "cap",
            "/audit/combined/" + str(index) + "/clean_to_final_tnr",
            cap["hashes"]["evaluation_summary"], row["clean_AP50"], row["final_AP50"])

    tsea = data["tsea"]
    require(tsea["status"] == "external_development_validation_complete" and tsea["updates"] == 25000,
            "wrong T-SEA completed evaluation")
    require(tsea["test_data_opened"] is False, "T-SEA test data were opened")
    for index, row in enumerate(tsea["summaries"]):
        if row["dataset"] == "combined" and row["state"] == "final":
            require(row["scene_count"] == 40 and row["tnr_defined"] is True, "wrong T-SEA evaluation scope")
            detector("tsea", row["model"], row["tnr_det"], "tsea", "/summaries/" + str(index) + "/tnr_det",
                SOURCES["tsea"][1], row["target_vehicle_ap50_clean"], row["target_vehicle_ap50_adversarial"])

    saam = data["saam"]
    require(saam["status"] == "verified_common_evaluation_and_controls", "SAAM comparison audit missing")
    for index, row in enumerate(saam["rows"]):
        if (row["method"] not in ("saam_monodepth2", "saam_dpt_large")
                or row["dataset"] != "combined" or row["reference_state"] != "clean"):
            continue
        require(row["evaluated_state"] == "final" and row["scene_count"] == 40 and row["updates"] == 5000,
                "wrong SAAM state or budget")
        method, model = row["method"], row["evaluation_model"]
        require(model in DEPTH_MODELS, "unexpected SAAM depth model")
        for metric in DEPTH_METRICS:
            factor = 100.0 if metric == "abs_rel_clean_ref" else 1.0
            raw_value = number(row["primary_macro"][metric]["mean"])
            comparisons.append({"id": "depth_" + method + "_" + model + "_" + metric,
                "task": "depth", "model": model, "metric": metric, "baseline_method": method,
                "value": raw_value * factor, "unit": "percent" if factor == 100.0 else "native",
                "direction": "larger_is_more_prediction_deviation", "dataset": "combined",
                "scene_count": 40, "reference_state": "clean", "evaluated_state": "final",
                "aggregation": "per_scene_macro_mean", "depth_representation": row["depth_representation"],
                "baseline_model_role": row["role"], "source_id": "saam",
                "source_field": "/rows/" + str(index) + "/primary_macro/" + metric + "/mean",
                "source_value": raw_value, "source_to_value_multiplier": factor,
                "remote_evaluation_summary_sha256": row["remote_evaluation_summary_sha256"]})
        diagnostics.append({"id": "depth_" + method + "_" + model + "_delta_sigma",
            "task": "depth", "model": model, "metric": "delta_sigma", "baseline_method": method,
            "value": number(row["primary_macro"]["delta_sigma"]["mean"]), "unit": "native_log_ratio",
            "direction": "signed_native_prediction_shift_not_larger_is_better",
            "positive_shift_interpretation": "nearer" if row["depth_representation"] == "relative_inverse_depth" else "farther",
            "included_in_crossing_comparisons": False, "dataset": "combined", "scene_count": 40,
            "reference_state": "clean", "evaluated_state": "final", "source_id": "saam",
            "source_field": "/rows/" + str(index) + "/primary_macro/delta_sigma/mean",
            "remote_evaluation_summary_sha256": row["remote_evaluation_summary_sha256"]})

    expected = {("detection", method, model, "tnr_det")
        for method in ("thys", "nap", "dpatch", "cap", "tsea") for model in DETECTORS}
    expected |= {("depth", method, model, metric) for method in ("saam_monodepth2", "saam_dpt_large")
        for model in DEPTH_MODELS for metric in DEPTH_METRICS}
    actual = {(r["task"], r["baseline_method"], r["model"], r["metric"]) for r in comparisons}
    require(len(comparisons) == 44 and actual == expected and len({r["id"] for r in comparisons}) == 44,
            "baseline grid must contain exactly 20 detection and 24 depth references")
    require(len(diagnostics) == 8, "expected 8 separate native-log-shift diagnostics")

    return {"schema": SCHEMA, "status": "historical_reference_values_frozen",
        "sources": sources, "comparison_count": 44, "comparisons": comparisons,
        "diagnostic_count": 8, "diagnostics": diagnostics,
        "evaluation_scope": {"split": "development-validation", "dataset": "combined",
            "scene_count": 40, "domain_counts": {"kitti": 20, "nuscenes": 20},
            "reference_state": "clean", "evaluated_state": "fixed_final",
            "test_data_used": False, "new_inference": False, "new_metric_recomputation": False,
            "threshold_selection_use": False},
        "methods": {
            "thys": {"method_id": "thys-author-entry-bev-long-20260928-v1", "epochs": 1000,
                "updates": 25000, "training_source_models": ["yolov2"], "all_four_evaluation_models": "transfer",
                "qualification": "author-entry BEV road experiment; not the default 10000-epoch author schedule"},
            "nap": {"method_id": "nap-author-entry-bev-long-20260928-v1", "epochs": 1000,
                "updates": 25000, "training_source_models": ["fasterrcnn"],
                "qualification": "author-entry BigGAN latent road experiment; Faster R-CNN shares source weights but pre/postprocessing differs"},
            "dpatch": {"method_id": dpatch["method_id"], "updates": 200000,
                "training_source_models": ["fasterrcnn_resnet50_fpn_coco"],
                "qualification": "paper-method road adaptation; original VOC/data-detector configuration replaced"},
            "cap": {"epochs": 100, "updates": 700, "update_counts": {"isp": 360, "patch": 340},
                "training_source_models": ["yolov5s_inria_person"], "all_four_evaluation_models": "cross_scene_transfer",
                "qualification": "author-entry ISP alternating optimization retained; person source evaluated on road vehicles"},
            "tsea": {"method_id": tsea["method_id"], "epochs": 1000, "updates": 25000,
                "training_source_models": tsea["training_source_models"], "all_four_evaluation_models": "transfer",
                "qualification": "author-entry road experiment; resumed after epoch 970, not claimed bitwise-exact GPU resume"},
            "saam_monodepth2": {"epochs": 200, "updates": 5000, "training_source_models": ["monodepth2"],
                "qualification": "SAAM paper-formula road reimplementation, not verified author source code"},
            "saam_dpt_large": {"epochs": 200, "updates": 5000, "training_source_models": ["dpt_large"],
                "qualification": "SAAM paper-formula road reimplementation, not verified author source code"}},
        "interpretation_limits": [
            "These are frozen descriptive baseline values, not a threshold-selection objective.",
            "Training budgets, source models, objectives, initialization, parameterization and constraints differ between methods.",
            "The future joint clipping sweep has 500 updates and two detector plus two depth training sources.",
            "Detection TNR is relative target-vehicle AP50 drop from clean, multiplied by 100; negative values mean AP50 increased.",
            "AbsRel/RMSE/logRMSE are deviations from clean predictions, not sensor-ground-truth accuracy deterioration or targeted-far success rates.",
            "Only compare the same model and metric; do not average native RMSE across depth models.",
            "Delta_sigma is signed and direction depends on native depth versus inverse-depth; it is excluded from larger-is-better crossings.",
            "The two SAAM runs are two source-model settings of one paper, not two separate papers.",
            "Historical Ours/SAAM clean/random/ROI cross-run controls were audited; no new joint raw-control audit against every detection baseline is asserted here.",
            "Crossings are exploratory observed comparisons, not statistical significance, causal equal-budget gains, or a guaranteed monotonic critical threshold.",
            "Superseded wrong-direction adapters and early formal/pre-normalization results are excluded."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    for path in (output, *output.parents):
        require(not path.is_symlink(), "output symlinks forbidden")
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen baseline references")
    result = build_references(args.project_root)
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": SCHEMA, "comparisons": 44, "diagnostics": 8,
                      "output": str(output), "sha256": sha(encoded)}))


if __name__ == "__main__":
    main()
