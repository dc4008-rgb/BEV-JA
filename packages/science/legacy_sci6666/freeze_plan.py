#!/usr/bin/env python3
"""Freeze independent diagnostics, then a resource-admitted formal shard."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
from sample_contract import (BASE_CONFIG, DIAGNOSTIC_IDS, IDENTITY, ROOT, SCHEMA,
    sha256, validate_plan, document_sha256, partition_samples, MASTER_SAMPLES_SHA256,
    validate_diagnostic_receipt, validate_admission_receipt, scientific_config_for, VARIANTS)
from sample_contract import planned_budget
from source_bindings import DATA_MANIFEST_SHA256, SOURCE_MANIFESTS, PAPER_SHA256, REFERENCE_MANIFEST_SHA256, SPECIFICATION_SHA256

def _samples(rows):
    return [{"sample_id": row["sample_id"], "case_id": row["sample_id"],
        "dataset": row["dataset"], "source_split": row["split"],
        "source_path": row["source_path"], "source_sha256": row["source_sha256"],
        "road_quad_norm": row["road_quad_norm"],
        "width": int(row["width"]), "height": int(row["height"])} for row in rows]


def receipt_binding(path, expected_sha):
    path = Path(path)
    if path.is_symlink() or sha256(path) != expected_sha:
        raise ValueError("acceptance/admission receipt file SHA differs")
    document = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path.resolve()), "sha256": expected_sha,
            "document_sha256": document_sha256(document), "document": document}


def build_plan(manifest, mode, *, variant="complete", shard_id=None, diagnostic_plan=None,
               diagnostic_receipt=None, diagnostic_receipt_sha256=None,
               admission_receipt=None, admission_receipt_sha256=None):
    if sha256(manifest) != DATA_MANIFEST_SHA256:
        raise ValueError("unreviewed data manifest")
    with Path(manifest).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    master = _samples(sorted((r for r in rows if r["split"] == "validation"),
        key=lambda r: (r["dataset"], r["sample_id"])))
    if document_sha256(master) != MASTER_SAMPLES_SHA256:
        raise ValueError("original40 development members changed")
    shards = partition_samples(master)
    if mode == "diagnostic":
        selected = [next(row for row in rows if row["sample_id"] == sample_id) for sample_id in DIAGNOSTIC_IDS]
        updates, status = 25, "frozen_diagnostic_plan"
    elif mode == "development-draft":
        updates, status = 500, "draft_not_executable"
        selected = None
    elif mode == "formal":
        if shard_id not in (0, 1):
            raise ValueError("explicit shard-id0 or1 required")
        if not all((diagnostic_plan, diagnostic_receipt, diagnostic_receipt_sha256,
                    admission_receipt, admission_receipt_sha256)):
            raise ValueError("formal freeze requires diagnostic plan and both accepted SHA-bound receipts")
        from sample_contract import load_plan
        diagnostic = load_plan(diagnostic_plan)
        if diagnostic["mode"] != "diagnostic":
            raise ValueError("two-train25 diagnostic plan required")
        if (diagnostic["ablation_variant"] != variant
            or diagnostic["scientific_config"] != scientific_config_for(variant, 25)):
            raise ValueError("formal variant must have its own exact completed diagnostic configuration")
        diag_binding = receipt_binding(diagnostic_receipt, diagnostic_receipt_sha256)
        admission_binding = receipt_binding(admission_receipt, admission_receipt_sha256)
        validate_diagnostic_receipt(diag_binding["document"], diagnostic_plan_sha256=sha256(diagnostic_plan))
        if (diag_binding["document"]["ablation_variant"] != diagnostic["ablation_variant"]
            or diag_binding["document"]["switches"] != diagnostic["scientific_config"]["switches"]):
            raise ValueError("diagnostic acceptance switches differ from their actual frozen plan")
        validate_admission_receipt(admission_binding["document"], shard_id=shard_id,
            diagnostic_receipt_sha256=diagnostic_receipt_sha256)
        updates, status, selected = 500, "frozen_formal_shard", None
    else:
        raise ValueError("unsupported plan mode")
    samples = _samples(selected) if mode == "diagnostic" else master if mode == "development-draft" else shards[str(shard_id)]
    plan = {
        "schema": SCHEMA, "identity": IDENTITY, "prepared_date": "2026-10-13", "ablation_variant": variant, "specification_sha256": SPECIFICATION_SHA256,
        "mode": "diagnostic" if mode == "diagnostic" else "development_attack_samples" if mode == "development-draft" else "sample_specific_development_shard",
        "status": status, "source_bindings": SOURCE_MANIFESTS,
        "paper_sha256": PAPER_SHA256,
        "accepted_reference_package_sha256": REFERENCE_MANIFEST_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "protocol_sha256": sha256(ROOT / "PROTOCOL.md"),
        "samples": samples,
        "budget": {"sample_count": len(samples), "updates_per_sample": updates,
                   "total_updates": updates * len(samples)},
        "call_budget_per_sample": planned_budget(updates),
        "clean_reference_detector_forwards_per_sample": 2,
        "scientific_config": scientific_config_for(variant, updates),
        "initialization": "own_unattacked_scene_road_crop",
        "reset_policy": "fresh_latent_scheduler_periodic_controller_rng_per_sample",
        "uncertified_reference_handling": "unit_deficit_preserve_scene_not_success",
        "checkpoint_selection": "fixed_final",
        "evaluation_policy": "each_patch_only_on_its_own_attack_input",
        "test_data_opened": False,
        "optimization_on_original_validation": mode != "diagnostic",
        "held_out_image_generalization": False,
        "gpu_validation_complete": False,
        "scope": "diagnostic only" if mode == "diagnostic" else "formal budget draft pending GPU validation, timing and storage",
    }
    if mode != "diagnostic":
        plan.update(master_samples=master, master_samples_sha256=MASTER_SAMPLES_SHA256,
            partition={"rule": "dataset_sorted_alternating", "shard_count": 2,
                "case_order": "interleave_kitti_nuscenes_within_shard",
                "shard_members": {key: [s["sample_id"] for s in values] for key, values in shards.items()}})
    if mode == "formal":
        plan.update(shard_id=shard_id, diagnostic_plan_sha256=sha256(diagnostic_plan),
            diagnostic_binding=diag_binding, admission_binding=admission_binding,
            gpu_diagnostic_complete=True,
            scope="20 independent own-input patches; 500 updates per sample, 10000 per shard, 20000 across40")
    return validate_plan(plan, allow_draft=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "development-draft", "formal"), required=True)
    parser.add_argument("--shard-id", type=int, choices=(0, 1))
    parser.add_argument("--diagnostic-plan", type=Path)
    parser.add_argument("--diagnostic-receipt", type=Path)
    parser.add_argument("--diagnostic-receipt-sha256")
    parser.add_argument("--admission-receipt", type=Path)
    parser.add_argument("--admission-receipt-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.manifest, args.mode, variant=args.variant, shard_id=args.shard_id,
        diagnostic_plan=args.diagnostic_plan, diagnostic_receipt=args.diagnostic_receipt,
        diagnostic_receipt_sha256=args.diagnostic_receipt_sha256,
        admission_receipt=args.admission_receipt, admission_receipt_sha256=args.admission_receipt_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(args.output), "sha256": sha256(args.output),
                      "status": plan["status"], "budget": plan["budget"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
