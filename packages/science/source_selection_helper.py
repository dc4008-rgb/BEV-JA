#!/usr/bin/env python3
"""Select source RGBs only; never load models, attack artifacts, or TEST images.

Drafts are non-executable until actual source bytes and image-only manual QA
have been bound. Historical manifests, split roles, and results stay read-only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath

SOURCE_SHA = "38ae14801d2e330236bd7b408f08ef29c5f93b4e802c6a8f2414a5a352ceb23f"
OLD_MASTER_SHA = "0538c8ab84521abb088e780be6706e7c340f3b47eddabaf482023d4d6b76c23b"
SEED = 20261014
SCHEMA = "periodic-foa-non-test-200-sample-selection-v1"
PROOF_SCHEMA = "periodic-foa-source-rgb-input-proof-v1"
QA_SCHEMA = "periodic-foa-source-rgb-road-qa-v1"
DIAGNOSTIC_IDS = (
    "kitti_000007",
    "nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404",
)
NATIVE_FIELDS = ("sample_id", "case_id", "dataset", "source_split", "source_path",
                 "source_sha256", "road_quad_norm", "width", "height")
SOURCE_FIELDS = ("sample_id", "dataset", "split", "source_path", "source_filename",
    "source_sha256", "width", "height", "road_quad_norm", "road_score",
    "selection_source", "scene_token", "scene_name", "timestamp", "manual_qa",
    "replaced_candidate_index", "test_slot")
MIN_NUSCENES_SEPARATION_US = 400000
V1_CLEAN_RGB_PROOF_HELPER_SHA = "bf0946faf1b9585f74a227b3155a2f39ac4c41c18bbea9beabce5545747c478d"
QA_REJECTS = (
    "reject_corrupt_image", "reject_non_road_scene", "reject_road_not_visible",
    "reject_severe_road_occlusion", "reject_gross_blur_or_exposure",
)
QA_CRITERIA = {
    "pass": "The source RGB decodes; a forward road is visible; at least half of the fixed road quadrilateral is recognizable road surface; no severe blur/exposure prevents assessment.",
    "reject_corrupt_image": "The actual source image is corrupt or cannot be decoded (source proof must also fail; it cannot be frozen).",
    "reject_non_road_scene": "No forward road scene exists in the source RGB.",
    "reject_road_not_visible": "Less than half of the fixed road quadrilateral is recognizable road surface.",
    "reject_severe_road_occlusion": "A foreground obstacle/vehicle/vegetation occupies more than half of the fixed road quadrilateral.",
    "reject_gross_blur_or_exposure": "Gross blur, darkness, or overexposure prevents judging whether the fixed road quadrilateral is road.",
    "forbidden": "No clean detector/depth output, attack result, confidence, loss, AP, TNR, or effect metric may guide acceptance or replacement; difficult but valid road scenes pass.",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_sha(document):
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "regular non-symlink file required: " + str(path))
    return path


def binding(path):
    p = regular(path)
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": file_sha(p)}


def read_json(path):
    return json.loads(regular(path).read_text(encoding="utf-8"))


def write_new(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return binding(path)


def derived_case_id(row):
    # Only NEW TRAIN paths need escaping. Historical40 native case IDs are exact.
    case_id = row["sample_id"].replace("+", "_plus_") if row["split"] == "train" else row["sample_id"]
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", case_id), "unsafe derived case ID")
    return case_id


def native(row):
    return {"sample_id": row["sample_id"], "case_id": derived_case_id(row),
        "dataset": row["dataset"], "source_split": row["split"],
        "source_path": row["source_path"], "source_sha256": row["source_sha256"],
        "road_quad_norm": row["road_quad_norm"], "width": int(row["width"]),
        "height": int(row["height"])}


def sample(row):
    return {**native(row), "selection_role": "retained_validation40" if row["split"] == "validation" else "added_train160",
        "source_scene_token": row["scene_token"], "source_scene_name": row["scene_name"],
        "source_timestamp": int(row["timestamp"]) if row["timestamp"] else None}


def validate_rows(rows):
    require(len(rows) == 240, "the frozen non-test source must contain exactly240 rows")
    expected = {("kitti", "train"): 100, ("kitti", "validation"): 20,
                ("nuscenes", "train"): 100, ("nuscenes", "validation"): 20}
    require(dict(collections.Counter((r["dataset"], r["split"]) for r in rows)) == expected,
        "only original TRAIN100+validation20 per dataset are allowed; no TEST")
    for key in ("sample_id", "source_sha256", "source_path"):
        require(len({r[key] for r in rows}) == len(rows), "duplicate source " + key)
    scene_splits = collections.defaultdict(set)
    scene_names, times = {}, set()
    for row in rows:
        require(set(row) == set(SOURCE_FIELDS), "unexpected source row fields")
        require(re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]), "source SHA malformed")
        path = PurePosixPath(row["source_path"])
        require(path.is_absolute() and ".." not in path.parts and path.name == row["source_filename"], "unsafe source image path")
        expected_root = "/root/autodl-tmp/kitti_full/image_2/" if row["dataset"] == "kitti" else "/root/autodl-tmp/nuscenes_mini/samples/CAM_FRONT/"
        require(row["source_path"].startswith(expected_root), "source image outside frozen dataset scope")
        require(int(row["width"]) > 0 and int(row["height"]) > 0, "invalid source image dimensions")
        quad = json.loads(row["road_quad_norm"])
        require(len(quad) == 4 and all(len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in p) for p in quad), "invalid fixed road quadrilateral")
        if row["dataset"] == "nuscenes":
            token, name, stamp = row["scene_token"], row["scene_name"], int(row["timestamp"])
            require(re.fullmatch(r"[0-9a-f]{32}", token) and name and stamp > 0, "nuScenes source metadata missing")
            require((token, stamp) not in times, "duplicate nuScenes scene timestamp")
            times.add((token, stamp)); scene_splits[token].add(row["split"])
            require(token not in scene_names or scene_names[token] == name, "nuScenes scene name mismatch")
            scene_names[token] = name
            match = re.fullmatch(r".+__CAM_FRONT__(\d+)\.jpg", row["source_filename"])
            require(match and int(match.group(1)) == stamp, "nuScenes filename timestamp differs")
        else:
            require(not row["scene_token"] and not row["scene_name"] and not row["timestamp"], "KITTI has no verified scene/time metadata")
    require(all(len(splits) == 1 for splits in scene_splits.values()), "nuScenes train and validation scene overlap")
    old = [native(r) for r in sorted(rows, key=lambda r: (r["dataset"], r["sample_id"])) if r["split"] == "validation"]
    require(canonical_sha(old) == OLD_MASTER_SHA, "original40 native source rows changed")
    for sample_id in DIAGNOSTIC_IDS:
        matches = [r for r in rows if r["sample_id"] == sample_id]
        require(len(matches) == 1 and matches[0]["split"] == "train", "fixed diagnostic input differs")
    return rows


def load_source(path):
    p = regular(path)
    require(file_sha(p) == SOURCE_SHA, "frozen source manifest SHA differs")
    with p.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == SOURCE_FIELDS, "frozen source CSV header differs")
        rows = list(reader)
    return validate_rows(rows)


def census(rows):
    scenes = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["dataset"] == "nuscenes":
            scenes[r["split"]][r["scene_name"]].append(int(r["timestamp"]))
    return {"registered_non_test_images": 240, "retained_original_validation": 40,
        "eligible_added_train_per_dataset": {d: sum(r["dataset"] == d and r["split"] == "train" and r["sample_id"] not in DIAGNOSTIC_IDS for r in rows) for d in ("kitti", "nuscenes")},
        "spares_after_80_each": {"kitti": 19, "nuscenes": 19},
        "nuScenes_scene_count": sum(len(s) for s in scenes.values()),
        "nuScenes_scenes": {split: {name: {"registered_images": len(stamps), "first_timestamp": min(stamps), "last_timestamp": max(stamps)} for name, stamps in sorted(groups.items())} for split, groups in sorted(scenes.items())},
        "images_are_not_independent_scene_count": True,
        "KITTI_temporal_independence_verified": False}


def ranked_groups(rows, dataset):
    pool = sorted((r for r in rows if r["dataset"] == dataset and r["split"] == "train"), key=lambda r: (r["scene_token"], int(r["timestamp"] or 0), r["sample_id"]))
    grouped = collections.defaultdict(list)
    if dataset == "kitti":
        pool.sort(key=lambda r: r["sample_id"])
        for index, row in enumerate(pool):
            grouped[("kitti_id_decile", str(index * 10 // len(pool)))].append(row)
    else:
        scenes = collections.defaultdict(list)
        for row in pool:
            scenes[row["scene_token"]].append(row)
        for scene, members in sorted(scenes.items()):
            for index, row in enumerate(members):
                grouped[(scene, str(index * 4 // len(members)))].append(row)
    rank = lambda row: hashlib.sha256((str(SEED) + "\0" + dataset + "\0" + row["sample_id"] + "\0" + row["source_sha256"]).encode()).hexdigest()
    return {group: sorted((r for r in members if r["sample_id"] not in DIAGNOSTIC_IDS), key=lambda r: (rank(r), r["sample_id"])) for group, members in sorted(grouped.items())}


def choose_train(rows, dataset, decisions=None):
    """Water-fill scenes and bins, replacing only recorded RGB-quality rejects.

    Pending rows retain their rank; passing a later image cannot skip an earlier
    pending candidate. The selection does not accept numeric performance input.
    """
    decisions = decisions or {}
    groups = ranked_groups(rows, dataset)
    selected, positions = [], {key: 0 for key in groups}
    group_counts = collections.Counter(); scene_counts = collections.Counter()
    rejected, spacing_skips = [], []
    group_order = {key: hashlib.sha256((str(SEED) + "\0" + "|".join(key)).encode()).hexdigest() for key in groups}
    while len(selected) < 80:
        available = []
        for key, members in groups.items():
            while positions[key] < len(members):
                row = members[positions[key]]
                decision = decisions.get(row["sample_id"], "pending")
                if decision in QA_REJECTS:
                    rejected.append(row["sample_id"]); positions[key] += 1; continue
                require(decision in ("pending", "pass"), "unknown manual QA outcome")
                if dataset == "nuscenes" and any(other["scene_token"] == row["scene_token"] and abs(int(other["timestamp"]) - int(row["timestamp"])) < MIN_NUSCENES_SEPARATION_US for other in selected):
                    spacing_skips.append(row["sample_id"]); positions[key] += 1; continue
                available.append(key); break
        require(available, "insufficient source-quality/spacing candidates for80 " + dataset + "; no TEST borrowing or criteria relaxation")
        key = min(available, key=lambda g: (scene_counts[g[0]] if dataset == "nuscenes" else 0, group_counts[g], group_order[g], g))
        row = groups[key][positions[key]]; positions[key] += 1
        selected.append(row); group_counts[key] += 1; scene_counts[key[0]] += 1
    return selected, {"group_selected_counts": {"|".join(k): v for k, v in sorted(group_counts.items())},
        "source_quality_rejects_encountered": rejected, "timestamp_spacing_skips": spacing_skips,
        "eligible_groups": {"|".join(k): [r["sample_id"] for r in v] for k, v in sorted(groups.items())}}


def eligible_source_rows(rows):
    return sorted((r for r in rows if r["sample_id"] not in DIAGNOSTIC_IDS), key=lambda r: (r["dataset"], r["sample_id"]))


def validate_input_proof(proof, rows):
    require(proof.get("schema") == PROOF_SCHEMA and proof.get("status") == "all238_non_test_source_rgb_verified", "real238 image proof required")
    require(proof.get("source_manifest_sha256") == SOURCE_SHA, "input proof source manifest changed")
    producer = proof.get("selector_source_binding", {})
    require(producer.get("sha256") in (V1_CLEAN_RGB_PROOF_HELPER_SHA, file_sha(__file__)) and type(producer.get("bytes")) is int and producer["bytes"] > 0, "input proof must bind the actual approved source-only helper")
    require(proof.get("test_data_opened") is False and proof.get("attack_outputs_opened") is False and proof.get("model_forward_calls") == 0, "input proof scope must be sourceRGB-only")
    expected = eligible_source_rows(rows)
    observed = proof.get("images")
    require(isinstance(observed, list) and len(observed) == len(expected), "source input proof member count differs")
    for row, member in zip(expected, observed):
        require(member.get("sample_id") == row["sample_id"] and member.get("source_path") == row["source_path"] and member.get("source_sha256") == row["source_sha256"], "source proof member identity/SHA differs")
        require(member.get("width") == int(row["width"]) and member.get("height") == int(row["height"]) and member.get("decoded") is True, "source proof native dimensions/decode failed")
        actual = member.get("actual_file_binding", {})
        require(actual.get("sha256") == row["source_sha256"] and type(actual.get("bytes")) is int and actual["bytes"] > 0 and isinstance(actual.get("path"), str), "real source image byte binding required")
    return proof


def validate_qa(qa, rows, input_proof_sha256):
    require(qa.get("schema") == QA_SCHEMA and qa.get("source_manifest_sha256") == SOURCE_SHA, "manual QA schema/source differs")
    require(qa.get("source_input_proof_sha256") == input_proof_sha256, "manual QA must bind the exact real image proof")
    require(qa.get("review_method") == "source_rgb_and_fixed_road_quad_only" and qa.get("attack_outputs_used_for_selection") is False and qa.get("test_data_opened") is False, "manual QA cannot use attack/model/TEST evidence")
    require(qa.get("criteria_sha256") == canonical_sha(QA_CRITERIA), "predeclared source quality criteria changed")
    entries = qa.get("images")
    expected = eligible_source_rows(rows)
    require(isinstance(entries, list) and len(entries) == len(expected), "manual QA must include all238 source row slots")
    decisions = {}
    allowed = {"sample_id", "source_sha256", "outcome", "reviewer", "reviewed_at", "note"}
    for row, entry in zip(expected, entries):
        require(set(entry) == allowed and entry["sample_id"] == row["sample_id"] and entry["source_sha256"] == row["source_sha256"], "QA member is missing, reordered, duplicate, or has unapproved performance fields")
        outcome = entry["outcome"]
        require(outcome in ("pending", "pass") + QA_REJECTS, "unsupported QA outcome")
        if outcome != "pending":
            require(isinstance(entry["reviewer"], str) and entry["reviewer"].strip() and isinstance(entry["reviewed_at"], str) and entry["reviewed_at"].strip(), "actual reviewer/time required")
        if outcome in QA_REJECTS:
            require(isinstance(entry["note"], str) and entry["note"].strip(), "an image-only rejection explanation is required")
        decisions[row["sample_id"]] = outcome
    return decisions


def make_selection(rows, *, source_binding, decisions=None, input_binding=None, qa_binding=None, frozen=False):
    retained = [r for r in rows if r["split"] == "validation"]
    selected, reports = list(retained), {}
    for dataset in ("kitti", "nuscenes"):
        added, reports[dataset] = choose_train(rows, dataset, decisions)
        selected.extend(added)
    selected.sort(key=lambda r: (r["dataset"], r["sample_id"]))
    require(len(selected) == 200 and len({r["sample_id"] for r in selected}) == 200, "selected IDs not unique200")
    require(len({derived_case_id(r) for r in selected}) == 200, "derived case ID collision")
    require(not any(r["sample_id"] in DIAGNOSTIC_IDS for r in selected), "diagnostic input leaked into200")
    if frozen:
        require(input_binding and qa_binding and decisions, "real source proof plus manual QA required before freeze")
        for row in selected:
            require(decisions.get(row["sample_id"]) == "pass", "selected image has not passed source-only manual QA: " + row["sample_id"])
        for row in retained:
            require(decisions.get(row["sample_id"]) == "pass", "original40 must remain; reject/pending stops freeze rather than replacing history")
    samples = [sample(r) for r in selected]
    return {"schema": SCHEMA, "status": "frozen_source_image_qa_passed" if frozen else "candidate_pending_source_image_qa",
        "prepared_date": "2026-10-14", "selection_seed": SEED,
        "source_manifest_sha256": SOURCE_SHA, "source_manifest_binding": source_binding,
        "old_master40_sha256": OLD_MASTER_SHA, "master_samples_sha256": canonical_sha(samples),
        "samples": samples, "sample_count": 200,
        "counts_by_dataset": {"kitti": 100, "nuscenes": 100},
        "counts_by_selection_role": {"retained_validation40": 40, "added_train160": 160},
        "diagnostic_sample_ids": list(DIAGNOSTIC_IDS), "source_pool_census": census(rows),
        "selection_algorithm": "seed20261014_SHA256_waterfill_scene_and_fixed_pool_bins",
        "case_id_derivation": "new_train_plus_to__plus__only_original40_case_ids_exact",
        "selection_reports": reports, "criteria": QA_CRITERIA, "criteria_sha256": canonical_sha(QA_CRITERIA),
        "min_nuScenes_same_scene_timestamp_separation_us": MIN_NUSCENES_SEPARATION_US,
        "source_input_proof_binding": input_binding, "manual_qa_binding": qa_binding,
        "all_selected_source_images_verified": frozen, "all_selected_manual_road_qa_passed": frozen,
        "test_data_opened": False, "attack_outputs_used_for_selection": False,
        "optimization_on_original_validation": True, "held_out_image_generalization": False,
        "scope": "non-test per-input optimization/evaluation development extension, not held-out200test",
        "training_authorized_by_this_artifact": False if not frozen else "requires_separately_admitted_scientific200_plan"}


def make_qa_template(rows, proof_binding=None):
    return {"schema": QA_SCHEMA, "source_manifest_sha256": SOURCE_SHA,
        "source_input_proof_sha256": proof_binding["sha256"] if proof_binding else None,
        "review_method": "source_rgb_and_fixed_road_quad_only", "criteria": QA_CRITERIA,
        "criteria_sha256": canonical_sha(QA_CRITERIA), "test_data_opened": False,
        "attack_outputs_used_for_selection": False,
        "images": [{"sample_id": r["sample_id"], "source_sha256": r["source_sha256"],
            "outcome": "pending", "reviewer": "", "reviewed_at": "", "note": ""}
            for r in eligible_source_rows(rows)]}


def mapped_path(source_path, prefixes):
    matches = [(old, new) for old, new in prefixes.items() if source_path.startswith(old.rstrip("/") + "/")]
    if not matches:
        return Path(source_path)
    old, new = max(matches, key=lambda pair: len(pair[0]))
    suffix = source_path[len(old.rstrip("/") + "/"):]
    require(".." not in PurePosixPath(suffix).parts, "mapped source path traversal")
    root = Path(new).resolve(); p = root / suffix
    resolved = p.resolve()
    require(resolved == root or root in resolved.parents, "mapped source path escaped")
    return p


def verify_inputs(rows, source_path, *, prefixes=None, preview_dir=None):
    from PIL import Image, ImageDraw
    prefixes = prefixes or {}
    previews, observed = [], []
    if preview_dir:
        preview_dir = Path(preview_dir)
        require(not preview_dir.exists(), "preview directory already exists; no overwrite")
        preview_dir.mkdir(parents=True)
    for row in eligible_source_rows(rows):
        path = regular(mapped_path(row["source_path"], prefixes))
        content = path.read_bytes(); digest = hashlib.sha256(content).hexdigest()
        require(digest == row["source_sha256"], "actual source RGB SHA differs: " + row["sample_id"])
        with Image.open(io.BytesIO(content)) as raw:
            raw.load()
            require(raw.size == (int(row["width"]), int(row["height"])), "actual source RGB dimensions differ")
            rgb = raw.convert("RGB")
        observed.append({"sample_id": row["sample_id"], "source_path": row["source_path"],
            "source_sha256": digest, "actual_file_binding": {"path": str(path.resolve()), "bytes": len(content), "sha256": digest},
            "width": rgb.width, "height": rgb.height, "decoded": True})
        if preview_dir:
            thumb = rgb.copy(); thumb.thumbnail((400, 200))
            canvas = Image.new("RGB", (410, 254), "white"); canvas.paste(thumb, (5, 5))
            draw = ImageDraw.Draw(canvas)
            quad = [(5 + x * thumb.width, 5 + y * thumb.height) for x, y in json.loads(row["road_quad_norm"])]
            draw.line(quad + [quad[0]], fill="red", width=2)
            label = row["sample_id"] if row["dataset"] == "kitti" else row["scene_name"] + " / " + row["timestamp"]
            draw.text((5, 207), label, fill="black")
            draw.text((5, 226), row["split"] + " | " + row["source_sha256"][:16], fill="black")
            previews.append(canvas)
    sheets = []
    if preview_dir:
        for index in range(0, len(previews), 20):
            sheet = Image.new("RGB", (5 * 410, 4 * 254), "white")
            for offset, img in enumerate(previews[index:index + 20]):
                sheet.paste(img, ((offset % 5) * 410, (offset // 5) * 254))
            path = preview_dir / ("source_rgb_road_qa_%02d.jpg" % (index // 20 + 1))
            sheet.save(path, quality=88)
            sheets.append(binding(path))
    return {"schema": PROOF_SCHEMA, "status": "all238_non_test_source_rgb_verified",
        "source_manifest_sha256": SOURCE_SHA, "source_manifest_binding": binding(source_path),
        "selector_source_binding": binding(__file__), "image_count": len(observed), "images": observed,
        "contact_sheets": sheets, "test_data_opened": False, "attack_outputs_opened": False,
        "model_forward_calls": 0, "scope": "byte SHA/native dimensions/CPU decode only; road manual QA still required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("census", "draft", "verify-inputs", "qa-template", "freeze"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-proof", type=Path)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--prefix-map", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    rows = load_source(args.manifest)
    if args.command == "census":
        print(json.dumps(census(rows), ensure_ascii=False, indent=2)); return
    require(args.output is not None, "explicit new output path required")
    if args.command == "draft":
        document = make_selection(rows, source_binding=binding(args.manifest))
    elif args.command == "verify-inputs":
        prefixes = read_json(args.prefix_map) if args.prefix_map else {}
        document = verify_inputs(rows, args.manifest, prefixes=prefixes, preview_dir=args.preview_dir)
    elif args.command == "qa-template":
        require(args.input_proof is not None, "source image proof required")
        validate_input_proof(read_json(args.input_proof), rows)
        document = make_qa_template(rows, binding(args.input_proof))
    else:
        require(args.input_proof is not None and args.qa is not None, "actual source proof/manual QA required")
        validate_input_proof(read_json(args.input_proof), rows)
        input_binding = binding(args.input_proof); qa_binding = binding(args.qa)
        decisions = validate_qa(read_json(args.qa), rows, input_binding["sha256"])
        document = make_selection(rows, source_binding=binding(args.manifest), decisions=decisions,
            input_binding=input_binding, qa_binding=qa_binding, frozen=True)
    print(json.dumps(write_new(args.output, document), ensure_ascii=False))


if __name__ == "__main__":
    main()
