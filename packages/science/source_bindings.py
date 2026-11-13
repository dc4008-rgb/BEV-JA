"""Immutable source gates for the user-specified periodic FOA candidate."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

SOURCE_MANIFESTS = {
    "child": "a4ee2aab2ab40fe9c122c16bc05fd5a32cbd06286413384f4ad4086c26b863cb",
    "r2_source": "92af9063dbe07898eb562b0d324e9d1bd746467ce1edea408f9b5f8940f82c85",
    "v2_source": "6a9c07f0d1fcb7d1fac6b2435935ede8c45c73f13f7b9bc01bd47c57d2c23c74",
    "clip_source": "65783ee74f559a8c94cfb022d461c721de61a213745812cca7d03fe1592bf20d",
}
DATA_MANIFEST_SHA256 = "38ae14801d2e330236bd7b408f08ef29c5f93b4e802c6a8f2414a5a352ceb23f"
MODEL_MANIFEST_SHA256 = "d994315183884c678c35e1b4cdb03c0b459b74ba27fd16bc95e3c17717c0b5dc"
REFERENCE_MANIFEST_SHA256 = "7ffb0f18766b752bf6db82ea475d77a524f6fb2e44f1ed5e26df41cdf9625b4b"
SPECIFICATION_SHA256 = "fa773c31bfc368d2ea4cf4b7fddb6ba828192257889d96f073848b2d6080c4e8"
PAPER_SHA256 = "bd18adfc3c039eb7c800f592d95e1a5f5be784e161fd2a8475ae471bb3ddb1b2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tree(root, expected):
    root = Path(root).resolve()
    manifest = root / "PACKAGE_MANIFEST.json"
    if sha256(manifest) != expected:
        raise ValueError("source package identity differs: " + str(root))
    document = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in document["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source package member")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("source package member escapes its tree")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("source package member changed: " + str(path))
    return {"manifest_sha256": expected, "verified_files": len(document["files"])}


def verify_reference_tree(root):
    """Only verify the accepted Ours source; no use of old Patch or old weights."""
    return verify_tree(root, REFERENCE_MANIFEST_SHA256)


def verify_plan_samples(plan, manifest):
    if sha256(manifest) != DATA_MANIFEST_SHA256:
        raise ValueError("data manifest identity differs")
    with Path(manifest).open(encoding="utf-8", newline="") as stream:
        rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    for sample in plan["samples"]:
        row = rows.get(sample["sample_id"])
        if row is None:
            raise ValueError("attack input is not in the bound development manifest")
        for field in ("dataset", "source_path", "source_sha256", "road_quad_norm"):
            if sample[field] != row[field]:
                raise ValueError("attack input differs from the frozen source row: " + field)
        if sample["source_split"] != row["split"] or row["split"] == "test":
            raise ValueError("attack input split differs or is test")
        for field in ("width", "height"):
            if int(sample[field]) != int(row[field]):
                raise ValueError("attack input dimensions differ")


def verify_sources(plan, *, child, r2_source, v2_source, clip_source):
    if plan["source_bindings"] != SOURCE_MANIFESTS:
        raise ValueError("unreviewed scientific source identity")
    paths = {key: Path(value).resolve() for key, value in {
        "child": child, "r2_source": r2_source, "v2_source": v2_source,
        "clip_source": clip_source,
    }.items()}
    verified = {key: verify_tree(path, SOURCE_MANIFESTS[key]) for key, path in paths.items()}
    parent = paths["child"] / "parent_v17"
    verify_plan_samples(plan, parent / "01_公共数据集/runtime_v1/train_validation_manifest.csv")
    if sha256(parent / "02_模型/model_manifest.json") != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest identity differs")
    return {**paths, "verified": verified}
