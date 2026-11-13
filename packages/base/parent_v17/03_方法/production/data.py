"""Strict manifest handling for normalized training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("kitti", "nuscenes")
TRAINING_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class SceneRecord:
    sample_id: str
    dataset: str
    split: str
    source_path: Path
    source_sha256: str
    road_quad_norm: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sample_id",
            "dataset",
            "split",
            "source_path",
            "source_sha256",
            "road_quad_norm",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest is missing required columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest contains duplicate sample ids")
    return rows


def _records(rows: list[dict[str, str]]) -> list[SceneRecord]:
    return [
        SceneRecord(
            sample_id=row["sample_id"],
            dataset=row["dataset"],
            split=row["split"],
            source_path=Path(row["source_path"]),
            source_sha256=row["source_sha256"],
            road_quad_norm=row["road_quad_norm"],
        )
        for row in rows
    ]


def load_training_manifest(path: Path) -> list[SceneRecord]:
    rows = _read_rows(path)
    unexpected = sorted({row["split"] for row in rows} - set(TRAINING_SPLITS))
    if unexpected:
        raise ValueError(f"training manifest contains forbidden splits: {unexpected}")
    counts = Counter((row["dataset"], row["split"]) for row in rows)
    expected = {
        ("kitti", "train"): 100,
        ("kitti", "validation"): 20,
        ("nuscenes", "train"): 100,
        ("nuscenes", "validation"): 20,
    }
    if counts != expected:
        raise ValueError(f"training manifest count mismatch: {dict(counts)}")
    return _records(rows)


def load_test_manifest(path: Path) -> list[SceneRecord]:
    rows = _read_rows(path)
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("evaluation manifest must contain only test rows")
    counts = Counter(row["dataset"] for row in rows)
    if counts != {"kitti": 100, "nuscenes": 100}:
        raise ValueError(f"test manifest count mismatch: {dict(counts)}")
    return _records(rows)


def calibration_records(records: list[SceneRecord]) -> tuple[SceneRecord, ...]:
    selected = []
    for dataset in DATASETS:
        candidates = sorted(
            (
                record
                for record in records
                if record.dataset == dataset and record.split == "train"
            ),
            key=lambda record: record.sample_id,
        )
        if len(candidates) < 2:
            raise ValueError(f"not enough calibration rows for {dataset}")
        selected.extend(candidates[:2])
    return tuple(selected)


def validation_records(records: list[SceneRecord]) -> tuple[SceneRecord, ...]:
    selected = tuple(record for record in records if record.split == "validation")
    if Counter(record.dataset for record in selected) != {
        "kitti": 20,
        "nuscenes": 20,
    }:
        raise ValueError("validation rows are incomplete")
    return selected


def seeded_round_robin(
    records: list[SceneRecord], updates: int, seed: int
) -> tuple[SceneRecord, ...]:
    if updates < 1:
        raise ValueError("updates must be positive")
    grouped = {
        dataset: [
            record
            for record in records
            if record.dataset == dataset and record.split == "train"
        ]
        for dataset in DATASETS
    }
    if any(not grouped[dataset] for dataset in DATASETS):
        raise ValueError("both datasets require train rows")
    generator = random.Random(seed)
    schedule: list[SceneRecord] = []
    while len(schedule) < updates:
        shuffled = {}
        for dataset in DATASETS:
            shuffled[dataset] = list(grouped[dataset])
            generator.shuffle(shuffled[dataset])
        cycle_length = max(len(shuffled[dataset]) for dataset in DATASETS)
        for index in range(cycle_length):
            for dataset in DATASETS:
                if index < len(shuffled[dataset]):
                    schedule.append(shuffled[dataset][index])
                    if len(schedule) == updates:
                        return tuple(schedule)
    return tuple(schedule)


def verify_source(record: SceneRecord) -> None:
    if not record.source_path.is_file():
        raise FileNotFoundError(record.source_path)
    actual = sha256(record.source_path)
    if actual != record.source_sha256:
        raise RuntimeError(f"source hash mismatch: {record.sample_id}")