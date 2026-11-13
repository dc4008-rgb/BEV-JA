#!/usr/bin/env python3
"""Validate ten real one-step runs and freeze the production implementation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from artifacts import implementation_fingerprint, load_artifact, sha256


REPRESENTATIVE_CONFIGS = {
    "upc": "upc__single_fasterrcnn",
    "das": "das__single_fasterrcnn",
    "fca": "fca__single_fasterrcnn",
    "tt3d_detector_adaptation": "tt3d_detector_adaptation__single_fasterrcnn",
    "apa": "apa__single_midas",
    "saam": "saam__single_midas",
    "3d2fool": "3d2fool__single_midas",
    "advrm": "advrm__single_midas",
    "bittaa": "bittaa__pair_fasterrcnn_midas",
    "ours": "ours__pair_fasterrcnn_midas",
}


def validate_trace(path: Path) -> tuple[dict[str, Any], str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"implementation smoke must have one trace record: {path}")
    record = json.loads(lines[0])
    gradient_norm = float(record["gradient_norm"])
    total_loss = float(record["total_loss"])
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise ValueError(f"invalid implementation gradient: {path}")
    if not math.isfinite(total_loss):
        raise ValueError(f"invalid implementation loss: {path}")
    return record, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    implementation_sha256, implementation_files = implementation_fingerprint(
        args.production_dir
    )
    evidence = []
    for method_id, config_id in REPRESENTATIVE_CONFIGS.items():
        root = args.smoke_root / config_id
        artifact = load_artifact(root)
        metadata = artifact.metadata
        if metadata["method_id"] != method_id or metadata["config_id"] != config_id:
            raise ValueError(f"method/config mismatch: {root}")
        if metadata["run_mode"] != "implementation-smoke":
            raise ValueError(f"unexpected run mode: {root}")
        if metadata["updates"] != 1 or metadata["trace_records"] != 1:
            raise ValueError(f"unexpected smoke scale: {root}")
        if metadata["implementation_sha256"] != implementation_sha256:
            raise ValueError(f"implementation hash mismatch: {root}")
        if metadata["test_time_optimization"] is not False:
            raise ValueError(f"test-time optimization enabled: {root}")
        trace, trace_sha256 = validate_trace(root / metadata["trace_file"])
        evidence.append(
            {
                "method_id": method_id,
                "config_id": config_id,
                "metadata_path": str(root / "metadata.json"),
                "metadata_sha256": sha256(root / "metadata.json"),
                "artifact_sha256": metadata["artifact_sha256"],
                "trace_sha256": trace_sha256,
                "gradient_norm": trace["gradient_norm"],
                "total_loss": trace["total_loss"],
                "elapsed_seconds": trace["elapsed_seconds"],
                "peak_cuda_memory_bytes": metadata["peak_cuda_memory_bytes"],
                "runtime": metadata["runtime"],
            }
        )

    if {item["method_id"] for item in evidence} != set(REPRESENTATIVE_CONFIGS):
        raise RuntimeError("implementation evidence does not cover ten unique methods")
    report = {
        "schema_version": 1,
        "status": "pass",
        "scope": "ten_method_real_model_one_step_and_replayable_artifact",
        "method_count": len(evidence),
        "implementation_sha256": implementation_sha256,
        "implementation_files_sha256": implementation_files,
        "smoke_root": str(args.smoke_root),
        "runtime_policy": {
            "frozen_model_runtime": "venv: transformers 4.46.3",
            "ours_sd35_runtime": "base: transformers 5.7.0 + diffusers 0.38.0",
            "evaluation_runtime": "venv: transformers 4.46.3",
            "package_mutation": False,
        },
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()