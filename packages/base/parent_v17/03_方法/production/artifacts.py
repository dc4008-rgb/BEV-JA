"""Replayable artifact storage for normalized universal attacks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


SCHEMA_VERSION = 2
IMPLEMENTATION_FILES = (
    "../../02_模型/model_adapters.py",
    "advrm_regularizer.py",
    "artifacts.py",
    "carriers.py",
    "data.py",
    "losses.py",
    "objectives.py",
    "printable30values.txt",
    "stable_optimizers.py",
    "trainer.py",
)
REQUIRED_METADATA = {
    "config_id",
    "method_id",
    "variant",
    "carrier_id",
    "carrier_kind",
    "train_models",
    "seed",
    "updates",
    "test_time_optimization",
    "dataset_manifest_sha256",
    "method_contracts_sha256",
    "model_manifest_sha256",
    "method_source_manifest_sha256",
    "implementation_sha256",
}


@dataclass(frozen=True)
class LoadedArtifact:
    metadata: dict[str, Any]
    trainable_state: dict[str, torch.Tensor]
    canonical_rgb: torch.Tensor
    zero_step: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint(
    production_dir: Path,
) -> tuple[str, dict[str, str]]:
    hashes = {
        name: sha256(production_dir / name)
        for name in IMPLEMENTATION_FILES
    }
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), hashes


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"artifact metadata is missing fields: {missing}")
    if metadata["test_time_optimization"] is not False:
        raise ValueError("artifact must forbid test-time optimization")
    if not isinstance(metadata["train_models"], list) or not metadata["train_models"]:
        raise ValueError("artifact requires a non-empty train model list")
    if not isinstance(metadata["updates"], int) or metadata["updates"] < 1:
        raise ValueError("artifact update count must be positive")
    for name in (
        "dataset_manifest_sha256",
        "method_contracts_sha256",
        "model_manifest_sha256",
        "method_source_manifest_sha256",
        "implementation_sha256",
    ):
        value = metadata[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid SHA-256 field: {name}")


def _validate_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"artifact entry is not a tensor: {name}")
    if not torch.isfinite(value).all():
        raise ValueError(f"artifact tensor is non-finite: {name}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def capture_zero_step(
    carrier: torch.nn.Module,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    control = {
        "attack_updates": 0,
        "canonical_rgb": carrier.canonical_rgb().detach().cpu().contiguous().clone(),
        "trainable_state": {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in carrier.named_parameters() if parameter.requires_grad
        },
        "reference_buffers": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in carrier.named_buffers(recurse=False)
        },
        "initialization": dict(initialization),
    }
    validate_zero_step(control)
    return control


def validate_zero_step(control: dict[str, Any]) -> None:
    if not isinstance(control, dict) or control.get("attack_updates") != 0:
        raise ValueError("artifact requires a zero-step control captured before attack updates")
    canonical = control.get("canonical_rgb")
    _validate_tensor("zero_step.canonical_rgb", canonical)
    if canonical.ndim != 4 or canonical.shape[:2] != (1, 3) or canonical.min() < 0 or canonical.max() > 1:
        raise ValueError("invalid zero-step canonical RGB")
    state = control.get("trainable_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("zero-step control has no trainable state")
    buffers = control.get("reference_buffers")
    if not isinstance(buffers, dict):
        raise ValueError("zero-step control has no reference buffers")
    for name, value in {**state, **buffers}.items():
        _validate_tensor(f"zero_step.{name}", value)
    if not isinstance(control.get("initialization"), dict):
        raise ValueError("zero-step control has no initialization metadata")


def save_artifact(
    output_dir: Path,
    carrier: torch.nn.Module,
    canonical_rgb: torch.Tensor,
    metadata: dict[str, Any],
    trace: Iterable[dict[str, Any]],
    *,
    zero_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_metadata(metadata)
    validate_zero_step(zero_step)
    canonical = canonical_rgb.detach().cpu().contiguous()
    if canonical.ndim != 4 or canonical.shape[:2] != (1, 3):
        raise ValueError("canonical RGB must have shape [1,3,H,W]")
    _validate_tensor("canonical_rgb", canonical)
    if canonical.min() < 0 or canonical.max() > 1:
        raise ValueError("canonical RGB is outside [0,1]")

    trainable_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in carrier.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_state:
        raise ValueError("carrier has no trainable state")
    for name, value in trainable_state.items():
        _validate_tensor(name, value)

    if set(zero_step["trainable_state"]) != set(trainable_state):
        raise ValueError("zero-step and final trainable state names differ")
    if zero_step["canonical_rgb"].shape != canonical.shape:
        raise ValueError("zero-step and final canonical RGB shapes differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "artifact.pt"
    weights_tmp = output_dir / "artifact.pt.tmp"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "carrier_id": metadata["carrier_id"],
            "carrier_class": type(carrier).__name__,
            "carrier_kind": metadata["carrier_kind"],
            "trainable_state": trainable_state,
            "canonical_rgb": canonical,
            "zero_step": zero_step,
        },
        weights_tmp,
    )
    os.replace(weights_tmp, weights_path)

    trace_path = output_dir / "training_trace.jsonl"
    trace_tmp = output_dir / "training_trace.jsonl.tmp"
    trace_count = 0
    with trace_tmp.open("w", encoding="utf-8") as stream:
        for record in trace:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            trace_count += 1
    if trace_count < 1:
        trace_tmp.unlink(missing_ok=True)
        raise ValueError("artifact trace is empty")
    os.replace(trace_tmp, trace_path)

    complete = dict(metadata)
    complete.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "artifact_file": weights_path.name,
            "artifact_sha256": sha256(weights_path),
            "trace_file": trace_path.name,
            "trace_sha256": sha256(trace_path),
            "trace_records": trace_count,
            "trainable_parameter_names": sorted(trainable_state),
            "trainable_parameter_count": sum(value.numel() for value in trainable_state.values()),
            "canonical_rgb_shape": list(canonical.shape),
            "zero_step_control": {
                "available": True,
                "attack_updates": 0,
                "initialization": zero_step["initialization"],
                "reference": "same_carrier_before_attack_updates_not_original_car_identity",
            },
        }
    )
    _atomic_json(output_dir / "metadata.json", complete)
    return complete


def load_artifact(output_dir: Path) -> LoadedArtifact:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata)
    if metadata.get("schema_version") not in {1, SCHEMA_VERSION} or metadata.get("status") != "complete":
        raise ValueError("artifact metadata is not a complete supported schema")

    weights_path = output_dir / metadata["artifact_file"]
    trace_path = output_dir / metadata["trace_file"]
    if sha256(weights_path) != metadata["artifact_sha256"]:
        raise RuntimeError("artifact weights hash mismatch")
    if sha256(trace_path) != metadata["trace_sha256"]:
        raise RuntimeError("artifact trace hash mismatch")
    with trace_path.open(encoding="utf-8") as stream:
        trace_count = sum(1 for line in stream if line.strip())
    if trace_count != metadata["trace_records"]:
        raise RuntimeError("artifact trace record count mismatch")

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != metadata["schema_version"]:
        raise ValueError("artifact weights use an unsupported schema")
    trainable_state = payload.get("trainable_state")
    canonical = payload.get("canonical_rgb")
    if not isinstance(trainable_state, dict) or not trainable_state:
        raise ValueError("artifact has no trainable state")
    for name, value in trainable_state.items():
        _validate_tensor(name, value)
    _validate_tensor("canonical_rgb", canonical)
    if list(canonical.shape) != metadata["canonical_rgb_shape"]:
        raise RuntimeError("canonical RGB shape does not match metadata")
    if sorted(trainable_state) != metadata["trainable_parameter_names"]:
        raise RuntimeError("trainable state names do not match metadata")
    zero_step = payload.get("zero_step") if metadata["schema_version"] == SCHEMA_VERSION else None
    if metadata["schema_version"] == SCHEMA_VERSION:
        validate_zero_step(zero_step)
        if set(zero_step["trainable_state"]) != set(trainable_state):
            raise ValueError("zero-step and final trainable state names differ")
        if zero_step["canonical_rgb"].shape != canonical.shape:
            raise ValueError("zero-step and final canonical RGB shapes differ")
        if metadata.get("zero_step_control", {}).get("initialization") != zero_step["initialization"]:
            raise ValueError("zero-step initialization metadata mismatch")
    return LoadedArtifact(metadata, trainable_state, canonical, zero_step)