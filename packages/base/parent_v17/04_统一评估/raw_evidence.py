"""Lossless content-addressed depth evidence; no pickle or numeric conversion.

All references are relative to an explicitly supplied evidence root. Tensor
objects bind dtype, shape and exact contiguous CPU bytes before zlib compression.
Original/zero/final observations have separate identities while identical RGB,
masks and depth stages are stored only once, including across configurations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 2
STATES = ("original", "zero_step", "final")
REQUIRED_STAGES = (
    "native_output", "resized_output", "clamped_output", "final_depth",
    "evaluation_output",
    "lower_clamp_mask", "upper_clamp_mask", "nonfinite_resized_mask",
)
REQUIRED_BINDINGS = (
    "config_id", "sample_id", "dataset", "split", "model", "source_sha256",
    "manifest_sha256", "artifact_sha256", "artifact_metadata_sha256",
    "evaluator_sha256", "model_adapters_sha256", "model_manifest_sha256",
    "model_definition_sha256", "car_sha256", "car_mask_sha256",
    "raw_evidence_sha256",
)
DTYPES = {str(value): value for value in (
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.complex64, torch.complex128,
)}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def referenced_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"raw reference escapes evidence root: {relative}")
    return path


def _write_immutable(path: Path, data: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing content-addressed object is corrupt: {path}")
        return False
    # Publish only complete files, without overwriting another writer's object.
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"conflicting raw evidence object: {path}")
            return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def tensor_payload(value: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    tensor = value.detach().cpu().contiguous()
    if str(tensor.dtype) not in DTYPES or tensor.layout != torch.strided:
        raise ValueError(f"unsupported raw tensor dtype/layout: {tensor.dtype}/{tensor.layout}")
    # uint8 view preserves bfloat16, signed zeros and NaN payloads exactly.
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    header = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "byteorder": sys.byteorder, "nbytes": len(raw)}
    encoded = canonical_json(header)
    return struct.pack(">I", len(encoded)) + encoded + raw, {**header, "bytes_sha256": digest_bytes(raw)}


def tensor_identity(value: torch.Tensor) -> str:
    return digest_bytes(tensor_payload(value)[0])


class RawEvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.created_bytes = 0
        self.created_objects = 0
        self.reused_objects = 0
        self.state_count = 0
        self.logical_tensor_bytes = 0

    def put_tensor(self, value: torch.Tensor) -> dict[str, Any]:
        payload, metadata = tensor_payload(value)
        content_hash = digest_bytes(payload)
        compressed = zlib.compress(payload, level=6)
        relative = f"tensors/{content_hash[:2]}/{content_hash}.tensor.zlib"
        created = _write_immutable(referenced_path(self.root, relative), compressed)
        self.created_objects += int(created)
        self.reused_objects += int(not created)
        self.created_bytes += len(compressed) if created else 0
        self.logical_tensor_bytes += metadata["nbytes"]
        return {
            "path": relative, "sha256": digest_bytes(compressed), "tensor_sha256": content_hash,
            "codec": "zlib", "stored_bytes": len(compressed), **metadata,
        }

    def put_state(
        self, state: str, prediction: Any, input_rgb: torch.Tensor,
        object_mask: torch.Tensor, carrier_support: torch.Tensor,
        bindings: dict[str, Any], parameters: dict[str, Any],
    ) -> dict[str, Any]:
        if state not in STATES:
            raise ValueError(f"unknown observation state: {state}")
        if not set(REQUIRED_STAGES).issubset(prediction.stages):
            raise ValueError("diagnostics omit mandatory raw depth stages")
        if tensor_identity(prediction.depth) != tensor_identity(prediction.stages["final_depth"]):
            raise ValueError("diagnostic final_depth differs from the adapter objective output")
        if not prediction.metadata.get("evaluation_semantics"):
            raise ValueError("diagnostics omit evaluation semantics")
        content = {
            "schema_version": SCHEMA_VERSION, "state": state, "bindings": bindings,
            "parameters": parameters, "adapter_metadata": prediction.metadata,
            "tensors": {
                "input_rgb": self.put_tensor(input_rgb),
                "object_mask": self.put_tensor(object_mask),
                "carrier_support": self.put_tensor(carrier_support),
                "stages": {name: self.put_tensor(value) for name, value in prediction.stages.items()},
            },
        }
        encoded = canonical_json(content) + b"\n"
        content_hash = digest_bytes(encoded)
        relative = f"states/{content_hash[:2]}/{content_hash}.json"
        created = _write_immutable(referenced_path(self.root, relative), encoded)
        self.created_bytes += len(encoded) if created else 0
        self.created_objects += int(created)
        self.reused_objects += int(not created)
        self.state_count += 1
        return {"path": relative, "sha256": content_hash}

    def statistics(self) -> dict[str, int]:
        return {
            "stored_states": self.state_count, "created_objects": self.created_objects,
            "reused_objects": self.reused_objects, "created_bytes": self.created_bytes,
            "logical_tensor_bytes": self.logical_tensor_bytes,
        }


def load_tensor(root: Path, reference: dict[str, Any]) -> torch.Tensor:
    path = referenced_path(root, reference["path"])
    data = path.read_bytes()
    if len(data) != reference["stored_bytes"] or digest_bytes(data) != reference["sha256"]:
        raise ValueError(f"compressed tensor hash/length mismatch: {path}")
    if reference.get("codec") != "zlib":
        raise ValueError("unsupported tensor codec")
    if reference.get("dtype") not in DTYPES:
        raise ValueError("unsupported tensor dtype")
    shape = reference.get("shape")
    if not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape):
        raise ValueError("invalid tensor shape")
    dtype = DTYPES[reference["dtype"]]
    expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if expected_bytes != reference["nbytes"] or expected_bytes > 1024 ** 3:
        raise ValueError("invalid or excessive tensor byte count")
    decoder = zlib.decompressobj()
    payload = decoder.decompress(data, expected_bytes + 65540)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid or oversized compressed tensor stream")
    if digest_bytes(payload) != reference["tensor_sha256"] or len(payload) < 4:
        raise ValueError("tensor content hash mismatch")
    header_length = struct.unpack(">I", payload[:4])[0]
    if header_length > 65536:
        raise ValueError("oversized tensor header")
    header = json.loads(payload[4:4 + header_length])
    raw = payload[4 + header_length:]
    for key in ("dtype", "shape", "nbytes", "byteorder"):
        if header.get(key) != reference.get(key):
            raise ValueError(f"tensor metadata mismatch: {key}")
    if header["byteorder"] != sys.byteorder:
        raise ValueError("cross-endian tensor loading is unsupported; raw bytes preserved")
    if len(raw) != expected_bytes or digest_bytes(raw) != reference["bytes_sha256"]:
        raise ValueError("raw tensor byte hash/length mismatch")
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()


def _tensor_references(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = manifest["tensors"]
    return [tensors[name] for name in ("input_rgb", "object_mask", "carrier_support")] + list(tensors["stages"].values())


def load_state(root: Path, state_ref: dict[str, Any], *, load_tensors: bool = True) -> dict[str, Any]:
    path = referenced_path(root, state_ref["path"])
    encoded = path.read_bytes()
    if digest_bytes(encoded) != state_ref["sha256"]:
        raise ValueError(f"state manifest hash mismatch: {path}")
    manifest = json.loads(encoded)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("state") not in STATES:
        raise ValueError("invalid state manifest schema/state")
    if not set(REQUIRED_STAGES).issubset(manifest["tensors"]["stages"]):
        raise ValueError("state omits mandatory depth stages")
    tensors: dict[str, Any] = {}
    if load_tensors:
        tensors = {name: load_tensor(root, manifest["tensors"][name]) for name in ("input_rgb", "object_mask", "carrier_support")}
        tensors["stages"] = {name: load_tensor(root, reference) for name, reference in manifest["tensors"]["stages"].items()}
    return {"manifest": manifest, "tensors": tensors}


def validate_record_raw(
    record: dict[str, Any], raw_root: Path, *,
    expected_bindings: dict[str, Any] | None = None, require_complete: bool = True,
) -> dict[str, Any]:
    """Independently check every state and tensor; return errors instead of raising.

    expected_bindings must be supplied from the independently loaded protocol,
    model/sample manifest and artifact when accepting a formal delivery. Merely
    passing this self-consistency check cannot establish provenance by itself.
    """
    errors: list[str] = []
    files: set[str] = set()
    count = 0
    shared_identity = None
    evidence = record.get("raw_depth_evidence", {})
    if record.get("task") != "depth":
        return {"valid": True, "errors": [], "referenced_files": [], "state_count": 0}
    if evidence.get("schema_version") != SCHEMA_VERSION:
        errors.append("raw evidence schema mismatch")
    if not evidence.get("available"):
        if require_complete:
            errors.append("raw evidence unavailable: " + "; ".join(evidence.get("missing_reasons", [])))
        return {"valid": not errors, "errors": errors, "referenced_files": [], "state_count": 0}
    states = evidence.get("states", {})
    if set(states) != set(STATES):
        errors.append("raw evidence must contain exactly original/zero_step/final")
    for state in STATES:
        try:
            loaded = load_state(raw_root, states[state])
            manifest = loaded["manifest"]
            bindings = manifest["bindings"]
            if manifest["state"] != state:
                raise ValueError("state identity mismatch")
            if any(not bindings.get(key) for key in REQUIRED_BINDINGS):
                raise ValueError("state omits required provenance binding")
            if any(not isinstance(bindings[key], str) or not re.fullmatch(r"[0-9a-f]{64}", bindings[key]) for key in REQUIRED_BINDINGS if key.endswith("sha256")):
                raise ValueError("invalid provenance digest format")
            for key in ("sample_id", "dataset", "split", "model"):
                if bindings[key] != record.get(key):
                    raise ValueError(f"record/state binding mismatch: {key}")
            for key in ("config_id", "source_sha256"):
                if key in record and bindings[key] != record[key]:
                    raise ValueError(f"record/state binding mismatch: {key}")
            for key, value in (expected_bindings or {}).items():
                if bindings.get(key) != value:
                    raise ValueError(f"independent binding mismatch: {key}")
            model_files = bindings.get("model_files", [])
            if not model_files or any(item.get("status") != "verified" or item.get("actual_sha256") != item.get("expected_sha256") or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("actual_sha256", ""))) for item in model_files):
                raise ValueError("model file hashes were not fully verified")
            if not manifest.get("adapter_metadata") or not manifest.get("parameters"):
                raise ValueError("state omits adapter policy or rendering/metric parameters")
            identity = canonical_json({
                "bindings": bindings, "parameters": manifest["parameters"],
                "adapter_metadata": manifest["adapter_metadata"],
                "object_mask": manifest["tensors"]["object_mask"]["tensor_sha256"],
            })
            if shared_identity is None:
                shared_identity = identity
            elif shared_identity != identity:
                raise ValueError("three states disagree on provenance, mask or numeric/render policy")
            files.add(states[state]["path"])
            files.update(reference["path"] for reference in _tensor_references(manifest))
            count += 1
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, zlib.error) as error:
            errors.append(f"{state}: {error}")
    return {"valid": not errors, "errors": errors, "referenced_files": sorted(files), "state_count": count}


def collect_record_files(record: dict[str, Any], raw_root: Path) -> list[Path]:
    report = validate_record_raw(record, raw_root)
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]))
    return [referenced_path(raw_root, relative) for relative in report["referenced_files"]]


def model_binding(model_manifest: Path, model_id: str) -> dict[str, Any]:
    """Bind actual registered weight/source files once per evaluated model."""
    manifest = json.loads(Path(model_manifest).read_text(encoding="utf-8"))
    matches = [item for item in manifest["models"] if item["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"model manifest must identify exactly one {model_id}")
    definition = matches[0]
    files = []
    for item in definition.get("artifacts", []):
        path = Path(item["path"])
        actual = file_sha256(path)
        if actual != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"registered model artifact differs: {path}")
        files.append({"path": str(path), "bytes": path.stat().st_size, "expected_sha256": item["sha256"], "actual_sha256": actual, "status": "verified"})
    if not files:
        raise ValueError(f"no model artifacts registered for {model_id}")
    return {
        "model_manifest_sha256": file_sha256(model_manifest),
        "model_definition_sha256": digest_bytes(canonical_json(definition)),
        "model_files": files,
    }
