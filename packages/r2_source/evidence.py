"""Lossless, typed CPU training-state evidence; no automatic resume machinery."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def safe_snapshot(value):
    """Reject unknown scheduler fields rather than silently dropping state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return {"__numpy_array__": True, "dtype": str(value.dtype), "shape": list(value.shape),
                "bytes": torch.from_numpy(np.frombuffer(value.tobytes(), dtype=np.uint8).copy())}
    if isinstance(value, np.generic):
        return {"__numpy_scalar__": str(value.dtype), "value": value.item()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, (str, int)) for key in value):
            raise TypeError("state mapping keys must be strings or integers")
        return {key: safe_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(safe_snapshot(item) for item in value)
    if isinstance(value, torch.dtype):
        return {"__torch_dtype__": str(value)}
    if isinstance(value, torch.device):
        return {"__torch_device__": str(value)}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise TypeError("unsupported state field type: " + type(value).__module__ + "." + type(value).__qualname__)


def fingerprint(value):
    result = hashlib.sha256()
    def consume(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            result.update(str(("tensor", str(x.dtype), tuple(x.shape))).encode())
            result.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b"dict{")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                consume(key); consume(item[key])
            result.update(b"}")
        elif isinstance(item, (tuple, list)):
            result.update((type(item).__name__ + "[").encode())
            for child in item:
                consume(child)
            result.update(b"]")
        else:
            result.update((type(item).__name__ + ":" + repr(item) + ";").encode())
    consume(value)
    return result.hexdigest()


def rng_snapshot(named_generators=None):
    cuda_initialized = torch.cuda.is_initialized()
    return {"python": safe_snapshot(random.getstate()), "numpy": safe_snapshot(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(), "cuda_initialized": cuda_initialized,
            "torch_cuda": [x.clone().cpu() for x in torch.cuda.get_rng_state_all()] if cuda_initialized else [],
            "named_generators": {name: {"device": str(generator.device), "state": generator.get_state().clone().cpu()}
                                 for name, generator in (named_generators or {}).items()}}


def scheduler_snapshot(scheduler):
    cls = type(scheduler)
    source = inspect.getsourcefile(cls)
    state = safe_snapshot(vars(scheduler))
    if not isinstance(state, dict):
        raise TypeError("scheduler runtime state is not a mapping")
    return {"class_module": cls.__module__, "class_qualname": cls.__qualname__,
            "source_file": source, "source_sha256": sha(source) if source else None,
            "config": safe_snapshot(dict(scheduler.config)), "runtime_fields": state,
            "runtime_field_names": sorted(state), "runtime_sha256": fingerprint(state),
            "complete_instance_dict_saved": True,
            "note": "Config plus all instance fields; reconstruction/automatic resume is not implemented."}


def gradient_stats(tensor, eps=1e-8):
    if tensor is None:
        return {"connected": False, "classification": "disconnected"}
    x = tensor.detach().cpu()
    finite = bool(torch.isfinite(x).all())
    norm32 = float(torch.linalg.vector_norm(x.float())) if finite else None
    norm64 = float(torch.linalg.vector_norm(x.double())) if finite else None
    count = int(torch.count_nonzero(x))
    return {"connected": True, "dtype": str(x.dtype), "shape": list(x.shape), "numel": x.numel(),
            "all_finite": finite, "nonzero_count": count, "all_components_exact_zero": count == 0,
            "l2_float32": norm32, "l2_float64": norm64, "controller_epsilon": eps,
            "cpu_recomputed_classification": "nonfinite" if not finite else "exact_zero" if count == 0 else
                              "nonzero_below_original_epsilon" if norm32 <= eps else "active"}


class CallLedger:
    def __init__(self, planned):
        if any(type(value) is not int or value < 0 for value in planned.values()):
            raise ValueError("budgets require nonnegative integer counts")
        self.planned = dict(planned)
        self.attempted = dict.fromkeys(planned, 0)
        self.completed = dict.fromkeys(planned, 0)
        self.phase = None

    def call(self, name, function, *args, **kwargs):
        if name not in self.planned or self.attempted[name] >= self.planned[name]:
            raise RuntimeError("undeclared or exhausted callback budget: " + name)
        self.attempted[name] += 1
        value = function(*args, **kwargs)
        self.completed[name] += 1
        return value

    def snapshot(self):
        return {"planned": dict(self.planned), "attempted": dict(self.attempted), "completed": dict(self.completed),
                "counter_semantics": "callback counts, not physical kernel counts: attempt before callback; completed only on callback return; transformer callback includes scheduler.step and can fail after a completed transformer forward; failures retain partial counts"}

    def require_complete(self):
        if self.attempted != self.planned or self.completed != self.planned:
            raise RuntimeError("real callback budget did not complete exactly")


class StateRecorder:
    """Raw tensor/tree evidence; scientific identity independent of run location."""
    def __init__(self, output, candidate_id):
        self.output = Path(output)
        self.directory = self.output / "state_evidence"
        self.directory.mkdir(exist_ok=False)
        self.candidate_id = candidate_id
        self.entries = []
        self.updates_applied = 0
        self.ledger_path = self.directory / "ledger.jsonl"
        self.ledger_path.open("x").close()

    def save(self, phase, step, state):
        if type(step) is not int or not 0 <= step <= 100 or not phase.replace("_", "").isalnum():
            raise ValueError("invalid state phase/global step")
        before = rng_snapshot()
        index = len(self.entries) + 1
        path = self.directory / f"{index:04d}_{phase}_step{step:04d}.pt"
        payload = {"schema": "meeting-full-state-v1", "metadata": {
            "candidate_id": self.candidate_id, "global_step": step,
            "phase": phase, "resume_enabled": False}, "state": safe_snapshot(state)}
        with path.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        after = rng_snapshot()
        entry = {"index": index, "phase": phase, "global_step": step,
                 "path": str(path.relative_to(self.output)), "sha256": sha(path), "bytes": path.stat().st_size,
                 "scientific_fingerprint": fingerprint(payload),
                 "recording_preserved_rng": fingerprint(before) == fingerprint(after)}
        with self.ledger_path.open("a") as stream:
            stream.write(canonical_json(entry).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.entries.append(entry)
        if not entry["recording_preserved_rng"]:
            raise RuntimeError("state serialization consumed scientific RNG")
        return {"index": index, "phase": phase, "global_step": step,
                "path": entry["path"], "scientific_fingerprint": entry["scientific_fingerprint"]}

    def finish(self, status, executed_updates):
        value = {"schema": "meeting-full-state-manifest-v1", "candidate_id": self.candidate_id,
                 "status": status, "executed_updates": executed_updates, "resume_enabled": False,
                 "entries": self.entries, "ledger_sha256": sha(self.ledger_path)}
        path = self.directory / "MANIFEST.json"
        with path.open("x") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return sha(path)
