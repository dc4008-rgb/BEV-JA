#!/usr/bin/env python3
"""One new, root-admitted TWO-CASE SD3.5 connectivity prefix; zero attacks.

No SSH, downloads, process classifier, power action, retry, or victim forward is
implemented here. The root operator must admit and bound this owned child with
the current strict process/resource/lifecycle controls before --execute. Merely
preparing this source or validating request metadata is not such admission.

The original SCI1940 loader is used ONLY for frozen input/context/crop helpers
and source/asset verification. Its VAE-only pipeline/denoising helpers are never
called. This version explicitly retains original two-CLIP/no-T5 conditioning;
it does not complete the separate sealed001 three-encoder contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
import traceback
import uuid as uuid_module
from types import ModuleType

IDENTITY = "bevja_cg01_formal200_budget135_v1"
CONDITIONING_MODE = "two_clip_no_t5_explicit"
REQUEST_SCHEMA = "conditional-connectivity-two-clip-actual-request-v1"
PROTOCOL_SHA = "438aa6392e5c1253ad304cb5a188b71c3b608f7a41813bd42318a8dcd5f58d05"
PLAN_SHA = "661b32e3f309a2a5924edd270f31f79bbe9c4e25c044aac97b3f77ecb9c4eb1b"
SCI_SHA = "1940406257965a181371af89634a7378433392147719a4b16ae28d7f33c9081d"
KERNEL_SHA = "946c415b14c8119e70b389f3ec3c03444e51dd854742878cd0cf191495c263cb"
SNAPSHOT_INVENTORY_SHA = "0c21d98782b4a88f524b9377d779d7143d51fea72f390f068f479caeed36b1dc"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODULES = ("text_encoder", "text_encoder_2", "transformer", "vae")
TOKENIZERS = ("tokenizer", "tokenizer_2")
HELPERS = ("child", "r2_source", "v2_source", "clip_source")
OUTPUT_NAMESPACE = "conditional_generation_supplement_20261028_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute() and ".." not in Path(value).parts,
            "explicit absolute path without traversal required")
    return Path(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    path = Path(path)
    return {"path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": sha(path)}


def validate_descriptor(d):
    require(isinstance(d, dict) and set(d) == {"path", "bytes", "sha256"}, "closed whole-file descriptor required")
    absolute(d["path"])
    require(type(d["bytes"]) is int and d["bytes"] > 0, "nonempty file descriptor required")
    require(isinstance(d["sha256"], str) and re.fullmatch("[0-9a-f]{64}", d["sha256"]), "whole SHA256 required")


def checked_file(d):
    validate_descriptor(d)
    p = absolute(d["path"])
    require(not p.is_symlink() and p.is_file(), "regular nonsymlink control/source file required")
    before = p.stat()
    data = p.read_bytes()
    with p.open("rb") as f:
        os.fsync(f.fileno())
    again = p.read_bytes()
    after = p.stat()
    require(data == again and len(data) == d["bytes"] and hashlib.sha256(data).hexdigest() == d["sha256"],
            "bound file bytes changed")
    require(all(getattr(before, k) == getattr(after, k) for k in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "control file unstable")
    return data


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return descriptor(path)


def load_bound_source(name, d):
    """Compile the exact verified bytes, without a sys.path/pyc substitute."""
    require(name not in sys.modules, "bound module name already exists")
    data = checked_file(d)
    module = ModuleType(name)
    module.__file__ = d["path"]
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(data, d["path"], "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def validate_request(q):
    keys = {"schema", "identity", "expected_hostname", "expected_boot_id", "deadline_unix",
            "protocol", "reference_plan", "science_package", "helper_roots", "snapshot",
            "source_mapping", "weights_root", "output_base", "output_directory",
            "operator_admission", "source_files"}
    require(isinstance(q, dict) and set(q) == keys, "closed actual connectivity request required")
    require(q["schema"] == REQUEST_SCHEMA and q["identity"] == IDENTITY, "wrong experiment request")
    require(isinstance(q["expected_hostname"], str) and re.fullmatch(r"autodl-container-[a-z0-9-]+", q["expected_hostname"]),
            "actual complete hostname required")
    require(isinstance(q["expected_boot_id"], str) and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", q["expected_boot_id"]),
            "actual complete boot UUID required")
    require(type(q["deadline_unix"]) in (int, float) and math.isfinite(q["deadline_unix"]), "finite absolute deadline required")
    for key in ("protocol", "reference_plan", "operator_admission"):
        validate_descriptor(q[key])
    require(q["protocol"]["sha256"] == PROTOCOL_SHA and q["reference_plan"]["sha256"] == PLAN_SHA,
            "frozen protocol/input reference mismatch")
    require(set(q["science_package"]) == {"root", "manifest"}, "science package descriptor required")
    absolute(q["science_package"]["root"])
    validate_descriptor(q["science_package"]["manifest"])
    require(q["science_package"]["manifest"]["sha256"] == SCI_SHA, "original SCI1940 required")
    require(set(q["helper_roots"]) == set(HELPERS), "all four frozen source roots required")
    for root in q["helper_roots"].values():
        absolute(root)
    require(set(q["snapshot"]) == {"root", "inventory"}, "actual full SD3.5 snapshot descriptor required")
    absolute(q["snapshot"]["root"])
    validate_descriptor(q["snapshot"]["inventory"])
    require(q["snapshot"]["inventory"]["sha256"] == SNAPSHOT_INVENTORY_SHA, "original two-CLIP snapshot inventory required")
    require(q["source_mapping"] is None or (isinstance(q["source_mapping"], dict) and
            set(q["source_mapping"]) == {"prefix", "root"}), "paired source remapping only")
    if q["source_mapping"] is not None:
        for value in q["source_mapping"].values():
            absolute(value)
    if q["weights_root"] is not None:
        absolute(q["weights_root"])
    base, output = absolute(q["output_base"]), absolute(q["output_directory"])
    require(OUTPUT_NAMESPACE in base.parts and output.parent == base and
            re.fullmatch(r"connectivity_[A-Za-z0-9_-]+", output.name),
            "a new direct connectivity child under the new output_base is required")
    require(set(q["source_files"]) == {"probe", "kernel"}, "both new source files must be bound")
    for d in q["source_files"].values():
        validate_descriptor(d)
    require(q["source_files"]["kernel"]["sha256"] == KERNEL_SHA, "unreviewed kernel source")
    return q


def validate_inventory_metadata(document):
    """Original inventory rehashed whole-byte for this declared no-T5 version."""
    require(document.get("model_id") == MODEL_ID, "the declared SD3.5-medium identity is required")
    entries = document.get("files")
    require(isinstance(entries, list) and entries, "nonempty complete snapshot inventory required")
    names = []
    for entry in entries:
        require(set(entry) == {"path", "bytes", "sha256"}, "closed snapshot member binding required")
        p = Path(entry["path"])
        require(isinstance(entry["path"], str) and p.parts and not p.is_absolute() and ".." not in p.parts,
                "safe snapshot-relative member required")
        validate_descriptor({**entry, "path": "/" + entry["path"]})
        names.append(entry["path"])
    require(len(names) == len(set(names)), "duplicate snapshot member")
    require("model_index.json" in names and "scheduler/scheduler_config.json" in names, "pipeline/scheduler metadata missing")
    for module in MODULES:
        require(module + "/config.json" in names, "full component config missing: " + module)
        require(any(n.startswith(module + "/") and n.endswith(".safetensors") for n in names),
                "safetensors weights missing: " + module)
    for tokenizer in TOKENIZERS:
        require(tokenizer + "/tokenizer_config.json" in names, "tokenizer config missing: " + tokenizer)
        require(any(n.startswith(tokenizer + "/") and n.rsplit("/", 1)[-1] in
                    ("tokenizer.json", "vocab.json", "spiece.model") for n in names), "tokenizer vocabulary missing: " + tokenizer)
    return entries


def check_snapshot(q):
    root = absolute(q["root"])
    require(root.is_dir() and not root.is_symlink(), "actual full snapshot directory required")
    document = json.loads(checked_file(q["inventory"]))
    entries = validate_inventory_metadata(document)
    try:
        os.lstat(root / "text_encoder_3")
    except FileNotFoundError as error:
        require(error.errno == 2, "unexpected missing-component error")
    else:
        raise ValueError("this actual no-T5 snapshot binding requires the observed text_encoder_3 path to remain ENOENT")
    names = {e["path"] for e in entries}
    actual = set()
    for path in root.rglob("*"):
        require(not (path.is_symlink() and path.is_dir()), "snapshot directory symlinks forbidden")
        if path.is_file():
            actual.add(str(path.relative_to(root)))
    require(names == actual, "snapshot file set differs from the actual inventory")
    for e in entries:
        path = root / e["path"]
        before = path.stat()
        require(before.st_size == e["bytes"] and sha(path) == e["sha256"], "snapshot whole bytes mismatch: " + e["path"])
        after = path.stat()
        require(all(getattr(before, k) == getattr(after, k) for k in
                    ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "snapshot member unstable")
    index = json.loads((root / "model_index.json").read_text())
    require(index.get("_class_name") == "StableDiffusion3Pipeline", "official SD3 pipeline index required")
    for name in (*MODULES, *TOKENIZERS):
        item = index.get(name)
        require(isinstance(item, list) and len(item) == 2 and item[0] in ("transformers", "diffusers")
                and isinstance(item[1], str), "nonempty official component declaration required: " + name)
    return {"root": str(root), "inventory": q["inventory"], "file_count": len(entries),
            "bytes": sum(e["bytes"] for e in entries), "declared_two_CLIP_inventory_whole_bytes_verified": True,
            "conditioning_mode": CONDITIONING_MODE,
            "actual_text_encoder_3_path_ENOENT": True,
            "not_a_three_encoder_001_acceptance": True,
            "file_symlink_policy": "HF cache file links allowed only with every resolved target whole-byte binding; directory links refused"}


def time_and_identity(q):
    require(socket.gethostname() == q["expected_hostname"], "fresh hostname differs")
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == q["expected_boot_id"], "fresh boot differs")
    require(0 < q["deadline_unix"] - time.time() <= 1800, "expired or unbounded operator deadline")


def observe_legacy_ledger_source(kernels):
    """Record source identities without constructing a legacy call ledger.

    SCI load_verified_helpers returns a TrainingKernels dataclass instance, not
    a module. Observe its actual type and the contained ledger class separately.
    """
    container_class = type(kernels)
    container_source = inspect.getsourcefile(container_class)
    ledger_class = getattr(kernels, "CallLedger", None)
    ledger_source = inspect.getsourcefile(ledger_class) if isinstance(ledger_class, type) else None
    return {"container_type": container_class.__module__ + "." + container_class.__qualname__,
        "container_source_file": descriptor(container_source) if container_source else None,
        "CallLedger_class_observed": isinstance(ledger_class, type),
        "CallLedger_qualified_name": (ledger_class.__module__ + "." + ledger_class.__qualname__)
            if isinstance(ledger_class, type) else None,
        "CallLedger_source_file": descriptor(ledger_source) if ledger_source else None,
        "ledger_instantiated": False, "ledger_counter_keys_observed": None,
        "counter_semantics_observed": None, "attack_or_victim_executed_for_this_observation": False,
        "scope": "actual verified helper source identity only, not a runtime ledger ABI observation"}


def load_science(q, plan):
    root = absolute(q["science_package"]["root"])
    manifest = json.loads(checked_file(q["science_package"]["manifest"]))
    require(absolute(q["science_package"]["manifest"]["path"]) == root / "PACKAGE_MANIFEST.json", "science root/manifest mismatch")
    for e in manifest["files"]:
        p = Path(e["path"])
        require(not p.is_absolute() and ".." not in p.parts, "unsafe science member")
        checked_file({"path": str(root / p), "bytes": e["bytes"], "sha256": e["sha256"]})
    sys.path.insert(0, str(root))
    # A fresh child cannot silently reuse modules from another scientific tree.
    for name in ("benchmark_mode", "source_bindings", "sample_contract", "geometry_eot"):
        require(name not in sys.modules, "scientific module collision: " + name)
    import benchmark_mode
    benchmark_mode.configure("C1")
    import source_bindings
    paths = source_bindings.verify_sources(plan, **{k: Path(v) for k, v in q["helper_roots"].items()})
    import sample_contract
    sample_contract.validate_plan(plan)
    spec = importlib.util.spec_from_file_location("_conditional_original_input_helpers", root / "sample_training.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    inputs, parent, clip, unused_old_kernels = module.load_verified_helpers(paths)
    paths["legacy_ledger_source_observation"] = observe_legacy_ledger_source(unused_old_kernels)
    return inputs, parent, clip, paths, sample_contract


class GPUIdentityBindingError(ValueError):
    """A pre-model identity failure with actual narrow CUDA observations."""
    def __init__(self, message, observation):
        super().__init__(message)
        self.observation = observation


def validate_operator_GPU_query(operator_ready):
    """Check the separate, whole-byte-bound root proof's selected XML identity.

    This is not the root process/resource/strict gate and does not claim an
    allocation from host-wide NVIDIA proc rows. No device selection occurs here.
    """
    require(isinstance(operator_ready, dict) and
            operator_ready.get("schema") == "cg01-budget200-root-ready-proof-v1" and
            operator_ready.get("identity") == IDENTITY,
            "whole-bound root ready proof has wrong schema or identity")
    query = operator_ready.get("selected_GPU_query")
    require(isinstance(query, dict), "root ready proof lacks selected GPU XML query")
    selected = query.get("UUID")
    require(isinstance(selected, str) and re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", selected),
        "root ready proof lacks one complete observed GPU UUID")
    require(isinstance(query.get("selection_mode"), str) and query["selection_mode"],
            "root GPU query selection mode must remain explicit")
    for key in ("observed_CUDA_VISIBLE_DEVICES", "observed_NVIDIA_VISIBLE_DEVICES"):
        require(key in query and (query[key] is None or isinstance(query[key], str)),
                "root ready proof must preserve original narrow environment: " + key)
    return query


def cuda_identity_before_model(torch, operator_ready):
    """Compare actual CUDA0 to the root's actual single NVML XML query.

    Inherit both visibility variables unchanged. The root XML selector is only a
    candidate until this UUID comparison succeeds. Absent/unparseable properties
    and mismatch fail before from_pretrained; never change devices or retry.
    """
    query = validate_operator_GPU_query(operator_ready)
    observed = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "root_selected_GPU_UUID": query["UUID"],
        "root_query_selection_mode": query["selection_mode"],
        "cuda_index": 0, "cuda_device_count": None,
        "cuda_uuid_property": {"observed": False, "reason": "not queried yet"},
        "canonical_CUDA_GPU_UUID": None, "CUDA0_matches_root_XML_UUID": False,
        "model_load_attempted_at_this_gate": False,
        "visibility_environment_modified_by_probe": False,
        "not_hostproc_allocation_or_process_resource_admission": True}

    def bind(ok, message):
        if not ok:
            raise GPUIdentityBindingError(message, observed)

    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        bind(observed[name] == query["observed_" + name],
             "inherited visibility environment differs from actual root query: " + name)
    observed["cuda_device_count"] = torch.cuda.device_count()
    bind(observed["cuda_device_count"] == 1,
         "actual CUDA view must contain exactly one device before model load")
    props = torch.cuda.get_device_properties(0)
    observed["cuda_device_name"] = props.name
    observed["cuda_total_memory_bytes"] = props.total_memory
    try:
        raw_uuid = props.uuid
    except AttributeError:
        observed["cuda_uuid_property"] = {"observed": False,
            "reason": "CUDA property uuid attribute absent"}
        raise GPUIdentityBindingError("CUDA0 UUID property is unobserved; zero model loads", observed)
    observed["cuda_uuid_property"] = {"observed": True,
        "python_type": type(raw_uuid).__module__ + "." + type(raw_uuid).__qualname__,
        "repr": repr(raw_uuid),
        "string_value": raw_uuid if isinstance(raw_uuid, str) else None,
        "bytes_hex": bytes(raw_uuid).hex() if isinstance(raw_uuid, (bytes, bytearray)) else None}
    # Only explicit UUID representations of the actual API value are accepted.
    # Adding the NVML GPU- label is formatting, not choosing or inventing a UUID.
    if isinstance(raw_uuid, str):
        value = raw_uuid[4:] if raw_uuid.startswith("GPU-") else raw_uuid
        bind(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value) is not None,
             "actual CUDA0 UUID string is not a comparable complete UUID")
        cuda_uuid = uuid_module.UUID(value)
        observed["UUID_representation_rule"] = "actual canonical UUID string, optional NVML GPU- label"
    elif isinstance(raw_uuid, (bytes, bytearray)):
        bind(len(raw_uuid) == 16, "actual CUDA0 UUID byte property must contain exactly 16 bytes")
        cuda_uuid = uuid_module.UUID(bytes=bytes(raw_uuid))
        observed["UUID_representation_rule"] = "actual 16 UUID bytes in UUID network byte order"
    elif type(raw_uuid) is getattr(getattr(torch, "_C", None), "_CUuuid", None):
        # Official PyTorch v2.8.0 Module.cpp exposes this exact native type.
        # Its .bytes is vector<uint8_t> in UUID order and __str__ formats those
        # same16 bytes as 8-4-4-4-12. Never parse repr or duck-type unknown types.
        try:
            native_bytes = raw_uuid.bytes
        except AttributeError:
            observed["cuda_uuid_property"]["native_CUuuid_bytes_property_observed"] = False
            raise GPUIdentityBindingError("exact PyTorch CUDA UUID lacks its native bytes property; zero model loads", observed)
        native_string = str(raw_uuid)
        observed["cuda_uuid_property"].update({
            "native_CUuuid_exact_type_observed": True,
            "native_CUuuid_bytes_property_observed": True,
            "native_CUuuid_bytes_property_type": type(native_bytes).__module__ + "." + type(native_bytes).__qualname__,
            "native_CUuuid_bytes_property_repr": repr(native_bytes),
            "native_CUuuid_string_value": native_string})
        bind(type(native_bytes) is list and len(native_bytes) == 16 and
             all(type(value) is int and 0 <= value <= 255 for value in native_bytes),
             "actual PyTorch CUDA UUID bytes must be16 typed uint8 integer values")
        observed["cuda_uuid_property"]["native_CUuuid_bytes_list"] = list(native_bytes)
        observed["cuda_uuid_property"]["native_CUuuid_bytes_hex"] = bytes(native_bytes).hex()
        string_value = native_string[4:] if native_string.startswith("GPU-") else native_string
        bind(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", string_value) is not None,
             "actual PyTorch CUDA UUID string must be one complete canonical UUID")
        cuda_uuid = uuid_module.UUID(bytes=bytes(native_bytes))
        string_uuid = uuid_module.UUID(string_value)
        observed["cuda_uuid_property"]["native_CUuuid_bytes_and_string_same_UUID"] = cuda_uuid == string_uuid
        bind(cuda_uuid == string_uuid, "actual PyTorch CUDA UUID bytes and string differ; zero model loads")
        observed["UUID_representation_rule"] = "exact torch._C._CUuuid actual16 bytes, independently equal to its actual canonical string"
    elif isinstance(raw_uuid, uuid_module.UUID):
        cuda_uuid = raw_uuid
        observed["UUID_representation_rule"] = "actual stdlib UUID value"
    else:
        raise GPUIdentityBindingError("actual CUDA0 UUID property type is not comparable; zero model loads", observed)
    observed["canonical_CUDA_GPU_UUID"] = "GPU-" + str(cuda_uuid)
    root_uuid = uuid_module.UUID(query["UUID"][4:])
    observed["CUDA0_matches_root_XML_UUID"] = cuda_uuid == root_uuid
    bind(observed["CUDA0_matches_root_XML_UUID"],
         "actual CUDA0 UUID differs from actual root NVML XML UUID; zero model loads")
    return observed


def run(q, request_d):
    """Requires an external real strict/lifecycle gate; never claims that gate."""
    time_and_identity(q)
    protocol = json.loads(checked_file(q["protocol"]))
    plan = json.loads(checked_file(q["reference_plan"]))
    operator_ready = json.loads(checked_file(q["operator_admission"]))
    validate_operator_GPU_query(operator_ready)
    require(protocol["identity"] == IDENTITY and protocol["conditioning_mode"] == CONDITIONING_MODE, "explicit two-CLIP protocol required")
    require(protocol["cases"] == plan["samples"] and len(plan["samples"]) == 2,
            "exact two frozen engineering samples required")
    require([s["dataset"] for s in plan["samples"]] == ["kitti", "nuscenes"] and
            all(s["source_split"] == "train" for s in plan["samples"]), "fixed two TRAIN domains only")
    for name, path in (("probe", Path(__file__)), ("kernel", Path(__file__).with_name("sd35_conditional_two_clip_v1.py"))):
        require(absolute(q["source_files"][name]["path"]).resolve() == path.resolve(), "new source path mismatch")
        checked_file(q["source_files"][name])
    # Do this before legacy helpers add their import directories. Whole-byte
    # verification must bind the code actually executed, not only a nearby file.
    kernel_source = load_bound_source("_conditional_connectivity_bound_two_clip_kernel", q["source_files"]["kernel"])
    snapshot = check_snapshot(q["snapshot"])
    inputs, parent, clip, paths, contract = load_science(q, plan)
    import torch
    import torchvision
    import diffusers
    import transformers
    require(torch.cuda.is_available(), "this actual connectivity check requires CUDA")
    require((torch.__version__.split("+")[0], torchvision.__version__.split("+")[0], diffusers.__version__.split("+")[0])
            == ("2.8.0", "0.23.0", "0.38.0"), "frozen runtime versions required")
    clip.configure_determinism()
    pr = paths["child"] / "parent_v17"
    mapping = q["source_mapping"]
    records = parent.training_records(pr / "01_公共数据集/runtime_v1/train_validation_manifest.csv",
        Path(mapping["prefix"]) if mapping else None, Path(mapping["root"]) if mapping else None)
    selected = [contract.record_for_sample(plan, sample, records) for sample in plan["samples"]]
    for sample, record in zip(plan["samples"], selected):
        require(sha(record.source_path) == sample["source_sha256"], "actual source image differs from frozen case")
    car_file, mask_file = (pr / "01_公共数据集" / n for n in ("car_00.jpg", "car_00_mask.jpg"))
    model_manifest = pr / "02_模型/model_manifest.json"
    assets, depth_assets, ready = clip.preflight_training_assets(parent, selected, model_manifest,
        car_file, mask_file, Path(q["weights_root"]) if q["weights_root"] else None, 1.0)
    require(ready, "frozen source/car/four-surrogate weight assets unavailable; no victim is loaded here")
    output, base = absolute(q["output_directory"]), absolute(q["output_base"])
    for path in (output, *output.parents):
        require(not path.is_symlink(), "output symlink forbidden")
    require(base.is_dir() and not output.exists(), "new exclusive output under existing new base required")
    old_roots = [Path(v).resolve() for v in q["helper_roots"].values()] + [paths["child"], Path(q["snapshot"]["root"]).resolve(), Path(q["science_package"]["root"]).resolve()]
    require(not any(output.resolve().is_relative_to(p) for p in old_roots), "new output cannot be inside an original input/source tree")
    time_and_identity(q)
    output.mkdir()
    started = time.monotonic()
    write_new(output / "INTENT.json", {"identity": IDENTITY, "request": request_d,
        "scope": "connectivity only; one generated step each / zero attack / zero victims / zero VJP",
        "external_operator_admission_bound_not_revalidated_by_probe": q["operator_admission"],
        "hostname": socket.gethostname(), "boot_id": q["expected_boot_id"], "pid": os.getpid(),
        "deadline_unix": q["deadline_unix"]})
    kernel = None
    gpu_identity = None
    try:
        from diffusers import StableDiffusion3Pipeline
        gpu_identity = cuda_identity_before_model(torch, operator_ready)
        ConditionalConfig = kernel_source.ConditionalTwoClipConfig
        SD35ConditionalKernel = kernel_source.SD35ConditionalTwoClipKernel
        embedding_descriptor = kernel_source.embedding_descriptor
        device, dtype = torch.device("cuda:0"), torch.bfloat16
        torch.cuda.reset_peak_memory_stats(device)
        # Predeclared official no-T5 mode, never an automatic runtime fallback.
        pipeline = StableDiffusion3Pipeline.from_pretrained(Path(q["snapshot"]["root"]),
            torch_dtype=dtype, local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True,
            text_encoder_3=None, tokenizer_3=None).to(device)
        car, mask = parent.load_car(car_file, mask_file, device)
        rows = []
        for sample, record in zip(plan["samples"], selected):
            time_and_identity(q)
            torch.manual_seed(20260919)
            torch.cuda.manual_seed_all(20260919)
            context = parent.build_context(record, car, mask, device, set())
            initial_small = parent.extract_road_patch(context.background, context.road_quad)
            require(tuple(initial_small.shape) == (1, 3, 64, 128), "original canonical crop shape differs")
            initial = torch.nn.functional.interpolate(initial_small, size=(512, 512), mode="bilinear", align_corners=False)
            kernel = SD35ConditionalKernel(pipeline, ConditionalConfig())
            condition = kernel.encode_condition(device)
            require(condition["conditioning_mode"] == "two_clip_no_t5" and condition["T5_enabled"] is False
                    and condition["T5_execution"] == "not_executed"
                    and condition["T5_adaptation"] == "official_zero_T5_sequence_adaptation",
                    "new kernel must expose the explicitly declared no-T5 mode")
            latent, timesteps = kernel.start_round(initial, 0)
            leaf = kernel.generated_leaf(latent, timesteps[0])
            decoded = kernel.decode_leaf(leaf)
            require(leaf.is_leaf and leaf.requires_grad and leaf.grad_fn is None and
                    decoded.requires_grad and decoded.grad_fn is not None, "generation/decoder graph boundary differs")
            observation = kernel.observation()
            calls = observation["calls"]
            for key in ("encode_condition_completed", "vae_encode_completed", "vae_decode_completed",
                        "transformer_completed", "scheduler_step_completed", "noise_draws_completed", "round_starts_completed"):
                require(calls[key] == 1, "connectivity actual completed call count differs: " + key)
            tensor_path = output / (sample["case_id"] + ".SERVER_ONLY_TENSORS.pt")
            tensors = {"canonical_road_crop": initial_small.detach().cpu(), "raw512_initial_crop": initial.detach().cpu(),
                       "noised_latent": latent.detach().cpu(), "generated_latent": leaf.detach().cpu(),
                       "decoded_one_step": decoded.detach().cpu()}
            with tensor_path.open("xb") as f:
                torch.save(tensors, f); f.flush(); os.fsync(f.fileno())
            row = {"sample": sample, "actual_source_path": str(record.source_path),
                "actual_source_sha256": sha(record.source_path), "condition": condition,
                "observation": observation, "tensor_descriptors": {k: embedding_descriptor(v) for k, v in tensors.items()},
                "server_only_tensor_file": descriptor(tensor_path), "generated_steps": 1,
                "attack_updates": 0, "victim_forward_calls": 0, "VJP_calls": 0,
                "graph_metadata_is_not_backward_execution": True}
            # Preserve the first real prefix even if the second case later fails.
            case_result = write_new(output / (sample["case_id"] + ".CONNECTIVITY.json"), row)
            rows.append({**row, "case_result": case_result})
            kernel.close(); kernel = None
            del leaf, decoded, tensors, latent, initial, initial_small, context
        torch.cuda.synchronize(device)
        time_and_identity(q)
        result = {"schema": "conditional-two-clip-two-case-connectivity-actual-result-v1", "identity": IDENTITY,
            "status": "CONNECTIVITY_TWO_CLIP_PREFIX_EXECUTED_NOT_25_OR_500_ACCEPTANCE", "request": request_d,
            "conditioning_mode": CONDITIONING_MODE, "T5_encoder_or_tokenizer_loaded": False,
            "not_a_three_encoder_001_acceptance": True,
            "root_operator_gate_not_reimplemented_by_probe": True,
            "full_snapshot": snapshot, "source_package_verification": paths["verified"],
            "legacy_ledger_source_observation": paths["legacy_ledger_source_observation"],
            "source_assets": assets, "depth_weight_assets": depth_assets,
            "runtime": {"hostname": socket.gethostname(), "boot_id": q["expected_boot_id"], "pid": os.getpid(),
                "torch": torch.__version__, "torchvision": torchvision.__version__, "diffusers": diffusers.__version__,
                "transformers": transformers.__version__, "cuda_device_name": torch.cuda.get_device_name(device),
                "actual_CUDA_identity_before_model_load": gpu_identity,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "ru_maxrss_Linux_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "elapsed_seconds": time.monotonic() - started}, "rows": rows,
            "total_generated_steps": 2, "total_attack_updates": 0, "total_VJP_calls": 0,
            "four_surrogate_gradient_integration_verified": False, "scientific_or_full_engineering_acceptance": False,
            "not_heldout_not_formal200": True, "test_data_opened": False,
            "server_only_tensors_not_for_LOCAL_or_USB_export": True}
        result_d = write_new(output / "CONNECTIVITY_RESULT.json", result)
        print(json.dumps({"result": result_d, "status": result["status"], "total_attack_updates": 0}, ensure_ascii=False))
    except BaseException as error:
        if isinstance(error, GPUIdentityBindingError):
            gpu_identity = error.observation
        write_new(output / "FAILURE.json", {"identity": IDENTITY, "request": request_d,
            "error": repr(error), "traceback": traceback.format_exc(), "no_retry": True,
            "elapsed_seconds": time.monotonic() - started,
            "actual_CUDA_identity_before_model_load": gpu_identity,
            "partial_kernel_observation": kernel.observation() if kernel is not None else None})
        raise
    finally:
        if kernel is not None:
            kernel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true", help="requires root's real current strict admission and bounded owned lifecycle")
    args = parser.parse_args()
    require(re.fullmatch("[0-9a-f]{64}", args.request_sha256), "explicit request wholeSHA required")
    request_d = descriptor(args.request)
    require(request_d["sha256"] == args.request_sha256, "actual request SHA differs")
    q = validate_request(json.loads(checked_file(request_d)))
    if not args.execute:
        print(json.dumps({"status": "REQUEST_METADATA_VALID_ONLY_NO_MODEL_NO_ADMISSION", "request": request_d,
            "actual_GPU_or_root_strict_admission": False}))
        return
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True
    run(q, request_d)


if __name__ == "__main__":
    main()
