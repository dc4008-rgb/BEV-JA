#!/usr/bin/env python3
"""Accept only the observed part1 completed prefix after the preserved BrokenPipe termination."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import stat
import time
import types

REQUEST_SCHEMA = "ours5-ablation-part1-observed-prefix-CPU-request-v1"
REQUEST_FIELDS = {"schema", "old_request", "old_prefix", "old_END", "repaired_contract",
                  "patch_contract_source", "retention_source", "output", "admission"}
CONTRACT_SHA = "6f04352f1c904891619b604517909f721e41d4ed13f0d8a4c3e35301303eab38"
PATCH_SHA = "565066784eaac77f422c7943185221492263cfe590ba51b5642dc9b653c225a5"
RETENTION_SHA = "3aec61b45b4a27a06edf09733bb6fefbd36434bcb7071189e7fb46e44a9cb17f"


OBSERVED_PREFIX = {'ours5_fixed_one_sided': {'completed_cases': 28, 'old_request': [4555, 'c2385fb9ccd73510ca50ac303ffa3bef85ebe671fa5d6a272af7532724b568a3'], 'old_prefix': [169930, '4fd3d591c4ede879d4b732d6dc1cca4e0389f9f3cd81a7c4199f4a7a925cba63'], 'old_END': [444155, 'be568aaf77d8c9eb89d9f5c3b32130b83cddd580a51d50a6c2c2bb0987458200']}, 'ours5_no_conflict_guard': {'completed_cases': 27, 'old_request': [4564, '6e42b9c409fb1c2eed2dbe8a3edaad15dbd5c3b4f53f9d6b843a0353bd4df733'], 'old_prefix': [164021, 'e79f58a9c66269c0ce8c65c16068a30275b51aa7498a50ee70f381f408491672'], 'old_END': [424368, '7ff0a4d7ad446d141f398d058f343c9d03d85cefd8bd1f76cd1b9bd538089e03']}}

def need(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    path = Path(value)
    need(path.is_absolute() and ".." not in path.parts
         and not any(parent.is_symlink() for parent in (path, *path.parents)),
         "literal nonsymlink path required")
    return path


def descriptor(path):
    path = plain(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def bound(binding):
    need(set(binding) == {"path", "bytes", "sha256"}
         and type(binding["bytes"]) is int and 0 < binding["bytes"] <= 8 * 1024 ** 2,
         "closed bounded metadata descriptor required")
    path = plain(binding["path"])
    data = path.read_bytes()
    need(len(data) == binding["bytes"] and hashlib.sha256(data).hexdigest() == binding["sha256"],
         "metadata whole changed")
    return data


def read(binding):
    return json.loads(bound(binding), parse_constant=lambda value: need(False, "nonfinite JSON"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return descriptor(path)


def original_functions(binding, expected_sha, names, namespace, constants=()):
    need(binding["sha256"] == expected_sha, "original function source pin differs")
    tree = ast.parse(bound(binding), filename=binding["path"])
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constants:
                selected.append(node)
    found_functions = {node.name for node in selected if isinstance(node, ast.FunctionDef)}
    found_constants = {node.targets[0].id for node in selected if isinstance(node, ast.Assign)}
    need(found_functions == set(names) and found_constants == set(constants), "exact original functions required")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    selected_tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(selected_tree, binding["path"], "exec"), namespace)
    return namespace


def verify(q, request_descriptor):
    need(set(q) == REQUEST_FIELDS and q["schema"] == REQUEST_SCHEMA, "closed original50 CPU request required")
    admitted = q["admission"]
    need(set(admitted) == {"expected_hostname", "expected_boot_id", "deadline_unix", "wall_seconds",
                           "max_RSS_bytes", "disk_reserve_bytes"}, "closed finite CPU admission required")
    need(socket.gethostname() == admitted["expected_hostname"]
         and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == admitted["expected_boot_id"],
         "current CPU host and boot differ")
    for key in ("deadline_unix", "wall_seconds", "max_RSS_bytes", "disk_reserve_bytes"):
        need(type(admitted[key]) in (int, float) and math.isfinite(admitted[key]) and admitted[key] > 0,
             "finite positive CPU resource bound required")
    remaining = min(admitted["wall_seconds"], admitted["deadline_unix"] - time.time())
    need(0 < remaining <= 3600 and admitted["max_RSS_bytes"] <= 4 * 1024 ** 3
         and admitted["disk_reserve_bytes"] >= 1024 ** 3, "finite original50 CPU profile required")
    output = plain(q["output"])
    need(not output.exists() and output.parent.is_dir(), "new owned CPU output required")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU deadline")))
    signal.setitimer(signal.ITIMER_REAL, remaining)

    def guard():
        need(time.time() < admitted["deadline_unix"], "CPU deadline exceeded")
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= admitted["max_RSS_bytes"],
             "CPU RSS bound exceeded")
        need(shutil.disk_usage(output.parent).free >= admitted["disk_reserve_bytes"], "CPU disk reserve exceeded")

    guard()
    old_q, result, end = (read(q[key]) for key in ("old_request", "old_prefix", "old_END"))
    expected = OBSERVED_PREFIX[old_q["method_id"]]
    for key in ("old_request", "old_prefix", "old_END"):
        need((q[key]["bytes"], q[key]["sha256"]) == tuple(expected[key]),
             "exact observed failed-attempt original required")
    count = expected["completed_cases"]
    need(old_q["part_id"] == 1 and result["request"] == q["old_request"]
         and result["completed_cases"] == count and result["completed_updates"] == count * 125,
         "actual observed part1 completed prefix required")
    need(old_q["expected_hostname"] == admitted["expected_hostname"]
         and old_q["expected_boot_id"] == admitted["expected_boot_id"], "original training node identity required")
    training_root = plain(old_q["output_directory"])
    need(not output.is_relative_to(training_root) and not training_root.is_relative_to(output),
         "CPU output overlaps completed training")
    need(end["actual_wait_return"] == -15
         and end["failure"] == {"type": "BrokenPipeError", "message": "[Errno 32] Broken pipe"}
         and end["only_owned_pidfd_TERM_attempt"]["returned_success"] is True
         and end["only_owned_pidfd_TERM_attempt"]["signal"] == "SIGTERM"
         and end["session_group_and_children_after_wait"]["actual_members"] == []
         and end["session_group_and_children_after_wait"]["root_actual_direct_children"] == [],
         "observed failed leaf wait and empty owned group required")
    argv = end["actual_OS_argv"]
    need(argv[-4] == "--stage-spec" and argv[-2] == "--stage-spec-sha256", "original stage spec argv required")
    spec_binding = descriptor(plain(argv[-3]))
    need(spec_binding["sha256"] == argv[-1], "original stage spec changed")
    spec = read(spec_binding)
    need(spec["purpose"] == "TRAIN_1" and spec["entrypoint"] == old_q["source_files"]["probe"]
         and spec["args"] == ["--request", q["old_request"]["path"], "--request-sha256",
                              q["old_request"]["sha256"], "--execute"], "leaf END does not bind original TRAIN request")
    need(q["repaired_contract"]["sha256"] == CONTRACT_SHA, "reviewed repaired contract required")
    contract = types.ModuleType("_actual_repaired_ablation_contract")
    contract.__file__ = q["repaired_contract"]["path"]
    exec(compile(bound(q["repaired_contract"]), contract.__file__, "exec"), contract.__dict__)
    rows = contract.validate_completed_prefix(old_q, result)
    need(len(rows) == count, "exact observed completed prefix required")
    acceptance = {"method_id": old_q["method_id"], "part_id": 1,
                  "completed_cases": count, "completed_updates": count * 125}
    guard()
    retain = original_functions(q["retention_source"], RETENTION_SHA, ("retain_opaque",),
                                {"plain": plain, "need": need, "hashlib": hashlib, "stat": stat, "os": os})["retain_opaque"]
    import torch
    decoder = original_functions(q["patch_contract_source"], PATCH_SHA,
        ("_tensor_digest", "_validate_rgb", "load_patch"),
        {"torch": torch, "Path": Path, "hashlib": hashlib, "json": json},
        ("CANONICAL_HW", "PATCH_SCHEMA", "RENDERER_CONTRACT"))
    output.mkdir()
    catalogue = {}
    cases = []
    artifact_count = tensor_count = decoded_count = artifact_bytes = 0

    def retained(binding):
        previous = catalogue.get(binding["path"])
        if previous is not None:
            need(all(previous[key] == binding[key] for key in ("path", "bytes", "sha256")),
                 "same path has conflicting retained descriptors")
            return previous
        record = retain(binding, guard)
        catalogue[binding["path"]] = record
        need(len(catalogue) <= 2048, "finite original50 retention count exceeded")
        return record

    try:
        for key in ("old_request", "old_prefix", "old_END", "repaired_contract", "patch_contract_source", "retention_source"):
            retained(q[key])
        retained(spec_binding)
        for row in result["rows"]:
            guard()
            sample = row["sample"]
            train = training_root / "samples" / sample["case_id"] / "train"
            need(plain(row["summary"]["path"]) == train / "summary.json"
                 and plain(row["server_only_inventory"]["path"]) == train / "SERVER_ONLY_ARTIFACT_INDEX.json",
                 "original per-case train and inventory namespace required")
            summary, returned = read(row["summary"]), read(row["returned"])
            need(returned["patch_bindings"] == summary["patch_bindings"], "case RETURN patch bindings differ")
            inventory = read(row["server_only_inventory"])
            files = inventory["files"]
            paths = {item["path"] for item in files}
            need(len(files) == len(paths) == row["artifact_count_before_index"], "actual inventory count differs")
            need(all(plain(item["path"]).is_relative_to(train) for item in files), "inventory escaped own case train directory")
            current_files = set()
            for path in train.rglob("*"):
                info = path.lstat()
                need(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), "ordinary inventory entries required")
                if path.is_file():
                    current_files.add(str(path))
            need(current_files == paths | {row["server_only_inventory"]["path"]}, "complete current inventory path set differs")
            case_tensor_count = sum(Path(item["path"]).suffix == ".pt" for item in files)
            need(case_tensor_count == row["tensor_file_count"], "actual tensor inventory count differs")
            for item in files:
                retained(item)
                artifact_bytes += item["bytes"]
                need(artifact_bytes <= 2 * 1024 ** 3, "finite original50 artifact bytes exceeded")
            artifact_count += len(files)
            tensor_count += case_tensor_count
            for key in ("server_only_inventory", "returned", "generation_observation", "actual_measurement",
                        "actual_total_measurement", "actual_training_plan"):
                retained(row[key])
            source_path = plain(row["actual_source_path"])
            need(row["source_sha256"] == sample["source_sha256"], "original source binding differs")
            retained({"path": str(source_path), "bytes": source_path.stat().st_size,
                      "sha256": sample["source_sha256"]})
            need(summary["trace_sha256"] == row["attack_trace"]["sha256"]
                 and summary["cycle_trace_sha256"] == row["cycle_trace"]["sha256"]
                 and summary["state_evidence_manifest_sha256"] == row["state_manifest"]["sha256"],
                 "original trace and state bindings differ")
            trace = [json.loads(line) for line in bound(row["attack_trace"]).splitlines()]
            need(len(trace) == 125 and [item["global_step"] for item in trace] == list(range(1, 126))
                 and all(item["sample_id"] == sample["sample_id"] and item["case_id"] == sample["case_id"]
                         and item["switches"] == summary["switches"] and item["updates_per_sample"] == 125
                         and item["outer_round"] == (index // 25) + 1 and item["inner_step"] == (index % 25) + 1
                         for index, item in enumerate(trace)), "original125 trace sequence differs")
            manifest = read(row["state_manifest"])
            need(manifest["status"] == "complete" and manifest["executed_updates"] == 125
                 and manifest["sample_id"] == sample["sample_id"]
                 and all(item["recording_preserved_rng"] is True for item in manifest["entries"]),
                 "original state evidence completion differs")
            patches = {}
            for state, name in (("zero", "zero_step_patch.pt"), ("final", "final_patch.pt")):
                binding = dict(summary["patch_bindings"][name], path=str(train / name))
                record = retained(binding)
                payload = torch.load(plain(binding["path"]), map_location="cpu", weights_only=True)
                need(payload["metadata"] == returned["native_metadata_expected"][state], "native metadata differs from actual RETURN")
                patch = decoder["load_patch"](plain(binding["path"]), map_location="cpu")
                need(patch.device.type == "cpu" and patch.dtype == torch.float32 and tuple(patch.shape) == (1, 3, 64, 128),
                     "canonical CPU float32 patch required")
                after = plain(binding["path"]).lstat()
                after9 = [after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_uid,
                          after.st_gid, after.st_size, after.st_mtime_ns, after.st_ctime_ns]
                need(after9 == record["full9"], "native changed during CPU decoding")
                patches[state] = dict(binding, tensor_sha256=decoder["_tensor_digest"](patch))
                decoded_count += 1
                del patch, payload
            cases.append({"sample_id": sample["sample_id"], "patches": patches})
        need(len(cases) == count and decoded_count == count * 2, "all observed prefix patches must decode")
        catalogue_binding = save(output / "PART1_PREFIX_SERVER_RETENTION_CATALOGUE.json", {
            "schema": "ours5-ablation-part1-observed-prefix-server-retention-v1", "old_prefix": q["old_prefix"],
            "files": list(catalogue.values()), "cases": cases,
            "artifact_file_count": artifact_count, "tensor_file_count": tensor_count,
            "artifact_file_bytes": artifact_bytes, "decoded_zero_final_patch_count": decoded_count})
        return save(output / "PART1_PREFIX_CPU_ACCEPTED.json", {
            "schema": "ours5-ablation-part1-observed-prefix-independent-CPU-accepted-v1", "accepted": True,
            "actual_request": request_descriptor, **acceptance, "old_request": q["old_request"],
            "old_prefix": q["old_prefix"], "old_END": q["old_END"],
            "repaired_contract": q["repaired_contract"], "catalogue": catalogue_binding,
            "artifact_file_count": artifact_count, "tensor_file_count": tensor_count,
            "artifact_file_bytes": artifact_bytes, "decoded_zero_final_patch_count": decoded_count,
            "all_retained_file_count": len(catalogue), "actual_new_model_forwards": 0,
            "actual_new_optimizer_updates": 0, "original_leaf_failure_and_owner_clock_preserved": True,
            "old_leaf_wait_return": end["actual_wait_return"], "old_leaf_failure": end["failure"],
            "old_original_deadline_unix": old_q["deadline_unix"], "unfinished_next_case_reused": False,
            "remaining_original_part1_cases": 50 - count})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request_descriptor = descriptor(args.request)
    need(request_descriptor["sha256"] == args.request_sha256, "actual CPU request SHA differs")
    request = read(request_descriptor)
    need(set(request) == REQUEST_FIELDS and request["schema"] == REQUEST_SCHEMA, "closed CPU request required")
    if args.execute:
        print(json.dumps(verify(request, request_descriptor), sort_keys=True))
    else:
        print(json.dumps({"status": "METADATA_ONLY", "request": request_descriptor}))
