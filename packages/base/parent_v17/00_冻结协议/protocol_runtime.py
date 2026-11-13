"""Fail-closed runtime for the normalized-v9 pre-formal candidate."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_ID = "normalized-v9-20260916"
RUN_ID = "preformal_v9_20260916"
FORMAL_PROTOCOL_ID = "normalized-v13-20260917"
FORMAL_RUN_ID = "formal_v13_20260917"
MODEL_IDS = (
    "fasterrcnn", "ssd300", "monodepth2", "dpt_large",
    "maskrcnn", "fcos", "depth_anything_v1", "zoedepth",
)
CORE_MODEL_IDS = MODEL_IDS[:4]
TRANSFER_MODEL_IDS = MODEL_IDS[4:]
REQUIRED_GATES = (
    "document_contract", "model_identity", "remote_inputs", "new_clean_baseline",
    "method_contracts", "metric_unit_tests", "all_config_smokes",
    "validation_three_state_raw", "resume_integrity", "resource_budget",
    "delivery_recompute",
)
VGG19_SHA256 = "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
FORMAL_CONFIG_IDS = (
    "upc__single_fasterrcnn",
    "das__single_fasterrcnn",
    "das__single_ssd300",
    "das__ensemble_det2",
    "tt3d_detector_adaptation__single_fasterrcnn",
    "tt3d_detector_adaptation__single_ssd300",
    "tt3d_detector_adaptation__ensemble_det2",
    "saam__single_monodepth2",
    "saam__single_dpt_large",
    "saam__ensemble_depth2",
    "advrm__single_monodepth2",
    "advrm__single_dpt_large",
    "advrm__ensemble_depth2",
    "bittaa__pair_fasterrcnn_monodepth2",
    "bittaa__pair_fasterrcnn_dpt_large",
    "bittaa__pair_ssd300_monodepth2",
    "bittaa__pair_ssd300_dpt_large",
    "bittaa__ensemble_all4",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True)
class ProtocolContext:
    root: Path
    protocol_dir: Path
    protocol_path: Path
    protocol: dict[str, Any]
    matrix: list[dict[str, str]]

    def path(self, relative: str | Path) -> Path:
        value = Path(relative)
        resolved = (self.root / value).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"candidate path escapes root: {relative}")
        return resolved

    def row(self, config_id: str) -> dict[str, str]:
        selected = [row for row in self.matrix if row.get("config_id") == config_id]
        if len(selected) != 1:
            raise ValueError(f"configuration must occur exactly once: {config_id}")
        return selected[0]

    @property
    def test_rows(self) -> list[dict[str, str]]:
        return read_csv(self.path(self.protocol["test_manifest_path"]))

    @property
    def training_rows(self) -> list[dict[str, str]]:
        return read_csv(self.path(self.protocol["training_manifest_path"]))

    @property
    def evaluation_model_ids(self) -> tuple[str, ...]:
        return tuple(self.protocol["evaluation_model_ids"])

    @property
    def expected_config_ids(self) -> tuple[str, ...]:
        return tuple(row["config_id"] for row in self.matrix)

    @property
    def num_steps(self) -> int:
        """Return the project unified-adaptation budget.

        This compatibility property is intentionally limited to the v9
        unified 2D-adaptation matrix.  Paper-anchor runners must use
        ``paper_anchor_steps`` so that AdvRM's model-dependent 500/1000
        schedule cannot be collapsed into the project budget.
        """
        budget = self.protocol["budget"]
        return int(budget["unified_2d_adaptation"]["updates"])

    def paper_anchor_steps(self, method_id: str, model_id: str) -> int:
        budget = self.protocol["budget"]["paper_anchor"][method_id]
        if budget.get("status") != "published":
            raise ValueError(f"paper budget is not disclosed/frozen: {method_id}")
        per_model = budget.get("updates_by_model", {})
        if model_id not in per_model:
            raise ValueError(f"paper budget is not mapped: {method_id}/{model_id}")
        return int(per_model[model_id])

    @property
    def run_id(self) -> str:
        return self.protocol["run_id"]


def load_protocol(root: str | Path, protocol_dir: str | Path | None = None) -> ProtocolContext:
    root = Path(root).resolve()
    directory = (root / (Path(protocol_dir) if protocol_dir else Path("00_冻结协议"))).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("protocol directory must be inside candidate root")
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    matrix_path = root / protocol["run_matrix_path"]
    return ProtocolContext(root, directory, protocol_path, protocol, read_csv(matrix_path))


load_protocol_context = load_protocol


def candidate_fingerprint(protocol: dict[str, Any]) -> str:
    if protocol.get("protocol_id") == FORMAL_PROTOCOL_ID:
        keys = (
            "protocol_id", "run_id", "run_scope", "evaluation_model_ids",
            "core_models", "transfer_models", "budget", "metrics",
            "formal_config_ids", "formal_input_sha256", "formal_source",
            "test_use_policy", "storage_policy", "bindings",
        )
        return canonical_hash({key: protocol[key] for key in keys})
    keys = (
        "protocol_id", "run_id", "run_scope", "evaluation_model_ids",
        "core_models", "transfer_models", "budget", "metrics", "matrix_summary",
        "required_gates", "test_use_policy", "bindings", "requirements_source_sha256",
    )
    return canonical_hash({key: protocol[key] for key in keys})


def expected_record_keys(
    ctx: ProtocolContext, mode: str = "production", limit_per_dataset: int = 0
) -> set[tuple[str, str]]:
    if mode == "production":
        if limit_per_dataset:
            raise ValueError("test coverage cannot be reduced")
        rows = ctx.test_rows
    elif mode in {"pilot", "validation"}:
        if not 1 <= limit_per_dataset <= 20:
            raise ValueError("validation limit must be 1..20 per dataset")
        rows = [
            row
            for dataset in ("kitti", "nuscenes")
            for row in sorted(
                (row for row in ctx.training_rows if row["split"] == "validation" and row["dataset"] == dataset),
                key=lambda row: row["sample_id"],
            )[:limit_per_dataset]
        ]
    else:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    return {(row["sample_id"], model) for row in rows for model in ctx.evaluation_model_ids}


def local_input_paths(root: Path) -> list[str]:
    selected: set[str] = set()
    for relative in (
        "README_CANDIDATE.md", "run_local_candidate_tests.py",
        "00_冻结协议/PROTOCOL.md",
        "00_冻结协议/METHOD_AUDIT.md", "00_冻结协议/SERVER_PREFLIGHT.md",
        "00_冻结协议/METRICS.md",
        "00_冻结协议/DEPTH_NUMERIC_POLICY.md",
        "00_冻结协议/protocol_runtime.py",
        "00_冻结协议/validate_candidate.py",
        "00_冻结协议/generate_run_matrix.py",
        "00_冻结协议/freeze_run_matrix.py",
        "00_冻结协议/prepare_formal_v5.py",
        "00_冻结协议/prepare_protocol_v2.py",
        "00_冻结协议/seal_preflight.py",
        "00_冻结协议/research_blockers.json", "00_冻结协议/run_matrix.csv",
        "01_公共数据集/runtime_v1/train_validation_manifest.csv",
        "01_公共数据集/runtime_v1/test_manifest.csv",
        "01_公共数据集/car_00.jpg", "01_公共数据集/car_00_mask.jpg",
        "02_模型/model_manifest.json", "02_模型/model_adapters.py",
        "02_模型/bind_v9_model_inputs.py",
        "02_模型/evaluate_clean_baselines.py",
        "02_模型/validate_v9_depth_models.py",
        "03_方法/method_source_manifest.json",
        "03_方法/SOURCE_TRACK_READINESS.md",
        "03_方法/audit_source_tracks.py",
        "03_方法/source_track_readiness.json",
        "03_方法/production/method_contracts.json",
        "03_方法/production/printable30values.txt",
        "03_方法/production/losses.py", "03_方法/production/objectives.py",
        "03_方法/production/trainer.py", "04_统一评估/evaluator.py",
    ):
        if (root / relative).is_file():
            selected.add(relative)
    for directory in ("01_公共数据集", "03_方法/production", "04_统一评估"):
        for path in (root / directory).glob("*.py"):
            if not path.name.startswith("test_"):
                selected.add(str(path.relative_to(root)))
    return sorted(selected)


def _dataset_errors(ctx: ProtocolContext) -> list[str]:
    rows = ctx.training_rows + ctx.test_rows
    expected = {
        (dataset, split): count
        for dataset in ("kitti", "nuscenes")
        for split, count in (("train", 100), ("validation", 20), ("test", 100))
    }
    errors = []
    if Counter((row["dataset"], row["split"]) for row in rows) != expected:
        errors.append("dataset split/domain counts differ from frozen 100/20/100")
    if len({row["sample_id"] for row in rows}) != len(rows):
        errors.append("sample identities overlap across splits")
    if any(row["split"] == "test" for row in ctx.training_rows):
        errors.append("test rows appear in training manifest")
    return errors


def _source_track_errors(ctx: ProtocolContext) -> list[str]:
    errors: list[str] = []
    path = ctx.path(ctx.protocol["source_track_readiness_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "blocked_external_inputs":
        errors.append("source tracks must remain blocked until external inputs are bound")
    expected_tracks = {"fca_official_3d", "3d2fool_official_3d"}
    if set(report.get("tracks", {})) != expected_tracks:
        errors.append("source-track inventory differs")
        return errors
    for name, track in report["tracks"].items():
        if track.get("runnable") is not False:
            errors.append(f"source track unexpectedly runnable: {name}")
        if not track.get("external_requirements"):
            errors.append(f"source track lacks external requirements: {name}")
        entry = ctx.path(track["source_entry"])
        if not entry.is_file():
            errors.append(f"source-track entry missing: {name}")
        for asset in track.get("local_assets", []):
            asset_path = ctx.path(asset["path"])
            if not asset_path.is_file():
                errors.append(f"source-track asset missing: {asset['path']}")
                continue
            if asset_path.stat().st_size != asset.get("bytes"):
                errors.append(f"source-track asset size drift: {asset['path']}")
            if sha256(asset_path) != asset.get("sha256"):
                errors.append(f"source-track asset hash drift: {asset['path']}")
    return errors


def _remote_input_errors(ctx: ProtocolContext) -> list[str]:
    errors: list[str] = []
    for item in ctx.protocol.get("external_inputs", []):
        path = Path(item["path"])
        if not path.is_file():
            errors.append(f"remote external input missing: {item['id']}")
            continue
        if item.get("bytes") is not None and path.stat().st_size != item["bytes"]:
            errors.append(f"remote external input size drift: {item['id']}")
        if sha256(path) != item["sha256"]:
            errors.append(f"remote external input hash drift: {item['id']}")
    for row in ctx.training_rows + ctx.test_rows:
        path = Path(row["source_path"])
        if not path.is_file():
            errors.append(f"remote dataset source missing: {row['sample_id']}")
            continue
        if sha256(path) != row["source_sha256"]:
            errors.append(f"remote dataset source hash drift: {row['sample_id']}")
    return errors


def validate_candidate(ctx: ProtocolContext, mode: str = "local") -> dict[str, Any]:
    if mode not in {"local", "remote"}:
        raise ValueError("mode must be local or remote")
    p = ctx.protocol
    if p.get("protocol_id") == FORMAL_PROTOCOL_ID:
        return validate_formal(ctx, mode)
    errors: list[str] = []
    blockers: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(p.get("protocol_id") == PROTOCOL_ID and p.get("run_id") == RUN_ID, "candidate identity differs")
    check(p.get("run_scope") == "validation_only", "candidate must be validation-only")
    check(p.get("formal_test_allowed") is False and p.get("long_run_allowed") is False, "formal/long-run flags must remain false")
    check(p.get("status") == "blocked_preformal_preparation", "candidate status must remain blocked")
    check(tuple(p.get("evaluation_model_ids", [])) == MODEL_IDS, "eight-model order differs")
    check(tuple(p.get("core_models", [])) == CORE_MODEL_IDS, "core model set differs")
    check(tuple(p.get("transfer_models", [])) == TRANSFER_MODEL_IDS, "transfer model set differs")
    check(p.get("required_gates") == {gate: False for gate in REQUIRED_GATES}, "pre-formal gates cannot be toggled ready")
    check(p.get("test_use_policy", {}).get("status") == "forbidden_until_separate_formal_seal", "test-use gate differs")
    check(len(ctx.matrix) == len(set(ctx.expected_config_ids)) == 32, "candidate matrix must contain 32 unique planned rows")
    check(sum(row.get("run_phase") == "baseline" for row in ctx.matrix) == 27, "baseline row count differs")
    check(sum(row.get("run_phase") == "deferred_ours" for row in ctx.matrix) == 5, "deferred ours row count differs")
    for row in ctx.matrix:
        cid = row.get("config_id", "<missing>")
        check(row.get("formal_status") == "blocked" and row.get("status") == "candidate_only", f"row unexpectedly runnable: {cid}")
        check(row.get("test_time_optimization") == "false", f"test-time optimization enabled: {cid}")
        check(row.get("evaluation_models", "").split("+") == list(MODEL_IDS), f"evaluation roster differs: {cid}")
        check(not ({"midas", "adabins"} & set(row.get("train_models", "").split("+"))), f"historical model in new matrix: {cid}")
    try:
        for field, relative in (
            ("model_manifest_sha256", p["model_manifest_path"]),
            ("run_matrix_sha256", p["run_matrix_path"]),
            ("training_manifest_sha256", p["training_manifest_path"]),
            ("test_manifest_sha256", p["test_manifest_path"]),
        ):
            check(sha256(ctx.path(relative)) == p["bindings"][field], f"binding drift: {field}")
        check(sha256(ctx.path("01_公共数据集/car_00.jpg")) == p["bindings"]["vehicle_sha256"], "vehicle asset drift")
        check(sha256(ctx.path("01_公共数据集/car_00_mask.jpg")) == p["bindings"]["vehicle_mask_sha256"], "vehicle mask drift")
        local_bindings = p["bindings"].get("local_input_sha256", {})
        check(set(local_bindings) == set(local_input_paths(ctx.root)), "local executable/input set drift")
        for relative, expected in local_bindings.items():
            check(sha256(ctx.path(relative)) == expected, f"local input drift: {relative}")
        manifest = json.loads(ctx.path(p["model_manifest_path"]).read_text(encoding="utf-8"))
        check(tuple(manifest.get("model_order", [])) == MODEL_IDS, "model manifest order differs")
        check({item["id"] for item in manifest.get("models", [])} == set(MODEL_IDS), "model manifest exact set differs")
        check(not ({"midas", "adabins"} & {item["id"] for item in manifest.get("models", [])}), "historical victim model remains active")
        project_root = ctx.root.parents[3]
        requirements = p.get("requirements_source_sha256", {})
        check(set(requirements) == set(p.get("requirements_sources", [])), "requirements source binding set differs")
        for relative, expected in requirements.items():
            source = project_root / relative
            if source.is_file():
                check(sha256(source) == expected, f"requirements source drift: {relative}")
            else:
                # The independent remote candidate does not contain the parent
                # workspace.  Its immutable source hashes were computed before
                # upload; validate their shape rather than inventing a remote
                # absolute path.
                check(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected), f"requirements source hash malformed: {relative}")
        check(candidate_fingerprint(p) == p.get("candidate_fingerprint"), "candidate fingerprint drift")
        errors.extend(_dataset_errors(ctx))
        errors.extend(_source_track_errors(ctx))
        if mode == "remote":
            errors.extend(_remote_input_errors(ctx))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"candidate validation failed: {exc}")

    ledger = json.loads(ctx.path(p["research_blockers_path"]).read_text(encoding="utf-8"))
    blockers.extend(f"research:{entry['id']}" for entry in ledger.get("blockers", []) if entry.get("status") == "blocked")
    blockers.extend(f"unbound_external:{name}" for name in p.get("unbound_external_inputs", []))
    blockers.extend(f"gate:{gate}" for gate, passed in p.get("required_gates", {}).items() if not passed)
    if mode == "remote":
        blockers.append("remote_live_model_and_gradient_smokes_not_completed")
    else:
        blockers.append("external_inputs_not_verified_live")
    blockers.append("separate_immutable_formal_protocol_not_created")
    return {
        "schema_version": 1, "protocol_id": p.get("protocol_id"), "run_id": p.get("run_id"),
        "mode": mode, "status": "fail" if errors else "pass", "ready": False,
        "formal_allowed": False, "errors": errors, "blockers": blockers,
        "checked": {"matrix_rows": len(ctx.matrix), "models": len(MODEL_IDS)},
        "candidate_fingerprint": p.get("candidate_fingerprint"),
    }


def validate_formal(ctx: ProtocolContext, mode: str = "local") -> dict[str, Any]:
    """Validate the separately sealed formal protocol and its exact input closure."""
    if mode not in {"local", "remote"}:
        raise ValueError("mode must be local or remote")
    p = ctx.protocol
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(p.get("protocol_id") == FORMAL_PROTOCOL_ID, "formal protocol identity differs")
    check(p.get("run_id") == FORMAL_RUN_ID, "formal run identity differs")
    check(p.get("run_scope") == "formal_test", "formal run scope differs")
    check(p.get("status") == "sealed_ready", "formal protocol is not sealed-ready")
    check(p.get("formal_test_allowed") is True and p.get("long_run_allowed") is True,
          "formal authorization flags differ")
    check(tuple(p.get("formal_config_ids", [])) == FORMAL_CONFIG_IDS,
          "formal configuration roster/order differs")
    check(tuple(ctx.expected_config_ids) == FORMAL_CONFIG_IDS,
          "formal matrix configuration roster/order differs")
    check(len(ctx.matrix) == len(FORMAL_CONFIG_IDS), "formal matrix must contain exactly 18 rows")
    check(tuple(p.get("evaluation_model_ids", [])) == MODEL_IDS, "eight-model order differs")
    check(tuple(p.get("core_models", [])) == CORE_MODEL_IDS, "core model set differs")
    check(tuple(p.get("transfer_models", [])) == TRANSFER_MODEL_IDS, "transfer model set differs")
    check(p.get("storage_policy", {}).get("raw_depth_tensors") == "metrics-only",
          "formal storage policy must remain metrics-only")
    check(p.get("test_use_policy", {}).get("optimization") is False,
          "test optimization must remain forbidden")
    check(p.get("test_use_policy", {}).get("single_execution_per_configuration") is True,
          "formal test must remain single-use per configuration")
    for row in ctx.matrix:
        cid = row.get("config_id", "<missing>")
        check(row.get("status") == "frozen_ready" and row.get("formal_status") == "ready",
              f"formal row is not ready: {cid}")
        check(row.get("test_time_optimization") == "false", f"test optimization enabled: {cid}")
        check(row.get("evaluation_models", "").split("+") == list(MODEL_IDS),
              f"evaluation roster differs: {cid}")
        check(row.get("artifact_output") == f"05_运行/{FORMAL_RUN_ID}/artifacts/{cid}",
              f"formal artifact output differs: {cid}")
        check(row.get("evaluation_output") == f"06_结果/{FORMAL_RUN_ID}/{cid}",
              f"formal evaluation output differs: {cid}")
    try:
        for field, relative in (
            ("model_manifest_sha256", p["model_manifest_path"]),
            ("run_matrix_sha256", p["run_matrix_path"]),
            ("training_manifest_sha256", p["training_manifest_path"]),
            ("test_manifest_sha256", p["test_manifest_path"]),
        ):
            check(sha256(ctx.path(relative)) == p["bindings"][field], f"binding drift: {field}")
        check(sha256(ctx.path("01_公共数据集/car_00.jpg")) == p["bindings"]["vehicle_sha256"],
              "vehicle asset drift")
        check(sha256(ctx.path("01_公共数据集/car_00_mask.jpg")) == p["bindings"]["vehicle_mask_sha256"],
              "vehicle mask drift")
        formal_inputs = p.get("formal_input_sha256", {})
        check(bool(formal_inputs), "formal input closure is empty")
        for relative, expected in formal_inputs.items():
            path = ctx.path(relative)
            check(path.is_file() and sha256(path) == expected, f"formal input drift: {relative}")
        source = p.get("formal_source", {})
        source_protocol_path = ctx.path(source["candidate_protocol_path"])
        check(sha256(source_protocol_path) == source["candidate_protocol_sha256"],
              "repaired source protocol hash drift")
        source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
        check(source_protocol.get("candidate_fingerprint") == source.get("candidate_fingerprint"),
              "repaired source protocol fingerprint differs")
        evidence_path = ctx.path(source["validation_audit_path"])
        check(sha256(evidence_path) == source["validation_audit_sha256"],
              "combined validation audit hash drift")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        check(evidence.get("formal_successor_eligible_by_this_audit") is True,
              "combined validation audit did not authorize a successor")
        check(evidence.get("candidate_fingerprint") == source.get("candidate_fingerprint"),
              "combined validation candidate fingerprint differs")
        check(source.get("candidate_fingerprint") ==
              "6f691954c0e88f37af398a1382ab2bb3a7cd0473edbd74e6ac8907ddd5ab2d35",
              "unexpected repaired candidate fingerprint")
        check(candidate_fingerprint(p) == p.get("candidate_fingerprint"),
              "formal candidate fingerprint drift")
        errors.extend(_dataset_errors(ctx))
        if mode == "remote":
            errors.extend(_remote_input_errors(ctx))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"formal validation failed: {exc}")
    return {
        "schema_version": 1,
        "protocol_id": p.get("protocol_id"),
        "run_id": p.get("run_id"),
        "mode": mode,
        "status": "fail" if errors else "pass",
        "ready": not errors,
        "formal_allowed": not errors,
        "errors": errors,
        "blockers": [],
        "checked": {"matrix_rows": len(ctx.matrix), "models": len(MODEL_IDS)},
        "candidate_fingerprint": p.get("candidate_fingerprint"),
    }


def require_run_allowed(
    ctx: ProtocolContext, purpose: str = "formal", config_id: str | None = None
) -> None:
    if purpose not in {"formal", "validation", "diagnostic"}:
        raise ValueError("unknown run purpose")
    if ctx.protocol.get("protocol_id") == FORMAL_PROTOCOL_ID:
        report = validate_formal(ctx, mode="local")
        if report["errors"]:
            raise ValueError("formal input validation failed: " + "; ".join(report["errors"][:5]))
        if purpose != "formal":
            raise ValueError("sealed formal protocol only permits formal execution")
        if config_id is not None and config_id not in FORMAL_CONFIG_IDS:
            raise ValueError(f"configuration is outside the sealed formal roster: {config_id}")
        return
    report = validate_candidate(ctx, mode="local")
    if report["errors"]:
        raise ValueError("candidate input validation failed: " + "; ".join(report["errors"][:5]))
    if purpose == "formal":
        raise ValueError("FORMAL BLOCKED: normalized-v9 is a validation-only preparation candidate")
    if config_id is None:
        raise ValueError("validation/diagnostic must identify a configuration")
    row = ctx.row(config_id)
    if row["run_phase"] == "deferred_ours":
        raise ValueError("ours is deferred until the baseline is frozen")
    if row["method"] in {"fca", "3d2fool"}:
        raise ValueError(f"{row['method']} requires its official 3D source-track runner")
    if row["method"] == "apa" and purpose != "diagnostic":
        raise ValueError("APA remains diagnostic until unit/direction/gradient validation passes")
    if row["method"] != "apa" and purpose == "diagnostic":
        raise ValueError("diagnostic purpose is currently reserved for APA")
