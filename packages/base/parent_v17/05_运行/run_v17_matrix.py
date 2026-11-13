"""Run only the four predeclared 100-update train/validation candidates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IDS = ("contextual_100", "robust_dpatch_100", "ours_det_only_100", "ours_dual_100")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(expected):
    p = ROOT / "PACKAGE_MANIFEST.json"
    if sha(p) != expected:
        raise ValueError("input package manifest SHA mismatch")
    m = json.loads(p.read_text())
    if m["candidate_id"] != ROOT.name or m["formal_test_allowed"] or m["test_data_included"]:
        raise ValueError("wrong candidate package identity/scope")
    for row in m["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or "runs" in rel.parts:
            raise ValueError("unsafe package path")
        p = ROOT / rel
        if not p.is_file() or p.is_symlink() or sha(p) != row["sha256"]:
            raise ValueError("changed input: " + str(rel))
    protocol = json.loads((ROOT / "00_冻结协议/full_validation_v17.json").read_text())
    if tuple(x["run_id"] for x in protocol["matrix"]) != IDS or protocol["updates"] != 100 or protocol["formal_test_allowed"]:
        raise ValueError("unregistered matrix")
    return protocol


def stage_commands(session, snapshot, inventory, inventory_sha):
    commands = []
    for name in IDS:
        run = session / name
        if name.startswith("ours_"):
            variant = "dual" if name == "ours_dual_100" else "det-only"
            train = ["05_运行/run_ours_stateful_validation.py", "--task-variant", variant,
                     "--mode", "validation", "--sd35-snapshot", str(snapshot),
                     "--sd35-file-manifest", str(inventory), "--sd35-file-manifest-sha256", inventory_sha,
                     "--output", str(run)]
        else:
            method = "contextual" if name == "contextual_100" else "robust_dpatch"
            train = ["05_运行/run_road_patch_candidate.py", "--method", method,
                     "--mode", "validation", "--output", str(run)]
        commands.append((name + "_train", train))
        commands.append((name + "_probe", ["05_运行/evaluate_road_train_probes.py", "--run-output", str(run),
                        "--output", str(session / (name + "_probe"))]))
        commands.append((name + "_detection", ["05_运行/evaluate_road_comparison.py", "--run-output", str(run),
                        "--output", str(session / (name + "_detection"))]))
        if name == "ours_dual_100":
            commands.append((name + "_depth", ["05_运行/evaluate_ours_stateful_validation.py", "--mode", "validation",
                            "--run-output", str(run), "--output", str(session / (name + "_depth"))]))
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--package-manifest-sha256", required=True)
    p.add_argument("--sd35-snapshot", type=Path, required=True)
    p.add_argument("--sd35-file-manifest", type=Path, required=True)
    p.add_argument("--sd35-file-manifest-sha256", required=True)
    p.add_argument("--session", default="full100_validation_v1")
    a = p.parse_args()
    protocol = verify_inputs(a.package_manifest_sha256)
    if platform.node() != a.expected_hostname:
        raise RuntimeError("server identity mismatch; recheck live endpoint")
    if Path(a.session).name != a.session or a.session in ("", ".", ".."):
        raise ValueError("session must be a single new directory name")
    if sha(a.sd35_file_manifest) != a.sd35_file_manifest_sha256:
        raise ValueError("SD3.5 inventory binding mismatch")
    out = ROOT / "runs" / a.session
    commands = stage_commands(out, a.sd35_snapshot, a.sd35_file_manifest, a.sd35_file_manifest_sha256)
    receipt = {"status": "preflight_pass", "hostname": platform.node(), "protocol_sha256": sha(ROOT / "00_冻结协议/full_validation_v17.json"),
               "package_manifest_sha256": a.package_manifest_sha256, "scope": "100update_train_validation_only",
               "test_data_opened": False, "stages": [name for name, _ in commands], "completed": []}
    if a.mode == "preflight":
        print(json.dumps(receipt, indent=2)); return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU already has compute processes; inspect ownership first")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    receipt["status"] = "running"
    receipt["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt_path = out / "MATRIX_STATUS.json"
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2))
    save()
    for name, args in commands:
        stage = out / "stage_reports" / name
        stage.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        entry = {"stage": name, "args": args, "timeout_seconds": 900, "status": "running"}
        (stage / "status.json").write_text(json.dumps(entry, indent=2))
        with (stage / "run.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-B", *args], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            entry["pid"] = process.pid
            try:
                code = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                code = 124
        entry.update(status="pass" if code == 0 else "failed", exit_code=code,
                     elapsed_seconds=time.monotonic()-started, log_sha256=sha(stage / "run.log"))
        (stage / "status.json").write_text(json.dumps(entry, indent=2))
        if code:
            receipt.update(status="failed_retained", failed_stage=name); save(); raise SystemExit(code)
        receipt["completed"].append(name); save()
    receipt.update(status="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())); save()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
