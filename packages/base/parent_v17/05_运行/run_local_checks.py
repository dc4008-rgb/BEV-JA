#!/usr/bin/env python3
"""Repeat local structural checks; this script never loads research checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = [("road_components", "03_方法/production", "test_road*.py"),
              ("baseline_integration", "05_运行", "test_road_runner_integration.py"),
              ("ours_road", "03_方法/production", "test_ours_road_v16.py")]
    results = []
    for name, directory, pattern in groups:
        started = time.monotonic()
        process = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s",
                                  str(ROOT / directory), "-p", pattern, "-v"],
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                 capture_output=True, text=True, timeout=180)
        output = process.stdout + process.stderr
        count = re.search(r"Ran (\d+) tests? in", output)
        test_count = int(count.group(1)) if count else 0
        (args.output / (name + ".log")).write_text(output)
        results.append({"group": name, "exit_code": process.returncode if test_count else 1,
                        "test_count": test_count, "seconds": time.monotonic()-started,
                        "log_sha256": hashlib.sha256(output.encode()).hexdigest()})
        print(name, process.returncode, flush=True)
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(ROOT.rglob("*.py")) if "runs" not in p.relative_to(ROOT).parts}
    summary = {"status": "pass" if all(r["exit_code"] == 0 for r in results) else "fail",
               "scope": "synthetic CPU scenes and random tiny torchvision heads; no pretrained/GPU efficacy",
               "test_images_opened": False, "real_model_smoke_passed": False,
               "checks": results, "code_sha256": source}
    (args.output / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
