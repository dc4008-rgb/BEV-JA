#!/usr/bin/env python3
"""Inventory an existing local snapshot only; neither downloads nor runs models.

The resulting SHA is an explicit runtime identity binding, not evidence that
any real GPU smoke or scientific validation has passed. Review/capture this
inventory before passing both it and its SHA to the v16 stateful runner.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from run_ours_stateful_validation import ROOT, validate_snapshot, sha256
from ours_road_v16 import require_candidate_run_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_candidate_run_path(args.output, ROOT)
    if output.exists():
        raise FileExistsError("refusing to overwrite snapshot inventory")
    evidence = validate_snapshot(args.snapshot)
    files = sorted(path for path in args.snapshot.rglob("*") if path.is_file())
    payload = {"schema_version": 1, "model_id": "stabilityai/stable-diffusion-3.5-medium",
               "snapshot_path": str(args.snapshot.resolve()), "scope": "local_file_identity_only",
               "gpu_verified": False, "formal_test_allowed": False,
               "files": [{"path": str(path.relative_to(args.snapshot)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)} for path in files]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"inventory": str(output), "inventory_sha256": sha256(output),
                      "snapshot": evidence, "gpu_verified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
