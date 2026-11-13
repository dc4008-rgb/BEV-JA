#!/usr/bin/env python3
"""Check the extracted input package against its explicit member manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())
    failures = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if path.is_symlink() or not path.is_file():
            failures.append(row["path"])
            continue
        if ROOT not in path.resolve().parents:
            raise ValueError("member escapes package root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append(row["path"])
    print(json.dumps({"status": "pass" if not failures else "fail", "files": len(manifest["files"]),
                      "failures": failures, "test_data_included": False}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
