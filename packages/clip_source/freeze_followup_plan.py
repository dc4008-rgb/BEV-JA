#!/usr/bin/env python3
"""Freeze the explicit cap120 post-evaluation follow-up only after exact OFF validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from followup_contract import build_followup_plan, sha


def write_new(path, plan):
    path = Path(path)
    if path.exists() or any(p.is_symlink() for p in (path, *path.absolute().parents)):
        raise FileExistsError("new nonsymlink plan output required")
    if not path.parent.is_dir():
        raise FileNotFoundError("plan parent must already exist")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return sha(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-run", "historical-off-run", "off-run", "comparison", "points", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refuse to overwrite a follow-up plan")
    plan = build_followup_plan(reference_run=args.reference_run,
        historical_off_run=args.historical_off_run, off_run=args.off_run,
        comparison_file=args.comparison, points_file=args.points)
    digest = write_new(args.output, plan)
    print(json.dumps({"status": plan["status"], "group": "cap120", "max_norm": 120.0,
                      "output": str(args.output), "sha256": digest,
                      "execution_off_origin": plan["execution_off"]["origin"]}))


if __name__ == "__main__":
    main()

