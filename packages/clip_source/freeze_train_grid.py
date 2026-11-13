"""Freeze thresholds only after matching the four retained training files."""
from __future__ import annotations

import argparse
from pathlib import Path

from clip_evidence import require_historical_reference
from train_norm_grid import build_plan, write_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require_historical_reference(args.reference_run)
    plan = build_plan(args.reference_run / 'summary.json', args.reference_run / 'attack_trace.jsonl')
    digest = write_plan(plan, args.output)
    print('Frozen TRAIN-only grid SHA256:', digest)
    print('Canonical groups:', ', '.join(group['id'] for group in plan['groups']))


if __name__ == '__main__':
    main()
