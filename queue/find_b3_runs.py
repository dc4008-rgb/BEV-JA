#!/usr/bin/env python3
"""Find the latest B3 run root for each job on a node."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--base', default='/root')
    ap.add_argument('--data-base', default='/root/autodl-tmp')
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-missing', action='store_true')
    args = ap.parse_args()
    jobs = json.loads(Path(args.manifest).read_text())
    base = Path(args.base); data_base = Path(args.data_base)
    runs = {}
    for j in jobs:
        ns = base / j['namespace']; data_ns = data_base / j['namespace']
        cands = [p for p in ns.glob(j['slug'] + '_*') if p.is_dir() and ((data_ns / p.name / 'evaluation_100').is_dir() or (p / 'evaluation_100').is_dir())]
        if not cands:
            if args.allow_missing:
                continue
            raise SystemExit(f'no evaluation_100 for {ns}/{j["slug"]}_*')
        runs[j['slug']] = str(max(cands, key=lambda p: p.stat().st_mtime))
    Path(args.out).write_text(json.dumps(runs, indent=2) + '\n')
    print(json.dumps(runs, indent=2))


if __name__ == '__main__':
    main()
