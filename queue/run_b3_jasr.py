#!/usr/bin/env python3
"""Run JASR on each B3 evaluation_100 directory."""
import argparse, json, shutil, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--jasr', required=True)
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    jobs = {j['slug']: j for j in json.loads(Path(args.manifest).read_text())}
    for slug, root in runs.items():
        root = Path(root)
        summary = root / 'control/QUEUE_RETURNED.json'
        data = Path(json.loads(summary.read_text())['data_root']) if summary.exists() else Path('/root/autodl-tmp') / root.parent.name / root.name
        eval_dir = data / 'evaluation_100'
        out = root / 'JASR' / f'{slug}_JASR.json'
        if out.exists():
            print('skip', out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, args.jasr, str(eval_dir),
               '--method-id', jobs[slug]['method_id'], '--output', str(out), '--label', slug]
        print('run', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        d = json.loads(out.read_text())
        assert d['domains']['nuscenes']['sample_count'] == 100
        assert d['domains']['nuscenes']['macro_all_valid_pairs']['JASR']['valid_pairs'] == 16
        rb = root / 'READBACK'; rb.mkdir(exist_ok=True)
        for name in ('detection_summary.json', 'depth_summary.json',
                     'ABLATION_FULL200_ALL8_EVALUATION_RETURNED.json'):
            src = eval_dir / name
            if src.exists():
                shutil.copy2(src, rb / name)
        print('ok', out, d['domains']['nuscenes']['macro_all_valid_pairs']['JASR']['value'])


if __name__ == '__main__':
    main()
