#!/usr/bin/env python3
"""Copy small B3 evidence to a readback directory and optionally delete raw outputs."""
import argparse, hashlib, json, shutil
from pathlib import Path

SMALL = [
 ('control/QUEUE_RETURNED.json', 'control/QUEUE_RETURNED.json'),
 ('control/RUNNER.log', 'control/RUNNER.log'),
 ('JASR/{slug}_JASR.json', 'JASR/{slug}_JASR.json'),
 ('READBACK/detection_summary.json', 'READBACK/detection_summary.json'),
 ('READBACK/depth_summary.json', 'READBACK/depth_summary.json'),
 ('READBACK/ABLATION_FULL200_ALL8_EVALUATION_RETURNED.json', 'READBACK/ABLATION_FULL200_ALL8_EVALUATION_RETURNED.json'),
]
FROM_DATA = [
 ('CPU_VERIFICATION/ABLATION_FULL200_CPU_VERIFIED.json', 'CPU_VERIFICATION/ABLATION_FULL200_CPU_VERIFIED.json'),
 ('TRAIN100_NATIVE_CPU/TRAIN100_NATIVE_CPU_ACCEPTED.json', 'TRAIN100_NATIVE_CPU/TRAIN100_NATIVE_CPU_ACCEPTED.json'),
 ('evaluation_100/detection_summary.json', 'evaluation_100/detection_summary.json'),
 ('evaluation_100/depth_summary.json', 'evaluation_100/depth_summary.json'),
]


def desc(p):
    b = Path(p).read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--delete-raw', action='store_true')
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    for slug, root in runs.items():
        root = Path(root)
        dst = out_root / slug; dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for rel, target in SMALL:
            src = root / rel.format(slug=slug)
            if src.exists():
                tgt = dst / target.format(slug=slug)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tgt); copied.append(desc(tgt))
        summary = root / 'control/QUEUE_RETURNED.json'
        data = None
        if summary.exists():
            data = Path(json.loads(summary.read_text())['data_root'])
        else:
            data = Path('/root/autodl-tmp') / root.parent.name / root.name
        for rel, target in FROM_DATA:
            src = (data / rel) if data is not None else None
            if src is not None and src.exists():
                (dst / target).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / target); copied.append(desc(dst / target))
        (dst / 'READBACK_MANIFEST.json').write_text(json.dumps(copied, indent=2) + '\n')
        print('readback', slug, len(copied), 'files')
        if args.delete_raw:
            if data is None:
                raise SystemExit(f'{slug}: no data_root in QUEUE_RETURNED.json')
            for rel in ('part_0', 'part_1', 'evaluation_100', 'CPU_VERIFICATION', 'TRAIN100_NATIVE_CPU'):
                p = data / rel
                if p.exists():
                    shutil.rmtree(p)
            print('deleted raw', slug, str(data))


if __name__ == '__main__':
    main()
