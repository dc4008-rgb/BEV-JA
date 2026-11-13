#!/usr/bin/env python3
"""Detection-only same-target matching from a saved-patch YOLOv5s evaluation stage
(layout: <stage>/evaluation/raw_predictions/NNN.pt with 'sample', 'target_box', 'states').
Produces rows {sample_id, dataset, source_split, detector, clean_match, final_match,
detection_success} using the unchanged jasr_core.matched rule and fixed thresholds.
These rows are later combined with depth outcomes by merge_jasr_atomic_v1.py --add-detector.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).absolute().parent))
from atomic_counts import THRESHOLDS, need  # noqa: E402
from native_numpy import numpy_value  # noqa: E402
import jasr_core as core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir')
    ap.add_argument('--detector', default='yolov5s')
    ap.add_argument('--output', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    import torch
    raw = Path(a.raw_dir)
    files = sorted(raw.glob('*.pt'))
    need(len(files) == 200, 'expected 200 raw files, found %d' % len(files))
    rows, hashes = [], []
    for f in files:
        x = torch.load(f, map_location='cpu', weights_only=False)
        s = x['sample']
        rows.append(dict(sample_id=s['sample_id'], dataset=s['dataset'], source_split=s['source_split'],
                         detector=a.detector, target_box=x['target_box'],
                         input_sha256={k: x['input_sha256'][k] for k in ('clean', 'final')},
                         clean_match=core.matched({k: numpy_value(x['states']['clean'][k], labels=(k == 'labels'))
                                                   for k in ('boxes', 'scores', 'labels')}, x['target_box'], THRESHOLDS),
                         final_match=core.matched({k: numpy_value(x['states']['final'][k], labels=(k == 'labels'))
                                                   for k in ('boxes', 'scores', 'labels')}, x['target_box'], THRESHOLDS)))
        rows[-1]['detection_success'] = not rows[-1]['final_match']['matched']
        hashes.append(hashlib.sha256(f.read_bytes()).hexdigest())
    need(len({r['sample_id'] for r in rows}) == 200, 'duplicate sample ids')
    out = Path(a.output)
    need(not out.exists(), 'output exists')
    json.dump(dict(schema='bevja-detection-only-rows-v1', label=a.label, raw_dir=str(raw.absolute()), detector=a.detector,
                   thresholds=THRESHOLDS, created_unix=time.time(), file_sha256=hashes, rows=rows),
              out.open('x', encoding='utf-8'), ensure_ascii=False, indent=1)
    n = len(rows)
    print('clean matched %d/%d, final matched %d/%d, detection_success %d' % (
        sum(r['clean_match']['matched'] for r in rows), n, sum(r['final_match']['matched'] for r in rows), n,
        sum(r['detection_success'] for r in rows)))


if __name__ == '__main__':
    main()
