#!/usr/bin/env python3
"""Merge per-sample atomic JASR rows from several runs of jasr_from_evaluation_dir_v1.py
(e.g. two 100-sample nodes) into one summary JSON with the same schema. Pure re-aggregation;
no per-sample value is recomputed.
Usage: merge_jasr_atomic_v1.py --label L --output OUT.json a.atomic.json b.atomic.json ...
"""
import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).absolute().parent))
import jasr_from_evaluation_dir_v1 as J  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('atomic', nargs='+')
    ap.add_argument('--label', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--detectors', default=None)
    ap.add_argument('--depths', default=None)
    ap.add_argument('--add-detector', action='append', default=[],
                    help='detection-only rows JSON (detection_only_from_stage_v1.py); combined with the depth '
                         'outcomes already present in the atomic rows of the same sample')
    a = ap.parse_args()
    atomic, sources = [], []
    for p in a.atomic:
        rows = json.load(open(p))
        atomic.extend(rows)
        sources.append({'path': str(Path(p).absolute()), 'rows': len(rows), 'sha256': J.file_sha(p)})
    depth_keys = ('clean_depth_valid', 'invalid_reason', 'roi_pixels', 'kappa', 'q_clean', 'native_dtype',
                  'FarShift', 'rho', 'depth_success')
    for p in a.add_detector:
        det = json.load(open(p))
        sources.append({'path': str(Path(p).absolute()), 'rows': len(det['rows']), 'sha256': J.file_sha(p), 'detection_only': True})
        existing = {(r['sample_id'], r['detector']) for r in atomic}
        for d in det['rows']:
            sid = d['sample_id']
            J.need((sid, d['detector']) not in existing, 'detector already present for ' + sid)
            # depth outcomes are detector-independent: take them from any existing detector of this sample
            per_depth = {}
            for r in atomic:
                if r['sample_id'] == sid and r['depth_model'] not in per_depth:
                    per_depth[r['depth_model']] = r
            J.need(per_depth, 'no depth rows for ' + sid)
            for z, r in per_depth.items():
                dep = {k: r[k] for k in depth_keys}
                eligible = d['clean_match']['matched'] and dep['clean_depth_valid']
                atomic.append(dict(sample_id=sid, dataset=d['dataset'], source_split=d['source_split'], detector=d['detector'],
                                   depth_model=z, clean_match=d['clean_match'], final_match=d['final_match'],
                                   eligible_clean_pair=bool(eligible), detection_success=not d['final_match']['matched'], **dep,
                                   joint_success=bool(not d['final_match']['matched'] and dep['depth_success']) if eligible else None))
    dets = tuple(a.detectors.split(',')) if a.detectors else tuple(sorted({r['detector'] for r in atomic}, key=J.DEFAULT_DETECTORS.index))
    deps = tuple(a.depths.split(',')) if a.depths else tuple(sorted({r['depth_model'] for r in atomic}, key=J.DEFAULT_DEPTHS.index))
    J.DETECTORS, J.DEPTHS = dets, deps
    sample_ids = sorted({r['sample_id'] for r in atomic})
    J.need(len(sample_ids) == len(set(sample_ids)), 'sample ids')
    black_det = tuple(d for d in dets if d not in J.WHITE_DET)
    black_dep = tuple(z for z in deps if z not in J.WHITE_DEP)
    out = Path(a.output).absolute()
    J.need(not out.exists(), 'output exists: ' + str(out))
    result = OrderedDict(schema='bevja-jasr-merged-atomic-v1', label=a.label, sources=sources, created_unix=time.time(),
                         thresholds=J.THRESHOLDS, numeric_policy=J.core.NUMERIC_POLICY, detectors=dets, depth_models=deps,
                         pair_count=len(dets) * len(deps), whitebox_models={'detectors': J.WHITE_DET, 'depth': J.WHITE_DEP},
                         sample_count=len(sample_ids), domains=OrderedDict())
    for dom in J.DOMAINS:
        sids = [s for s in sample_ids if dom == 'combined' or s.startswith(dom + '_')]
        sset = set(sids)
        rows = J.pair_table([r for r in atomic if r['sample_id'] in sset], sids)
        result['domains'][dom] = OrderedDict(
            sample_count=len(sids), pairs=rows,
            macro_all_valid_pairs={k: J.macro(rows, k) for k in J.METRICS},
            macro_whitebox_pairs={k: J.macro(rows, k, J.WHITE_DET, J.WHITE_DEP) for k in J.METRICS},
            macro_blackbox_pairs={k: J.macro(rows, k, black_det, black_dep) for k in J.METRICS},
            macro_per_detector={d: J.macro(rows, 'DetASR', (d,)) for d in dets},
            macro_per_depth={z: J.macro(rows, 'DepASR', None, (z,)) for z in deps})
    result['atomic_sha256'] = hashlib.sha256(json.dumps(atomic, sort_keys=True, default=str).encode()).hexdigest()
    result['atomic_count'] = len(atomic)
    json.dump(result, out.open('x', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(atomic, out.with_suffix('.atomic.json').open('x', encoding='utf-8'), ensure_ascii=False, default=str)
    for dom in J.DOMAINS:
        d = result['domains'][dom]
        print('%-9s N=%d macroJASR=%s (valid %d) white=%s black=%s' % (
            dom, d['sample_count'], d['macro_all_valid_pairs']['JASR']['value'], d['macro_all_valid_pairs']['JASR']['valid_pairs'],
            d['macro_whitebox_pairs']['JASR']['value'], d['macro_blackbox_pairs']['JASR']['value']))


if __name__ == '__main__':
    main()
