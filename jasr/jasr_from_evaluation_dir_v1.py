#!/usr/bin/env python3
"""Same-target Det-ASR / Dep-ASR / JASR / Far-Shift from a retained evaluation_200 directory.

Reuses the accepted numerical core (jasr_core.matched / jasr_core.depth_result) and the
fixed secondary thresholds (atomic_counts.THRESHOLDS) without modification. CPU only,
no model forwards, no recomputation of AP or AbsRel. One JSON output per evaluation dir.
"""
import argparse
import hashlib
import json
import socket
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
from atomic_counts import DETECTORS as DEFAULT_DETECTORS, DEPTHS as DEFAULT_DEPTHS, THRESHOLDS, need  # noqa: E402
from artifact_io import load_torch_lossless, file_sha  # noqa: E402
from native_numpy import numpy_value  # noqa: E402
import jasr_core as core  # noqa: E402

SCHEMA = 'bevja-jasr-from-evaluation-dir-v2'
DETECTORS, DEPTHS = DEFAULT_DETECTORS, DEFAULT_DEPTHS  # may be narrowed by --detectors / --depths
DOMAINS = ('combined', 'kitti', 'nuscenes')
# Surrogate (white-box) models of the complete BEV-JA configuration.
WHITE_DET, WHITE_DEP = ('fasterrcnn', 'ssd300'), ('monodepth2', 'dpt_large')
METRICS = ('DetASR', 'DepASR', 'JASR', 'FarShift_mean')


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


POOL_OVERRIDE = None  # --pool: external tensor pool used by older evaluation layouts


def index_manifests(roots, sub, models, states_required):
    """Merge the manifests of several evaluation directories (e.g. 50-sample parts, or a
    separate YOLOv5s run). Each entry remembers its own root/tensor pool."""
    by_key, manifests = {}, []
    for root in roots:
        manifest = read_json(root / sub / 'MANIFEST.json')
        manifests.append(manifest)
        need(set(states_required) <= set(manifest['states']), 'required states missing in ' + str(root))
        for row in manifest['entries']:
            if row['model'] not in models:
                continue
            key = (row['sample_id'], row['model'])
            need(key not in by_key, 'duplicate manifest entry ' + str(key))
            pool = POOL_OVERRIDE or ((root / 'tensor_pool') if (root / 'tensor_pool').is_dir() else None)
            by_key[key] = dict(row, _root=root, _pool=pool)
    sample_ids = sorted({k[0] for k in by_key})
    missing = [(sid, m) for sid in sample_ids for m in models if (sid, m) not in by_key]
    need(not missing, 'incomplete sample x model coverage, e.g. ' + str(missing[:3]))
    return manifests, by_key, sample_ids


def resolve(root, rel):
    rel = Path(rel)
    return rel if rel.is_absolute() else root / rel


def payload_of(entry, cap_bytes=256 * 1024 ** 2):
    root, pool = entry['_root'], entry['_pool']
    path = resolve(root, entry['path'])
    need(path.exists(), 'missing retained container ' + str(path))
    need(path.stat().st_size <= cap_bytes, 'container larger than cap: ' + str(path))
    if entry.get('sha256'):
        need(file_sha(path) == entry['sha256'], 'retained container sha differs: ' + str(path))
    return load_torch_lossless(path, pool_root=pool)


def compute_case(sid, det_idx, dep_idx, method_id):
    detections, depths = {}, {}
    target, input_hashes, dataset, split = None, None, None, None
    for model in DETECTORS:
        p = payload_of(det_idx[sid, model])
        need(p['sample_id'] == sid and p['model'] == model, 'detection payload identity differs')
        if method_id is not None:
            need(p.get('method_id') == method_id, 'method_id differs: %s vs %s' % (p.get('method_id'), method_id))
        hashes = {s: p['input_sha256'][s] for s in ('clean', 'final')}
        if target is None:
            target, input_hashes, dataset, split = p['target_box'], hashes, p['dataset'], p['source_split']
        need(p['target_box'] == target and hashes == input_hashes,
             'one clean/final input and target across detectors required')
        detections[model] = {
            s: core.matched({k: numpy_value(p[s][k], labels=(k == 'labels')) for k in ('boxes', 'scores', 'labels')},
                            target, THRESHOLDS)
            for s in ('clean', 'final')}
    mask_ref, mask = None, None
    for model in DEPTHS:
        entry = dep_idx[sid, model]
        p = payload_of(entry)
        need(p['sample_id'] == sid and p['model'] == model, 'depth payload identity differs')
        need({s: p['input_sha256'][s] for s in ('clean', 'final')} == input_hashes,
             'depth inputs differ from detection inputs')
        need(p['semantics'] == core.SEMANTICS[model][0], 'native depth semantics differ for ' + model)
        if mask_ref is None:
            mask_ref = p['object_mask']
            mask = numpy_value(load_torch_lossless(resolve(entry['_root'], mask_ref['path']), pool_root=entry['_pool']), mask=True)
        need(p['object_mask']['tensor_sha256'] == mask_ref['tensor_sha256'], 'one fixed foreground mask across depth models required')
        dtype = str(p['states']['clean'].dtype).split('torch.', 1)[-1]
        need(str(p['states']['final'].dtype).split('torch.', 1)[-1] == dtype, 'clean/final depth dtype differ')
        depths[model] = core.depth_result(model, dict(
            semantics=p['semantics'], native_dtype=dtype,
            clean=numpy_value(p['states']['clean']), final=numpy_value(p['states']['final']),
            object_mask=mask), THRESHOLDS)
    atomic = []
    for d in DETECTORS:
        clean, final = detections[d]['clean'], detections[d]['final']
        for z in DEPTHS:
            dep = depths[z]
            eligible = clean['matched'] and dep['clean_depth_valid']
            atomic.append(dict(sample_id=sid, dataset=dataset, source_split=split, detector=d, depth_model=z,
                               clean_match=clean, final_match=final, eligible_clean_pair=bool(eligible),
                               detection_success=not final['matched'], **dep,
                               joint_success=bool(not final['matched'] and dep['depth_success']) if eligible else None))
    return atomic


def summarize(atomic, sample_ids):
    expected = {(sid, d, z) for sid in sample_ids for d in DETECTORS for z in DEPTHS}
    keys = [(r['sample_id'], r['detector'], r['depth_model']) for r in atomic]
    need(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, 'exact case x pair atomic coverage')
    out = []
    for d in DETECTORS:
        for z in DEPTHS:
            valid = [r for r in atomic if r['detector'] == d and r['depth_model'] == z and r['eligible_clean_pair']]
            n = len(valid)
            dh = sum(int(r['detection_success']) for r in valid)
            zh = sum(int(r['depth_success']) for r in valid)
            jh = sum(int(r['joint_success']) for r in valid)
            out.append({'detector': d, 'depth_model': z, 'eligible_count': n, 'excluded_count': len(sample_ids) - n,
                        'detection_success_count': dh, 'depth_success_count': zh, 'joint_success_count': jh,
                        'DetASR': dh / n if n else None, 'DepASR': zh / n if n else None, 'JASR': jh / n if n else None})
    return out


def pair_table(atomic, sample_ids):
    rows = summarize(atomic, sample_ids)
    for row in rows:
        # Far-Shift: mean signed far-directed shift over the eligible clean-valid targets of this pair.
        vals = [r['FarShift'] for r in atomic if r['detector'] == row['detector']
                and r['depth_model'] == row['depth_model'] and r['eligible_clean_pair']]
        row['FarShift_mean'] = (sum(vals) / len(vals)) if vals else None
        for k in ('DetASR', 'DepASR', 'JASR'):
            row[k + '_percent'] = None if row[k] is None else 100.0 * row[k]
    return rows


def macro(rows, key, det=None, dep=None):
    vals = [r[key] for r in rows if r[key] is not None
            and (det is None or r['detector'] in det) and (dep is None or r['depth_model'] in dep)]
    return dict(value=(sum(vals) / len(vals)) if vals else None, valid_pairs=len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('evaluation_dir', nargs='+', help='one or more evaluation directories to merge')
    ap.add_argument('--detectors', default=','.join(DEFAULT_DETECTORS))
    ap.add_argument('--depths', default=','.join(DEFAULT_DEPTHS))
    ap.add_argument('--pool', default=None, help='external tensor pool root (older layouts without <eval>/tensor_pool)')
    ap.add_argument('--method-id', default=None, help='expected method_id bound in the payloads (optional check)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    global DETECTORS, DEPTHS, POOL_OVERRIDE
    POOL_OVERRIDE = Path(a.pool).absolute() if a.pool else None
    DETECTORS = tuple(a.detectors.split(','))
    DEPTHS = tuple(a.depths.split(','))
    roots = [Path(x).absolute() for x in a.evaluation_dir]
    out = Path(a.output).absolute()
    need(not out.exists(), 'output exists: ' + str(out))
    # older evaluation layouts (e.g. baseline200_20261016) store tensors inline without a tensor_pool
    det_manifests, det_idx, det_sids = index_manifests(roots, 'raw_predictions', DETECTORS, ('clean', 'final'))
    dep_manifests, dep_idx, dep_sids = index_manifests(roots, 'depth_raw', DEPTHS, ('clean', 'final'))
    need(det_sids == dep_sids, 'detection and depth sample sets differ')
    sample_ids = det_sids
    t0 = time.time()
    atomic = []
    for i, sid in enumerate(sample_ids):
        atomic.extend(compute_case(sid, det_idx, dep_idx, a.method_id))
        if (i + 1) % 25 == 0:
            print('[%d/%d] %.0fs' % (i + 1, len(sample_ids), time.time() - t0), flush=True)
    black_det = tuple(d for d in DETECTORS if d not in WHITE_DET)
    black_dep = tuple(z for z in DEPTHS if z not in WHITE_DEP)
    result = OrderedDict(schema=SCHEMA, label=a.label, evaluation_dirs=[str(r) for r in roots], hostname=socket.gethostname(),
                         created_unix=time.time(), thresholds=THRESHOLDS, numeric_policy=core.NUMERIC_POLICY,
                         detectors=DETECTORS, depth_models=DEPTHS, pair_count=len(DETECTORS) * len(DEPTHS),
                         pool_override=str(POOL_OVERRIDE) if POOL_OVERRIDE else None,
                         method_ids=sorted({m for mf in det_manifests for m in {r.get('method_id') for r in mf['entries']} if m}),
                         whitebox_models={'detectors': WHITE_DET, 'depth': WHITE_DEP},
                         manifests=[{'raw_predictions': file_sha(r / 'raw_predictions' / 'MANIFEST.json'),
                                     'depth_raw': file_sha(r / 'depth_raw' / 'MANIFEST.json')} for r in roots],
                         sample_count=len(sample_ids), domains=OrderedDict())
    for dom in DOMAINS:
        sids = [s for s in sample_ids if dom == 'combined' or s.startswith(dom + '_')]
        sset = set(sids)
        rows = pair_table([r for r in atomic if r['sample_id'] in sset], sids)
        result['domains'][dom] = OrderedDict(
            sample_count=len(sids), pairs=rows,
            macro_all_valid_pairs={k: macro(rows, k) for k in METRICS},
            macro_whitebox_pairs={k: macro(rows, k, WHITE_DET, WHITE_DEP) for k in METRICS},
            macro_blackbox_pairs={k: macro(rows, k, black_det, black_dep) for k in METRICS},
            macro_per_detector={d: macro(rows, 'DetASR', (d,)) for d in DETECTORS},
            macro_per_depth={z: macro(rows, 'DepASR', None, (z,)) for z in DEPTHS})
    result['atomic_sha256'] = hashlib.sha256(json.dumps(atomic, sort_keys=True, default=str).encode()).hexdigest()
    result['atomic_count'] = len(atomic)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    with out.with_suffix('.atomic.json').open('x', encoding='utf-8') as stream:
        json.dump(atomic, stream, ensure_ascii=False, default=str)
    print('wrote', out)
    for dom in DOMAINS:
        d = result['domains'][dom]
        print('%-9s N=%d macroJASR=%s (valid %d) white=%s black=%s' % (
            dom, d['sample_count'], d['macro_all_valid_pairs']['JASR']['value'],
            d['macro_all_valid_pairs']['JASR']['valid_pairs'],
            d['macro_whitebox_pairs']['JASR']['value'], d['macro_blackbox_pairs']['JASR']['value']))


if __name__ == '__main__':
    main()
