#!/usr/bin/env python3
"""Exact scientific-state comparison, separate from archive/container hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

PHASES = ('before_denoise', 'pre_backward', 'pre_update', 'post_update')
IDENTITY = 'meeting_full_repeat_20260924_r2'
MODELS = ('fasterrcnn', 'ssd300', 'maskrcnn', 'fcos')


def read_json(path):
    def reject(value):
        raise ValueError('non-finite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(canonical({'dtype': str(value.dtype), 'shape': list(value.shape)}))
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def bound_file(root, relative, digest=None):
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('unsafe relative evidence path: ' + str(relative))
    if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(len(relative.parts) + 1)):
        raise ValueError('symlink evidence rejected: ' + str(relative))
    path = root / relative
    if not path.is_file() or (digest is not None and sha(path) != digest):
        raise ValueError('missing or SHA-mismatched evidence: ' + str(path))
    return path


def tree_difference(left, right, path='$', stats=None):
    """First traversal-order difference, preserving dtype, shape and original bits."""
    import torch
    if stats is None:
        stats = {'tensors_checked': 0, 'scalars_checked': 0}
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return {'path': path, 'reason': 'tensor_type'}
        stats['tensors_checked'] += 1
        if left.dtype != right.dtype or left.shape != right.shape:
            return {'path': path, 'reason': 'tensor_dtype_or_shape',
                    'left': {'dtype': str(left.dtype), 'shape': list(left.shape)},
                    'right': {'dtype': str(right.dtype), 'shape': list(right.shape)}}
        l, r = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
        if (l.is_floating_point() or l.is_complex()) and (not torch.isfinite(l).all() or not torch.isfinite(r).all()):
            return {'path': path, 'reason': 'nonfinite_tensor'}
        lb, rb = l.reshape(-1).view(torch.uint8), r.reshape(-1).view(torch.uint8)
        if torch.equal(lb, rb):
            return None
        byte_index = int(torch.nonzero(lb != rb, as_tuple=False)[0, 0])
        element_index = byte_index // l.element_size()
        result = {'path': path, 'reason': 'tensor_bytes', 'first_different_flat_element': element_index,
                  'left_sha256': tensor_sha(l), 'right_sha256': tensor_sha(r),
                  'left_value': l.reshape(-1)[element_index].item(),
                  'right_value': r.reshape(-1)[element_index].item(),
                  'numerically_different_elements': int((l != r).sum()), 'elements': l.numel()}
        if l.numel() and not l.is_complex():
            result['max_abs_difference'] = float((l.to(torch.float64) - r.to(torch.float64)).abs().max())
        return result
    if type(left) is not type(right):
        return {'path': path, 'reason': 'type', 'left_type': type(left).__name__, 'right_type': type(right).__name__}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {'path': path, 'reason': 'keys', 'left_keys': sorted(map(repr, left)), 'right_keys': sorted(map(repr, right))}
        for key in left:
            difference = tree_difference(left[key], right[key], path + '.' + str(key), stats)
            if difference:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {'path': path, 'reason': 'length', 'left_length': len(left), 'right_length': len(right)}
        for i, (l, r) in enumerate(zip(left, right)):
            difference = tree_difference(l, r, '%s[%d]' % (path, i), stats)
            if difference:
                return difference
        return None
    if left is not None and not isinstance(left, (str, bool, int, float, bytes)):
        raise TypeError('unsupported scientific evidence type: ' + str(type(left)))
    stats['scalars_checked'] += 1
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            return {'path': path, 'reason': 'nonfinite_scalar'}
        equal = struct.pack('!d', left) == struct.pack('!d', right)
    else:
        equal = left == right
    return None if equal else {'path': path, 'reason': 'value', 'left': left, 'right': right}


def expected_order():
    result = [('initialization', 0)]
    for outer in range(4):
        result.append(('outer_start', 0))
        for step in range(outer * 25 + 1, outer * 25 + 26):
            result.extend((phase, step) for phase in PHASES)
        result.append(('outer_end', 0))
    return result


def load_training(run):
    run = Path(run)
    summary = read_json(bound_file(run, 'summary.json'))
    if (summary.get('status') != 'train_optimization_complete' or summary.get('executed_updates') != 100
            or summary.get('test_data_opened') is not False):
        raise ValueError('only complete train-only final100 runs are eligible: ' + str(run))
    manifest_path = bound_file(run, 'state_evidence/MANIFEST.json', summary.get('state_evidence_manifest_sha256'))
    manifest = read_json(manifest_path)
    if (manifest.get('schema') != 'meeting-full-state-manifest-v1' or manifest.get('status') != 'complete'
            or manifest.get('executed_updates') != 100 or manifest.get('candidate_id') != summary.get('candidate_id')):
        raise ValueError('state manifest completion/identity differs')
    entries = manifest.get('entries', [])
    if [(e.get('phase'), e.get('global_step')) for e in entries] != expected_order():
        raise ValueError('exact ordered 409-state complete trajectory required')
    if len({e['path'] for e in entries}) != len(entries):
        raise ValueError('duplicated training-state paths')
    for index, entry in enumerate(entries, 1):
        if entry.get('index') != index:
            raise ValueError('nonsequential state manifest index')
        path = bound_file(run, entry['path'], entry['sha256'])
        if path.stat().st_size != entry.get('bytes'):
            raise ValueError('state evidence size mismatch')
    return summary, entries


def load_state(run, entry):
    import torch
    from evidence import fingerprint
    value = torch.load(bound_file(run, entry['path'], entry['sha256']), map_location='cpu', weights_only=True)
    if (value.get('schema') != 'meeting-full-state-v1' or not isinstance(value.get('state'), dict)
            or value.get('metadata', {}).get('phase') != entry['phase']
            or value.get('metadata', {}).get('global_step') != entry['global_step']):
        raise ValueError('state payload schema/phase/step differs')
    if fingerprint(value) != entry.get('scientific_fingerprint'):
        raise ValueError('state scientific fingerprint differs')
    return value


def compare_training(left_run, right_run):
    left_run, right_run = Path(left_run), Path(right_run)
    ls, le = load_training(left_run)
    rs, re = load_training(right_run)
    stats = {'tensors_checked': 0, 'scalars_checked': 0, 'states_checked': 0}
    result = {'left_run': str(left_run), 'right_run': str(right_run), 'status': 'exact',
              'comparison': 'original dtype/shape/bits; no tolerance or decimal rounding', 'counts': stats}
    for field in ('candidate_id', 'protocol_sha256', 'package_manifest_sha256', 'training_manifest_sha256',
                  'model_manifest_sha256', 'seed'):
        if field not in ls or field not in rs:
            raise ValueError('missing scientific run identity: ' + field)
        difference = tree_difference(ls[field], rs[field], '$.summary.' + field, stats)
        if difference:
            return dict(result, status='different', first_difference=difference)
    # Runtime identity contains a dedicated stable subset. PIDs, elapsed time and
    # run paths remain recorded but are not scientific values to compare.
    lr = read_json(bound_file(left_run, 'runtime_provenance.json', ls.get('runtime_provenance_sha256')))
    rr = read_json(bound_file(right_run, 'runtime_provenance.json', rs.get('runtime_provenance_sha256')))
    if not isinstance(lr.get('repeatability_identity'), dict) or not isinstance(rr.get('repeatability_identity'), dict):
        raise ValueError('runtime must record repeatability_identity without ephemeral metadata')
    difference = tree_difference(lr['repeatability_identity'], rr['repeatability_identity'], '$.runtime.repeatability_identity', stats)
    if difference:
        return dict(result, status='different', first_difference=difference)
    for lentry, rentry in zip(le, re):
        lvalue, rvalue = load_state(left_run, lentry), load_state(right_run, rentry)
        stats['states_checked'] += 1
        difference = tree_difference(lvalue, rvalue, '$.state_payload', stats)
        if difference:
            return dict(result, status='different', first_difference=dict(difference,
                        state_index=lentry['index'], phase=lentry['phase'], global_step=lentry['global_step']))
    import torch
    for name in ('zero_step_patch.pt', 'checkpoint_0100.pt', 'final_patch.pt'):
        lp = torch.load(bound_file(left_run, name), map_location='cpu', weights_only=True)
        rp = torch.load(bound_file(right_run, name), map_location='cpu', weights_only=True)
        # Canonical patch tensors matter; archive-bound metadata is intentionally
        # excluded, since the training states above are already checked in full.
        for value in (lp, rp):
            if value.get('schema') != 'road_patch_rgb_v1' or 'canonical_rgb' not in value:
                raise ValueError('missing canonical patch schema/tensor: ' + name)
        keys = ('schema', 'renderer_contract', 'canonical_rgb', 'raw_generator_rgb')
        difference = tree_difference({k: lp[k] for k in keys if k in lp}, {k: rp[k] for k in keys if k in rp}, '$.' + name, stats)
        if difference:
            return dict(result, status='different', first_difference=difference)
    return dict(result, first_difference=None)


def compare_evaluations(left, right):
    import torch
    left, right = Path(left), Path(right)
    ls, rs = read_json(bound_file(left, 'summary.json')), read_json(bound_file(right, 'summary.json'))
    for root, summary in ((left, ls), (right, rs)):
        if (summary.get('status') != 'development_validation_complete' or summary.get('updates') != 100
                or summary.get('evaluation_forward_total') != 640 or summary.get('test_data_opened') is not False):
            raise ValueError('complete frozen final100 four-detector validation is required')
        bindings = summary.get('artifact_bindings')
        if not isinstance(bindings, dict) or not {'per_sample.json', 'rendered_input_hashes.json',
                                                  'geometry_audit.json', 'raw_predictions/MANIFEST.json'}.issubset(bindings):
            raise ValueError('evaluation artifact bindings missing')
        for name, digest in bindings.items():
            bound_file(root, name, digest)
    stats = {'tensors_checked': 0, 'scalars_checked': 0, 'prediction_files_checked': 0}
    result = {'status': 'exact', 'counts': stats, 'left_evaluation': str(left), 'right_evaluation': str(right)}
    fields = ('protocol_sha256', 'package_manifest_sha256', 'evaluation_models', 'control_tensor_sha256',
              'parent_metric_engine_source_sha256', 'summaries', 'zero_to_final')
    for field in fields:
        difference = tree_difference(ls[field], rs[field], '$.summary.' + field, stats)
        if difference:
            return dict(result, status='different', first_difference=difference)
    for name in ('per_sample.json', 'rendered_input_hashes.json', 'geometry_audit.json'):
        difference = tree_difference(read_json(bound_file(left, name)), read_json(bound_file(right, name)), '$.' + name, stats)
        if difference:
            return dict(result, status='different', first_difference=difference)
    lm, rm = read_json(bound_file(left, 'raw_predictions/MANIFEST.json')), read_json(bound_file(right, 'raw_predictions/MANIFEST.json'))
    for manifest in (lm, rm):
        if len(manifest.get('entries', [])) != 160 or manifest.get('model_forward_count') != 640:
            raise ValueError('all 640 raw box predictions are required')
        identities = [(e.get('model'), e.get('dataset'), e.get('sample_id')) for e in manifest['entries']]
        if len(set(identities)) != 160 or any(sum(e[0] == m for e in identities) != 40 for m in MODELS):
            raise ValueError('raw prediction coverage differs')
    for le, re in zip(lm['entries'], rm['entries']):
        if [le.get(k) for k in ('model', 'sample_id', 'dataset')] != [re.get(k) for k in ('model', 'sample_id', 'dataset')]:
            raise ValueError('raw prediction ordering/identity differs')
        lv = torch.load(bound_file(left, le['path'], le['sha256']), map_location='cpu', weights_only=True)
        rv = torch.load(bound_file(right, re['path'], re['sha256']), map_location='cpu', weights_only=True)
        for payload, entry in ((lv, le), (rv, re)):
            for state in ('clean', 'zero', 'random', 'final'):
                if not isinstance(payload.get(state), dict) or set(payload[state]) != {'boxes', 'labels', 'scores'}:
                    raise ValueError('missing complete raw prediction tensors')
                if {k: tensor_sha(v) for k, v in payload[state].items()} != entry.get('tensor_sha256', {}).get(state):
                    raise ValueError('raw prediction typed tensor SHA differs')
        stats['prediction_files_checked'] += 1
        difference = tree_difference(lv, rv, '$.raw_predictions.' + le['model'] + '.' + le['sample_id'], stats)
        if difference:
            return dict(result, status='different', first_difference=difference)
    return dict(result, first_difference=None)


def batch_summary(runs, evaluations=None):
    if len(runs) < 2 or len(runs) > 5 or len(set(map(lambda x: str(Path(x).resolve()), runs))) != len(runs):
        raise ValueError('supply two to five distinct fresh runs in execution order')
    if evaluations and len(evaluations) != len(runs):
        raise ValueError('one evaluation per run required')
    pairs = []
    for i in range(1, len(runs)):
        item = {'reference_run_number': 1, 'run_number': i + 1,
                'training': compare_training(runs[0], runs[i])}
        if evaluations:
            item['evaluation'] = compare_evaluations(evaluations[0], evaluations[i])
        pairs.append(item)
    exact = all(p['training']['status'] == 'exact' and
                (not evaluations or p['evaluation']['status'] == 'exact') for p in pairs)
    result = {'schema': 'meeting-full-repeat-comparison-v1', 'runs': [str(x) for x in runs],
            'evaluations': [str(x) for x in evaluations] if evaluations else [],
            'run_count': len(runs), 'pairs': pairs,
            'status': ('five_run_acceptance_pass' if len(runs) == 5 and evaluations and exact
                       else 'partial_exact_pending_five_complete_runs' if exact else 'difference_found'),
            'all_supplied_runs_retained': True, 'tolerance': 0,
            'acceptance_scope': 'same frozen code/protocol/software/GPU and seed; not cross-hardware or cross-seed'}
    if evaluations:
        result['individual_final_results'] = [
            {'run_number': index + 1, 'evaluation': str(path),
             'combined_final': [row for row in read_json(Path(path) / 'summary.json')['summaries']
                                if row.get('dataset') == 'combined' and row.get('state') == 'final']}
            for index, path in enumerate(evaluations)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--evaluations', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('comparison reports cannot be overwritten')
    result = batch_summary(args.runs, args.evaluations)
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'run_count': result['run_count']}, ensure_ascii=False))
    if result['status'] == 'difference_found':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
