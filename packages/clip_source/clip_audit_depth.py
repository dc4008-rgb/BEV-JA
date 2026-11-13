#!/usr/bin/env python3
"""Recompute every depth metric from saved tensors; no model inference or image IO."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics
import struct
import sys
import time

sys.dont_write_bytecode = True
MODELS = ('monodepth2', 'dpt_large', 'depth_anything_v1', 'zoedepth')
SEMANTICS = {'monodepth2': 'mono_scale_ambiguous_depth', 'dpt_large': 'relative_inverse_depth',
             'depth_anything_v1': 'relative_inverse_depth', 'zoedepth': 'metric_depth'}
STATES = ('clean', 'zero', 'random', 'final')
PAIRS = (('clean', 'zero'), ('clean', 'random'), ('clean', 'final'), ('zero', 'final'))
R2_MANIFEST = '92af9063dbe07898eb562b0d324e9d1bd746467ce1edea408f9b5f8940f82c85'
CHILD_MANIFEST = 'a4ee2aab2ab40fe9c122c16bc05fd5a32cbd06286413384f4ad4086c26b863cb'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def bound(root, name, digest):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe artifact path')
    path = Path(root) / relative
    if any((Path(root) / Path(*relative.parts[:i])).is_symlink() for i in range(len(relative.parts) + 1)):
        raise ValueError('symlink evidence is forbidden')
    if not isinstance(digest, str) or len(digest) != 64 or not path.is_file() or sha(path) != digest:
        raise ValueError('missing or SHA-mismatched artifact: ' + str(path))
    return path


def bind_run_model_wrapper(summary, child, source_entries, expected_package_sha, training_source,
                           training_root=None, wrapper=None):
    """Close the evaluation -> actual training/model/wrapper provenance links."""
    sources = summary['source_bindings']
    wrapper = Path(wrapper) if wrapper is not None else Path(__file__).with_name('clip_evaluate.py')
    bound(wrapper.parent, wrapper.name, sources['wrapper_sha256'])
    spec = importlib.util.spec_from_file_location('_long500_bound_evaluator', wrapper)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    package = implementation.bind_long_package(training_source, expected_package_sha)
    bound(training_source, 'clip_evaluate.py', sources['wrapper_sha256'])
    if sources.get('clip_training_package_manifest_sha256') != expected_package_sha:
        raise ValueError('evaluation package differs from explicitly required clipping-sweep SHA')
    model_name = 'parent_v17/02_模型/model_manifest.json'
    model_digest = sources['model_manifest_sha256']
    if source_entries[model_name]['sha256'] != model_digest:
        raise ValueError('model manifest is not the frozen child implementation')
    bound(child, model_name, model_digest)
    run_path = bound(Path(training_root or summary['run']), 'summary.json', summary['run_summary_sha256'])
    run = read_json(run_path)
    method, trained = implementation.run_role(run, expected_package_sha)
    if summary.get('method') != method or summary.get('training_models') != list(trained):
        raise ValueError('evaluation method/training models differ from bound clipping training')
    if (summary.get('exploratory_posthoc') is not True
            or summary.get('selection_context') != implementation.SELECTION_CONTEXT):
        raise ValueError('evaluation must disclose post-evaluation exploratory selection')
    implementation.validate_clip_policy(summary.get('gradient_clipping'))
    if summary['gradient_clipping'] != run.get('gradient_clipping'):
        raise ValueError('evaluation clipping policy differs from bound training')
    state_manifest = read_json(bound(run_path.parent, 'state_evidence/MANIFEST.json', run['state_evidence_manifest_sha256']))
    implementation.validate_comparison_gate(run_path.parent, run, state_manifest)
    if summary.get('clipping_comparison_gate_sha256') != run.get('clipping_comparison_gate_sha256'):
        raise ValueError('evaluation clipping gate differs from bound training')
    if run.get('candidate_id') != summary['candidate_id']:
        raise ValueError('evaluation/training candidate identity differs')
    for key in ('model_manifest_sha256', 'training_manifest_sha256'):
        if run.get(key) != sources[key]:
            raise ValueError('evaluation/training source binding differs: ' + key)
    return {'run_summary_sha256': summary['run_summary_sha256'],
            'model_manifest_sha256': model_digest, 'wrapper_sha256': sources['wrapper_sha256'],
            'clip_training_package': package, 'method': method, 'updates': 500}


def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True,
                        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(header)
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def metric_functions(path):
    import numpy as np
    import torch
    import torch.nn.functional as F
    required = {'depth_metrics', 'depth_summary', 'percentile_interval'}
    functions = [node for node in ast.parse(Path(path).read_text(encoding='utf-8')).body
                 if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in functions} != required or len(functions) != 3:
        raise ValueError('required original metric functions unavailable')
    scope = dict(math=math, statistics=statistics, np=np, torch=torch, F=F)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
    return scope['depth_metrics'], scope['depth_summary']


def exact(expected, observed, where, counts):
    if type(expected) is not type(observed):
        raise ValueError('type mismatch at ' + where)
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            raise ValueError('keys mismatch at ' + where)
        for key in expected:
            exact(expected[key], observed[key], where + '.' + str(key), counts)
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(observed):
            raise ValueError('length mismatch at ' + where)
        for index, (left, right) in enumerate(zip(expected, observed)):
            exact(left, right, where + '[' + str(index) + ']', counts)
    else:
        counts['scalar_checks'] += 1
        equal = (math.isfinite(expected) and math.isfinite(observed) and struct.pack('!d', expected) == struct.pack('!d', observed)) if isinstance(expected, float) else expected == observed
        if not equal:
            raise ValueError('exact-value mismatch at %s: %r != %r' % (where, expected, observed))


def audit(args):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.run.resolve()
    summary = read_json(root / 'summary.json')
    if (summary.get('schema') != 'r2-joint-clip120-eight-model-development-evaluation-v1'
            or summary.get('status') != 'development_validation_complete'
            or summary.get('test_data_opened') is not False or summary.get('scene_count') != 40
            or summary.get('updates') != 500 or summary.get('training_budget_updates') != 500
            or summary.get('evaluation_forward_total') != 1280 or summary.get('checkpoint_selection') != 'fixed_final500'):
        raise ValueError('incomplete or incompatible eight-model evaluation')
    bindings = summary['artifact_bindings']
    counts = {'artifact_sha_checks': 0, 'raw_tensor_checks': 0, 'scalar_checks': 0, 'rows_checked': 0, 'summaries_checked': 0}
    # Only JSON and depth tensor evidence are opened, never saved RGB previews or image tensors.
    for name, digest in bindings.items():
        if name.startswith('depth_') or name in ('rendered_input_hashes.json', 'detection_summary.json', 'CLIP_PLAN.json'):
            bound(root, name, digest)
            counts['artifact_sha_checks'] += 1
    bound(root, 'CLIP_PLAN.json', summary['gradient_clipping']['grid_sha256'])
    if bindings.get('CLIP_PLAN.json') != summary['gradient_clipping']['grid_sha256']:
        raise ValueError('evaluation clipping plan artifact binding differs')
    sources = summary['source_bindings']
    if sources.get('r2_manifest_sha256') != R2_MANIFEST or sources.get('child_manifest_sha256') != CHILD_MANIFEST:
        raise ValueError('accepted R2/child source identities differ')
    metric_path = bound(args.child, sources['depth_evaluator_relative_path'], sources['depth_evaluator_sha256'])
    bound(args.r2_source, 'PACKAGE_MANIFEST.json', sources['r2_manifest_sha256'])
    bound(args.r2_source, 'evaluate.py', sources['r2_evaluate_sha256'])
    child_manifest = read_json(bound(args.child, 'PACKAGE_MANIFEST.json', sources['child_manifest_sha256']))
    source_entries = {row['path']: row for row in child_manifest['files']}
    identity_bindings = bind_run_model_wrapper(summary, args.child, source_entries,
                                              args.training_package_manifest_sha256, args.training_source)
    bound(args.training_source, 'clip_audit_depth.py', sha(Path(__file__)))
    if source_entries[sources['depth_evaluator_relative_path']]['sha256'] != sources['depth_evaluator_sha256']:
        raise ValueError('metric source is not the frozen child implementation')
    csv_name = 'parent_v17/01_公共数据集/runtime_v1/train_validation_manifest.csv'
    if source_entries[csv_name]['sha256'] != sources['training_manifest_sha256']:
        raise ValueError('training/validation manifest binding differs')
    with bound(args.child, csv_name, sources['training_manifest_sha256']).open(newline='', encoding='utf-8') as stream:
        frozen_scenes = sorted((row['dataset'], row['sample_id']) for row in csv.DictReader(stream) if row['split'] == 'validation')
    metrics, summarize = metric_functions(metric_path)
    raw_manifest = read_json(bound(root, 'depth_raw/MANIFEST.json', bindings['depth_raw/MANIFEST.json']))
    if raw_manifest.get('schema') != 'joint-depth-raw-manifest-v1' or raw_manifest.get('model_forward_count') != 640:
        raise ValueError('depth raw manifest identity/coverage differs')
    exact(list(STATES), raw_manifest['states'], 'states', counts)
    exact([list(pair) for pair in PAIRS], raw_manifest['comparisons'], 'comparisons', counts)
    entries = raw_manifest['entries']
    scenes = [(row['dataset'], row['sample_id']) for row in summary['validation_scenes']]
    if scenes != frozen_scenes:
        raise ValueError('evaluation scenes differ from the frozen validation manifest')
    if (len(scenes) != 40 or len(set(scenes)) != 40 or Counter(domain for domain, _ in scenes) != {'kitti': 20, 'nuscenes': 20}
            or any(row['split'] != 'validation' for row in summary['validation_scenes'])):
        raise ValueError('invalid frozen validation scene coverage')
    expected_keys = {(model, *scene) for model in MODELS for scene in scenes}
    actual_keys = [(entry['model'], entry['dataset'], entry['sample_id']) for entry in entries]
    if len(entries) != 160 or len(set(actual_keys)) != 160 or set(actual_keys) != expected_keys:
        raise ValueError('missing or duplicated depth raw records')
    observed_rows = read_json(bound(root, 'depth_per_sample.json', bindings['depth_per_sample.json']))
    observed_index = {(row['model'], row['dataset'], row['sample_id'], row['reference_state'], row['evaluated_state']): row for row in observed_rows}
    if len(observed_rows) != 640 or len(observed_index) != 640:
        raise ValueError('missing or duplicate metric rows')
    detection_rows = read_json(bound(root, 'rendered_input_hashes.json', bindings['rendered_input_hashes.json']))['rows']
    input_index = {(row['dataset'], row['sample_id'], row['state']): row for row in detection_rows if row['model'] == 'fasterrcnn'}
    recomputed = []
    for entry in entries:
        raw = torch.load(bound(root, entry['path'], entry['sha256']), map_location='cpu', weights_only=True)
        for key in ('model', 'sample_id', 'dataset', 'split', 'semantics'):
            exact(entry[key], raw[key], 'raw.' + key, counts)
        model = raw['model']
        if raw['split'] != 'validation' or raw['semantics'] != SEMANTICS[model] or set(raw['states']) != set(STATES):
            raise ValueError('raw depth identity, split, or semantics differs')
        exact(entry['object_mask'], raw['object_mask'], 'mask_binding', counts)
        mask_binding = raw['object_mask']
        mask = torch.load(bound(root, mask_binding['path'], mask_binding['sha256']), map_location='cpu', weights_only=True)
        if tensor_sha(mask) != mask_binding['tensor_sha256'] or mask.shape != (1, 1, 320, 1024) or not bool(torch.isfinite(mask).all()):
            raise ValueError('invalid bound object mask')
        counts['raw_tensor_checks'] += 1
        for state in STATES:
            value = raw['states'][state]
            if value.shape != mask.shape or not value.is_floating_point() or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError('negative/nonfinite/invalid raw depth output')
            if tensor_sha(value) != entry['tensor_sha256'][state]:
                raise ValueError('raw depth typed tensor digest differs')
            counts['raw_tensor_checks'] += 1
            ref = input_index[(raw['dataset'], raw['sample_id'], 'zero' if state == 'clean' else state)]
            expected = ref['clean_input_sha256' if state == 'clean' else 'rendered_input_sha256']
            exact(expected, raw['input_sha256'][state], 'paired_rgb_hash', counts)
            if state != 'clean':
                exact(ref['support_tensor_sha256'], raw['support_sha256'][state], 'paired_support_hash', counts)
        for reference, evaluated in PAIRS:
            row = {'method': summary['method'], 'model': model, 'sample_id': raw['sample_id'],
                   'dataset': raw['dataset'], 'split': 'validation', 'reference_state': reference,
                   'evaluated_state': evaluated, 'used_in_training': model in summary['training_models']}
            row.update(metrics(raw['states'][evaluated], raw['states'][reference], mask, semantics=SEMANTICS[model]))
            key = (model, raw['dataset'], raw['sample_id'], reference, evaluated)
            exact(row, observed_index[key], 'per_sample.' + repr(key), counts)
            counts['rows_checked'] += 1
            recomputed.append(row)
    depth_summary = read_json(bound(root, 'depth_summary.json', summary['depth_summary_sha256']))
    if depth_summary.get('raw_manifest_sha256') != sha(root / 'depth_raw/MANIFEST.json'):
        raise ValueError('depth summary raw manifest binding differs')
    new_summaries = []
    for model in MODELS:
        for reference, evaluated in PAIRS:
            for domain in ('kitti', 'nuscenes', 'combined'):
                subset = [row for row in recomputed if row['model'] == model and row['reference_state'] == reference
                          and row['evaluated_state'] == evaluated and (domain == 'combined' or row['dataset'] == domain)]
                value = summarize(subset, 1000, 20260919)
                value.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated,
                             depth_representation=SEMANTICS[model], reference='same_model_prediction_on_' + reference + '_scene')
                new_summaries.append(value)
    exact(new_summaries, depth_summary['summaries'], 'depth_summaries', counts)
    counts['summaries_checked'] = len(new_summaries)
    return {'status': 'depth_raw_audit_passed', 'tolerance': 0, 'counts': counts,
            'candidate_id': summary['candidate_id'], 'updates': 500, 'checkpoint_selection': 'fixed_final500',
            'identity_bindings': identity_bindings,
            'selection_context': summary['selection_context'], 'exploratory_posthoc': True,
            'evaluation_summary_sha256': sha(root / 'summary.json'), 'depth_metric_source_sha256': sha(metric_path),
            'audit_source_sha256': sha(Path(__file__)), 'no_model_loading_or_inference': True,
            'no_source_images_opened': True, 'test_data_opened': False,
            'metric_execution': 'exact original AST function definitions from SHA-bound evaluator.py',
            'scope': 'depth evaluation-output tensors to 640 metric rows and 48 summaries; detector metrics audited separately'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r2-source', 'child', 'run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--training-package-manifest-sha256', required=True,
                        help='Required actual SHA of the newly sealed cap120 followup PACKAGE_MANIFEST.json')
    parser.add_argument('--training-source', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise ValueError('live hostname mismatch')
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError('fresh audit output with existing parent is required')
    if any(path.is_symlink() for path in (args.output, *args.output.absolute().parents)):
        raise ValueError('audit output symlinks forbidden')
    args.output = args.output.resolve()
    if args.run.resolve() == args.output or args.run.resolve() in args.output.parents:
        raise ValueError('audit output must be outside evaluated evidence')
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        result = audit(args)
        result['seconds'] = time.monotonic() - started
        write_json(args.output / 'summary.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__,
                                                'error': str(error), 'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
