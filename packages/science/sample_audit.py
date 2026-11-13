#!/usr/bin/env python3
"""Recompute same-sample raw metrics and merge shards by sample, never shard AP.

No victim models or source images are loaded. Raw native predictions, target
boxes, masks, artifact identities and source-bound original metric definitions
are checked. This does not establish GPU attack efficacy independently.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import sample_evaluate as evaluation
from artifact_io import load_torch_lossless, materialize_tensors, checked_pool, tensor_blob_bindings

DETECTION_MODELS = evaluation.DETECTION_MODELS
DEPTH_MODELS = evaluation.DEPTH_MODELS
STATES = evaluation.STATES
COMPARISONS = evaluation.COMPARISONS
SEMANTICS = evaluation.SEMANTICS
SEED = evaluation.SEED
METRIC_SHA256 = evaluation.METRIC_SHA256
IOU_ATOL = 1e-7


def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True,
                        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def exact(actual, expected, label='value'):
    if type(actual) is not type(expected):
        raise ValueError('type mismatch: ' + label)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError('field mismatch: ' + label)
        for key in actual:
            exact(actual[key], expected[key], label + '.' + str(key))
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError('length mismatch: ' + label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, label + '[%d]' % index)
    elif isinstance(actual, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or actual != expected:
            raise ValueError('numeric mismatch: ' + label)
    elif actual != expected:
        raise ValueError('value mismatch: ' + label)


def load_metrics(path):
    import statistics
    import numpy as np
    import torch
    import torch.nn.functional as F
    if evaluation.sha(path) != METRIC_SHA256:
        raise ValueError('original metric source SHA differs')
    names = {'box_iou', 'detection_candidates', 'average_precision', 'tnr_detection',
             'detection_summary', 'depth_metrics', 'depth_summary', 'percentile_interval'}
    nodes = [node for node in ast.parse(Path(path).read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != names:
        raise ValueError('missing or ambiguous original metric definitions')
    namespace = dict(math=math, statistics=statistics, np=np, torch=torch, F=F)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def unique_index(rows, keys, expected, label):
    indexed = {tuple(row[key] for key in keys): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError('missing, duplicate or unexpected ' + label)
    return indexed


def sample_binding(case, state, random_binding):
    result = {key: value for key, value in case.items() if key != 'patches'}
    result['patch_binding'] = (None if state == 'clean' else random_binding if state == 'random' else case['patches'][state])
    return result


def check_case(case, sample, plan):
    for key in ('sample_id', 'case_id', 'dataset', 'source_split', 'source_sha256'):
        exact(case[key], sample[key], 'case.' + key)
    from case_owners200 import resolve_case_owner, case_origin_binding
    owner_plan = resolve_case_owner(plan, sample)
    exact(case['plan_sha256'], owner_plan['_plan_sha256'], 'case.original_training_plan')
    exact(case['evaluation_plan_sha256'], plan['_plan_sha256'], 'case.current_evaluation_plan')
    exact(case['training_origin'], case_origin_binding(plan, sample), 'case.explicit_training_origin')
    evaluation.require_sha(case['train_summary_sha256'])
    if set(case['patches']) != {'zero', 'final'}:
        raise ValueError('own zero/final patch identities missing')
    for state, patch in case['patches'].items():
        for key in ('sample_id', 'case_id', 'source_sha256'):
            exact(patch[key], sample[key], 'patch.' + key)
        for key in ('sha256', 'tensor_sha256'):
            evaluation.require_sha(patch[key])


def prediction_valid(value):
    import torch
    if not isinstance(value, dict) or set(value) != {'boxes', 'labels', 'scores'}:
        raise ValueError('raw detection fields differ')
    boxes, scores, labels = value['boxes'], value['scores'], value['labels']
    if (boxes.dtype != torch.float32 or scores.dtype != torch.float32 or labels.dtype != torch.int64
            or boxes.shape != (scores.numel(), 4) or labels.shape != scores.shape or scores.ndim != 1
            or not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all())
            or not bool(((scores >= 0) & (scores <= 1)).all())):
        raise ValueError('raw detection dtype/shape/finite/range differs')


def compare_candidates(rebuilt, saved, label, tolerances):
    if len(rebuilt) != len(saved):
        raise ValueError('candidate count mismatch: ' + label)
    for index, (cpu, stored) in enumerate(zip(rebuilt, saved)):
        exact(set(cpu), set(stored), label + '.fields')
        exact(cpu['score'], stored['score'], label + '.score')
        exact(cpu['label'], stored['label'], label + '.label')
        left, right = cpu['iou'], stored['iou']
        if left == right:
            continue
        if (not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > IOU_ATOL
                or (left > 0) != (right > 0) or (left >= .5) != (right >= .5)):
            raise ValueError('true IoU/AP50 membership mismatch: ' + label)
        tolerances.append(dict(field=label, index=index, saved_iou=right, recomputed_iou=left))


def domains(rows):
    return [domain for domain in ('kitti', 'nuscenes', 'combined')
            if domain == 'combined' or any(row['dataset'] == domain for row in rows)]


def detection_summaries(metrics, rows):
    results = []
    for model in DETECTION_MODELS:
        index = {(row['dataset'], row['sample_id'], row['state']): row for row in rows if row['model'] == model}
        for reference, evaluated in COMPARISONS:
            selected = []
            for row in rows:
                if row['model'] != model or row['state'] != evaluated:
                    continue
                current = dict(row)
                if reference == 'zero':
                    zero = index[(row['dataset'], row['sample_id'], 'zero')]
                    current.update(clean_candidates=zero['adversarial_candidates'], clean_car_candidates=zero['adversarial_car_candidates'])
                selected.append(current)
            for domain in domains(selected):
                subset = [row for row in selected if domain == 'combined' or row['dataset'] == domain]
                value = metrics.detection_summary(subset, 1000, SEED, allow_zero_reference=True)
                value.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated,
                             aggregation='global_candidate_score_ranking_with_one_target_per_sample',
                             tnr_undefined_reason=None if value['tnr_defined'] else 'reference_AP50_is_zero')
                results.append(value)
    return results


def depth_summaries(metrics, rows):
    results = []
    for model in DEPTH_MODELS:
        for reference, evaluated in COMPARISONS:
            selected = [row for row in rows if row['model'] == model and row['reference_state'] == reference and row['evaluated_state'] == evaluated]
            for domain in domains(selected):
                subset = [row for row in selected if domain == 'combined' or row['dataset'] == domain]
                value = metrics.depth_summary(subset, 1000, SEED)
                value.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated,
                             depth_representation=SEMANTICS[model], aggregation='per_sample_macro_mean_primary',
                             reference='same_model_prediction_on_' + reference + '_scene')
                results.append(value)
    return results


def reconstruct_detection(root, plan, cases, metrics, random_binding, counts, tolerances, pool_root):
    import torch
    count = len(plan['samples'])
    manifest = evaluation.read_json(root / 'raw_predictions/MANIFEST.json')
    expected_entries = [(model, sample['sample_id']) for model in DETECTION_MODELS for sample in plan['samples']]
    entries = unique_index(manifest['entries'], ('model', 'sample_id'), expected_entries, 'detector raw cases')
    exact(manifest['model_forward_count'], count * 16, 'detector forward count')
    exact(manifest['states'], list(STATES), 'detector states')
    saved = evaluation.read_json(root / 'per_sample.json')
    keys = ('model', 'sample_id', 'state')
    expected_rows = [(model, sample['sample_id'], state) for model in DETECTION_MODELS for sample in plan['samples'] for state in STATES[1:]]
    saved_index = unique_index(saved, keys, expected_rows, 'detector metric rows')
    geometry = evaluation.read_json(root / 'geometry_audit.json')
    exact(geometry['status'], 'pass', 'geometry status')
    geometry_index = unique_index(geometry['checks'], keys, expected_rows, 'detector geometry rows')
    input_rows = evaluation.read_json(root / 'rendered_input_hashes.json')['rows']
    expected_inputs = [(model, sample['sample_id'], state) for model in DETECTION_MODELS for sample in plan['samples'] for state in STATES]
    inputs = unique_index(input_rows, keys, expected_inputs, 'detector rendered input identities')
    rebuilt, boxes = [], {}
    for model, sample_id in expected_entries:
        entry, case = entries[(model, sample_id)], cases[sample_id]
        sample = next(item for item in plan['samples'] if item['sample_id'] == sample_id)
        for key, value in case.items():
            exact(entry[key], value, 'detector entry.' + key)
        raw_path = evaluation.checked_file(root, entry['path'])
        exact(evaluation.sha(raw_path), entry['sha256'], 'raw detector file SHA')
        payload = load_torch_lossless(raw_path, pool_root=pool_root)
        for key, value in case.items():
            exact(payload[key], value, 'raw detector.' + key)
        exact(payload['model'], model, 'raw model')
        exact(payload['split'], sample['source_split'], 'raw original split')
        box = payload['target_box']
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box) or not (0 <= box[0] < box[2] <= 1024 and 0 <= box[1] < box[3] <= 320):
            raise ValueError('invalid inserted target box')
        if sample_id in boxes:
            exact(box, boxes[sample_id], 'same target box across models')
        boxes[sample_id] = box
        for state in STATES:
            prediction_valid(payload[state])
            for field, tensor in payload[state].items():
                exact(tensor_sha(tensor), entry['tensor_sha256'][state][field], 'raw detector typed SHA')
                counts['typed_sha_checks'] += 1
            input_row = inputs[(model, sample_id, state)]
            for key, value in sample_binding(case, state, random_binding).items():
                exact(input_row[key], value, 'detector input.' + key)
            exact(payload['input_sha256'][state], input_row['rendered_input_sha256'], 'raw rendered input SHA')
            exact(input_row['clean_input_sha256'], payload['input_sha256']['clean'], 'same-scene clean reference input')
            common_input = inputs[(DETECTION_MODELS[0], sample_id, state)]
            exact(input_row['rendered_input_sha256'], common_input['rendered_input_sha256'], 'same input across detector models')
            exact(input_row['support_tensor_sha256'], common_input['support_tensor_sha256'], 'same support across detector models')
            if state != 'clean':
                gt = geometry_index[(model, sample_id, state)]
                exact(gt['target_box'], box, 'raw box vs geometry')
                exact(gt['split'], sample['source_split'], 'geometry original split')
                exact(gt['road_quad'], torch.as_tensor(json.loads(sample['road_quad_norm'])).tolist(), 'bound road quadrilateral')
                if not gt['checks'] or not all(value is True for value in gt['checks'].values()):
                    raise ValueError('stored geometry invariant did not pass')
                exact(gt['sample_binding'], sample_binding(case, state, random_binding), 'detector geometry Patch owner')
        clean = metrics.detection_candidates(payload['clean'], box, (3, 6, 8))
        clean_car = metrics.detection_candidates(payload['clean'], box, (3,))
        for state in STATES[1:]:
            row = dict(sample_binding(case, state, random_binding), method=evaluation.IDENTITY, model=model,
                       split=sample['source_split'], state=state, used_in_training=model in DETECTION_MODELS[:2],
                       clean_candidates=clean, clean_car_candidates=clean_car,
                       adversarial_candidates=metrics.detection_candidates(payload[state], box, (3, 6, 8)),
                       adversarial_car_candidates=metrics.detection_candidates(payload[state], box, (3,)))
            stored = saved_index[(model, sample_id, state)]
            fields = ('clean_candidates', 'clean_car_candidates', 'adversarial_candidates', 'adversarial_car_candidates')
            exact({key: value for key, value in row.items() if key not in fields},
                  {key: value for key, value in stored.items() if key not in fields}, 'detector metric identity')
            for field in fields:
                compare_candidates(row[field], stored[field], sample_id + '.' + model + '.' + state + '.' + field, tolerances)
            rebuilt.append(row)
            counts['detection_rows_checked'] += 1
    summaries = detection_summaries(metrics, rebuilt)
    saved_summary = evaluation.read_json(root / 'detection_summary.json')
    exact(summaries, saved_summary['summaries'], 'detector full summaries')
    exact(saved_summary['records'], count * 12, 'detector records')
    exact(saved_summary['model_forward_count'], count * 16, 'detector summary forwards')
    return rebuilt, summaries, boxes, inputs


def reconstruct_depth(root, plan, cases, metrics, random_binding, counts, boxes, detector_inputs, pool_root):
    import torch
    count = len(plan['samples'])
    manifest = evaluation.read_json(root / 'depth_raw/MANIFEST.json')
    expected_entries = [(model, sample['sample_id']) for model in DEPTH_MODELS for sample in plan['samples']]
    entries = unique_index(manifest['entries'], ('model', 'sample_id'), expected_entries, 'depth raw cases')
    exact(manifest['model_forward_count'], count * 16, 'depth forward count')
    exact(manifest['metric_input_stage'], 'evaluation_output', 'native output stage')
    exact(manifest['states'], list(STATES), 'depth states')
    saved = evaluation.read_json(root / 'depth_per_sample.json')
    keys = ('model', 'sample_id', 'reference_state', 'evaluated_state')
    expected_rows = [(model, sample['sample_id'], ref, adv) for model in DEPTH_MODELS for sample in plan['samples'] for ref, adv in COMPARISONS]
    saved_index = unique_index(saved, keys, expected_rows, 'depth metric rows')
    input_rows = evaluation.read_json(root / 'depth_input_hashes.json')
    exact(input_rows['all_equal_to_detection_inputs'], True, 'depth/detection input identity flag')
    expected_inputs = [(model, sample['sample_id'], state) for model in DEPTH_MODELS for sample in plan['samples'] for state in STATES]
    inputs = unique_index(input_rows['rows'], ('model', 'sample_id', 'state'), expected_inputs, 'depth input identities')
    geometry = evaluation.read_json(root / 'depth_geometry_audit.json')
    exact(geometry['status'], 'pass', 'depth geometry status')
    geometry_index = unique_index(geometry['checks'], ('model', 'sample_id', 'state'),
                                  [(model, sample['sample_id'], state) for model in DEPTH_MODELS for sample in plan['samples'] for state in STATES[1:]],
                                  'depth geometry rows')
    masks, rebuilt = {}, []
    for model, sample_id in expected_entries:
        entry, case = entries[(model, sample_id)], cases[sample_id]
        sample = next(item for item in plan['samples'] if item['sample_id'] == sample_id)
        for key, value in case.items():
            exact(entry[key], value, 'depth entry.' + key)
        path = evaluation.checked_file(root, entry['path'])
        exact(evaluation.sha(path), entry['sha256'], 'raw depth file SHA')
        payload = load_torch_lossless(path, pool_root=pool_root)
        for key, value in case.items():
            exact(payload[key], value, 'raw depth.' + key)
        exact(payload['model'], model, 'raw depth model')
        exact(payload['split'], sample['source_split'], 'raw depth split')
        exact(payload['semantics'], SEMANTICS[model], 'raw depth semantics')
        exact(entry['semantics'], SEMANTICS[model], 'manifest depth semantics')
        exact(payload['object_mask'], entry['object_mask'], 'ROI binding')
        mask_binding = entry['object_mask']
        mask_path = evaluation.checked_file(root, mask_binding['path'])
        exact(evaluation.sha(mask_path), mask_binding['sha256'], 'ROI file SHA')
        mask = load_torch_lossless(mask_path, pool_root=pool_root)
        exact(tensor_sha(mask), mask_binding['tensor_sha256'], 'ROI typed SHA')
        if (mask.dtype != torch.float32 or mask.ndim != 4 or mask.shape[:2] != (1, 1)
                or not bool(torch.isfinite(mask).all()) or not bool(mask.ne(0).any())
                or not bool(((mask == 0) | (mask == 1)).all())):
            raise ValueError('invalid inserted vehicle ROI')
        ys, xs = torch.where(mask[0, 0].ne(0))
        box = boxes[sample_id]
        if int(xs.min()) < box[0] or int(xs.max()) >= box[2] or int(ys.min()) < box[1] or int(ys.max()) >= box[3]:
            raise ValueError('vehicle ROI lies outside bound target box')
        if sample_id in masks:
            exact(mask_binding, masks[sample_id], 'same ROI across models')
        masks[sample_id] = mask_binding
        for state in STATES:
            value = payload['states'][state]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != mask.shape or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError('invalid native depth prediction')
            exact(tensor_sha(value), entry['tensor_sha256'][state], 'raw depth typed SHA')
            counts['typed_sha_checks'] += 1
            source_input = detector_inputs[(DETECTION_MODELS[0], sample_id, state)]
            exact(payload['input_sha256'][state], source_input['rendered_input_sha256'], 'depth/detection raw input identity')
            input_row = inputs[(model, sample_id, state)]
            exact(input_row['input_sha256'], payload['input_sha256'][state], 'depth input SHA')
            for key, bound in sample_binding(case, state, random_binding).items():
                exact(input_row[key], bound, 'depth input.' + key)
            if state != 'clean':
                exact(payload['support_sha256'][state], source_input['support_tensor_sha256'], 'same projected support')
                gt = geometry_index[(model, sample_id, state)]
                exact(gt['target_box'], box, 'depth geometry target box')
                exact(gt['road_quad'], torch.as_tensor(json.loads(sample['road_quad_norm'])).tolist(), 'depth road quadrilateral')
                exact(gt['split'], sample['source_split'], 'depth original split')
                exact(gt['sample_binding'], sample_binding(case, state, random_binding), 'depth geometry Patch owner')
                if not gt['checks'] or not all(value is True for value in gt['checks'].values()):
                    raise ValueError('depth geometry invariants did not pass')
        for ref, adv in COMPARISONS:
            row = dict(sample_binding(case, adv, random_binding), method=evaluation.IDENTITY, model=model,
                       split=sample['source_split'], reference_state=ref, evaluated_state=adv,
                       reference_patch_binding=None if ref == 'clean' else case['patches'][ref],
                       used_in_training=model in plan['scientific_config']['depth_models'])
            row.update(metrics.depth_metrics(payload['states'][adv], payload['states'][ref], mask, semantics=SEMANTICS[model]))
            exact(row, saved_index[(model, sample_id, ref, adv)], 'native depth per-sample metric')
            rebuilt.append(row)
            counts['depth_rows_checked'] += 1
    summaries = depth_summaries(metrics, rebuilt)
    saved_summary = evaluation.read_json(root / 'depth_summary.json')
    exact(summaries, saved_summary['summaries'], 'depth full summaries')
    exact(saved_summary['records'], count * 16, 'depth records')
    exact(saved_summary['model_forward_count'], count * 16, 'depth summary forwards')
    return rebuilt, summaries


def audit_evaluation(root, plan, metrics, *, tensor_pool, check_patch_files=True):
    import torch
    root = Path(root).resolve()
    summary = evaluation.read_json(root / 'summary.json')
    pool_root = checked_pool(tensor_pool)
    exact(summary['raw_storage'], evaluation.RAW_STORAGE, 'lossless tensor pool storage policy')
    exact(summary['tensor_pool_root'], str(pool_root.resolve()), 'frozen tensor pool namespace')
    exact(summary['ablation_variant'], plan['ablation_variant'], 'evaluation ablation owner')
    observed_blobs = evaluation.collect_pool_bindings(root)
    exact(observed_blobs, summary['tensor_pool_bindings'], 'complete external tensor-pool bindings')
    for binding in observed_blobs.values():
        materialize_tensors({'__advrm_lossless_tensor_blob_v1__': binding}, pool_root)
    for key, expected in dict(identity=evaluation.IDENTITY, status='same_sample_attack_evaluation_complete',
                              plan_sha256=plan['_plan_sha256'], source_bindings=plan['source_bindings'],
                              sample_count=len(plan['samples']), evaluation_forward_total=32 * len(plan['samples']),
                              updates_per_sample=plan['budget']['updates_per_sample'], metric_source_sha256=METRIC_SHA256,
                              test_data_opened=False, same_sample_patch_routing=True, fixed_final=True).items():
        exact(summary[key], expected, 'evaluation summary.' + key)
    exact(summary['sample_ids'], [sample['sample_id'] for sample in plan['samples']], 'planned sample order')
    counts = dict(artifact_sha_checks=0, typed_sha_checks=0, detection_rows_checked=0, depth_rows_checked=0)
    files = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and path != root / 'summary.json'}
    exact(set(summary['artifact_bindings']), files, 'complete evaluation artifact set')
    for name, digest in summary['artifact_bindings'].items():
        path = evaluation.checked_file(root, name)
        exact(evaluation.sha(path), evaluation.require_sha(digest), 'evaluation artifact SHA')
        counts['artifact_sha_checks'] += 1
    exact(evaluation.sha(root / 'PLAN.json'), plan['_plan_sha256'], 'copied evaluation plan SHA')
    case_rows = evaluation.read_json(root / 'case_bindings.json')['cases']
    cases = unique_index(case_rows, ('sample_id',), [(sample['sample_id'],) for sample in plan['samples']], 'case bindings')
    cases = {key[0]: value for key, value in cases.items()}
    for sample in plan['samples']:
        case = cases[sample['sample_id']]
        check_case(case, sample, plan)
        if check_patch_files:
            for state, bound in case['patches'].items():
                path = Path(bound['path'])
                if path.name != {'zero': 'zero_step_patch.pt', 'final': 'final_patch.pt'}[state] or path.parent.name != 'train' or path.parent.parent.name != sample['case_id']:
                    raise ValueError('own patch path belongs to another case')
                evaluation.checked_file(path.parent, path.name, {key: bound[key] for key in ('sha256', 'bytes')})
                payload = load_torch_lossless(path, pool_root=pool_root)
                from case_owners200 import resolve_case_owner
                evaluation.validate_patch_metadata(payload, resolve_case_owner(plan, sample), sample,
                    0 if state == 'zero' else plan['budget']['updates_per_sample'])
                rgb = payload['canonical_rgb']
                if rgb.dtype != torch.float32 or rgb.shape != (1, 3, 64, 128) or not bool(torch.isfinite(rgb).all()) or bool(((rgb < 0) | (rgb > 1)).any()):
                    raise ValueError('invalid own canonical Patch')
                exact(tensor_sha(rgb), bound['tensor_sha256'], 'own canonical Patch typed SHA')
    # Derive the shared random-control identity from the evaluator's persisted rows.
    input_rows = evaluation.read_json(root / 'rendered_input_hashes.json')['rows']
    random_bindings = [row['patch_binding'] for row in input_rows if row['state'] == 'random']
    if not random_bindings:
        raise ValueError('random control identity missing')
    random_binding = random_bindings[0]
    for item in random_bindings:
        exact(item, random_binding, 'same random control')
    random_path = evaluation.checked_file(root, 'random_control_patch.pt')
    exact(evaluation.sha(random_path), random_binding['sha256'], 'random control file SHA')
    random_payload = torch.load(random_path, map_location='cpu', weights_only=True)
    random_rgb = random_payload['canonical_rgb']
    expected_random = torch.rand((1, 3, 64, 128), dtype=torch.float32, generator=torch.Generator().manual_seed(SEED))
    if not torch.equal(random_rgb, expected_random):
        raise ValueError('random control differs from seed20260919')
    exact(tensor_sha(random_rgb), random_binding['tensor_sha256'], 'random typed SHA')
    tolerances = []
    det_rows, det_summary, boxes, det_inputs = reconstruct_detection(root, plan, cases, metrics, random_binding, counts, tolerances, pool_root)
    dep_rows, dep_summary = reconstruct_depth(root, plan, cases, metrics, random_binding, counts, boxes, det_inputs, pool_root)
    return {'schema': 'sample-specific-raw-recomputed-v1', 'status': 'raw_audit_passed',
            'identity': evaluation.IDENTITY, 'ablation_variant': plan['ablation_variant'], 'plan_sha256': plan['_plan_sha256'],
            'source_bindings': plan['source_bindings'], 'scientific_config': plan['scientific_config'],
            'samples': plan['samples'], 'master_samples': plan.get('master_samples'), 'partition': plan.get('partition'),
            'shard_id': plan.get('shard_id'), 'sample_bindings': case_rows,
            'selection_binding': plan.get('selection_binding'), 'original_plan_bindings': plan.get('original_plan_bindings'),
            'case_owner_policy': plan.get('case_owner_policy'), 'admission_binding': plan.get('admission_binding'),
            'updates_per_sample': plan['budget']['updates_per_sample'], 'metric_source_sha256': METRIC_SHA256,
            'counts': counts, 'detection_rows': det_rows, 'depth_rows': dep_rows,
            'detection_summaries': det_summary, 'depth_summaries': dep_summary,
            'evaluation_summary_sha256': evaluation.sha(root / 'summary.json'),
            'tolerated_iou': tolerances, 'cpu_gpu_iou_atol': IOU_ATOL,
            'metric_numeric_tolerance': 0, 'raw_storage': evaluation.RAW_STORAGE,
            'tensor_pool_root': str(pool_root.resolve()), 'tensor_pool_bindings': observed_blobs, 'no_models_loaded_or_called': True, 'no_source_images_opened': True,
            'test_data_opened': False, 'scope': 'raw prediction reconstruction and identity audit; not held-out generalization'}


def merge_results(plan, documents, metrics):
    """Union predefined members, then recompute AP/bootstrap and depth macro."""
    master, partition = plan['master_samples'], plan['partition']
    if not isinstance(master, list) or not master or partition.get('shard_count') != len(documents):
        raise ValueError('complete predefined master samples and shards required')
    expected_shards = set(range(partition['shard_count']))
    if {doc.get('shard_id') for doc in documents} != expected_shards or len(documents) != len(expected_shards):
        raise ValueError('missing/duplicate shard')
    samples, det_rows, depth_rows, bindings = [], [], [], []
    for doc in sorted(documents, key=lambda item: item['shard_id']):
        for key, expected in {'schema': 'sample-specific-raw-recomputed-v1', 'status': 'raw_audit_passed',
                              'identity': evaluation.IDENTITY, 'source_bindings': plan['source_bindings'],
                              'ablation_variant': plan['ablation_variant'],
                              'scientific_config': plan['scientific_config'], 'metric_source_sha256': METRIC_SHA256,
                              'updates_per_sample': plan['budget']['updates_per_sample'],
                              'master_samples': master, 'partition': partition, 'test_data_opened': False}.items():
            exact(doc[key], expected, 'audited shard.' + key)
        expected_ids = partition['shard_members'][str(doc['shard_id'])]
        exact([sample['sample_id'] for sample in doc['samples']], expected_ids, 'predefined shard membership')
        evaluation.require_sha(doc['plan_sha256'])
        case_index = unique_index(doc['sample_bindings'], ('sample_id',), [(sample_id,) for sample_id in expected_ids], 'audited shard case identities')
        for sample in doc['samples']:
            matches = [candidate for candidate in master if candidate['sample_id'] == sample['sample_id']]
            if len(matches) != 1:
                raise ValueError('shard contains an unexpected master sample')
            exact(sample, matches[0], 'master source metadata')
            check_case(case_index[(sample['sample_id'],)], sample, {**plan, 'samples': doc['samples'],
                'admission_binding': doc['admission_binding'], '_plan_sha256': doc['plan_sha256']})
        for row in [*doc['detection_rows'], *doc['depth_rows']]:
            if row['sample_id'] not in expected_ids:
                raise ValueError('metric row belongs to another shard')
            case = case_index[(row['sample_id'],)]
            for key, value in case.items():
                if key != 'patches':
                    exact(row[key], value, 'merged per-sample metric owner.' + key)
            exact(row['split'], case['source_split'], 'merged original source split')
            state = row.get('state', row.get('evaluated_state'))
            if state in ('zero', 'final'):
                exact(row['patch_binding'], case['patches'][state], 'merged own Patch identity')
            elif state == 'random':
                random = row['patch_binding']
                if random.get('role') != 'common_random_control_only':
                    raise ValueError('merged random control identity differs')
            else:
                raise ValueError('unexpected merged metric state')
            if row.get('reference_state') == 'zero':
                exact(row['reference_patch_binding'], case['patches']['zero'], 'merged zero-reference owner')
        samples.extend(doc['samples']); det_rows.extend(doc['detection_rows']); depth_rows.extend(doc['depth_rows']); bindings.extend(doc['sample_bindings'])
    expected_ids = [sample['sample_id'] for sample in master]
    if len(samples) != len(master) or {sample['sample_id'] for sample in samples} != set(expected_ids):
        raise ValueError('master sample coverage incomplete/duplicate')
    det_index = unique_index(det_rows, ('model', 'sample_id', 'state'),
                             [(model, sample_id, state) for model in DETECTION_MODELS for sample_id in expected_ids for state in STATES[1:]], 'merged detector rows')
    depth_index = unique_index(depth_rows, ('model', 'sample_id', 'reference_state', 'evaluated_state'),
                               [(model, sample_id, ref, adv) for model in DEPTH_MODELS for sample_id in expected_ids for ref, adv in COMPARISONS], 'merged depth rows')
    ordered_det = [det_index[(model, sample_id, state)] for model in DETECTION_MODELS for sample_id in expected_ids for state in STATES[1:]]
    ordered_depth = [depth_index[(model, sample_id, ref, adv)] for model in DEPTH_MODELS for sample_id in expected_ids for ref, adv in COMPARISONS]
    return {'schema': 'sample-specific-global-merged-v1', 'status': 'all_predefined_samples_merged',
            'identity': evaluation.IDENTITY, 'sample_count': len(master), 'samples': master,
            'ablation_variant': plan['ablation_variant'],
            'source_bindings': plan['source_bindings'], 'scientific_config': plan['scientific_config'],
            'updates_per_sample': plan['budget']['updates_per_sample'], 'partition': partition,
            'metric_source_sha256': METRIC_SHA256, 'sample_bindings': bindings,
            'shard_plan_sha256': {str(doc['shard_id']): doc['plan_sha256'] for doc in documents},
            'detection_summaries': detection_summaries(metrics, ordered_det),
            'depth_summaries': depth_summaries(metrics, ordered_depth),
            'detection_aggregation': 'global candidate ranking over all master samples; not shard AP/TNR mean',
            'depth_aggregation': 'per-sample macro over all master samples; bootstrap master order',
            'test_data_opened': False, 'heldout_generalization_claim_allowed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--tensor-pool', type=Path)
    parser.add_argument('--merge-results', type=Path, nargs='+')
    parser.add_argument('--merge-result-sha256', nargs='+', help='Required exact transport SHA for each audited shard JSON')
    parser.add_argument('--metric-source', type=Path)
    for name in ('child', 'r2-source', 'v2-source', 'clip-source'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if bool(args.evaluation) == bool(args.merge_results):
        parser.error('exactly one of --evaluation or --merge-results is required')
    from sample_contract import load_plan
    plan = load_plan(args.plan)
    if args.output.exists() or not args.output.parent.is_dir() or any(path.is_symlink() for path in (args.output, *args.output.absolute().parents)):
        raise FileExistsError('fresh audit output with an existing nonsymlink parent required')
    if args.evaluation and (args.output.resolve() == args.evaluation.resolve() or args.evaluation.resolve() in args.output.resolve().parents):
        raise ValueError('audit must be outside evaluation evidence')
    if args.merge_results:
        if args.metric_source is None:
            parser.error('--metric-source is required for merge')
        if args.merge_result_sha256 is None or len(args.merge_result_sha256) != len(args.merge_results):
            parser.error('one --merge-result-sha256 value per audited shard JSON is required')
        for path, expected in zip(args.merge_results, args.merge_result_sha256):
            if evaluation.sha(path) != evaluation.require_sha(expected):
                raise ValueError('audited shard JSON transport SHA differs')
        metric_path = args.metric_source
    else:
        if any(getattr(args, name) is None for name in ('child', 'r2_source', 'v2_source', 'clip_source')):
            parser.error('all four scientific source directories are required for raw audit')
        from source_bindings import verify_sources
        sources = verify_sources(plan, child=args.child, r2_source=args.r2_source, v2_source=args.v2_source, clip_source=args.clip_source)
        metric_path = sources['child'] / 'parent_v17/04_统一评估/evaluator.py'
    metrics = load_metrics(metric_path)
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        if args.merge_results:
            documents = [evaluation.read_json(path) for path in args.merge_results]
            result = merge_results(plan, documents, metrics)
            result['input_result_file_sha256'] = {str(path): evaluation.sha(path) for path in args.merge_results}
        else:
            if args.tensor_pool is None:
                parser.error('--tensor-pool required for new content-addressed raw audit')
            result = audit_evaluation(args.evaluation, plan, metrics, tensor_pool=args.tensor_pool)
        result.update(audit_source_sha256=evaluation.sha(Path(__file__)), seconds=time.monotonic() - started)
        evaluation.write_json(args.output / 'RECOMPUTED.json', result)
        small = {key: value for key, value in result.items() if key not in ('detection_rows', 'depth_rows')}
        small['recomputed_sha256'] = evaluation.sha(args.output / 'RECOMPUTED.json')
        evaluation.write_json(args.output / 'summary.json', small)
        print(json.dumps({'status': result['status'], 'output': str(args.output)}, ensure_ascii=False), flush=True)
    except BaseException as error:
        evaluation.write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
