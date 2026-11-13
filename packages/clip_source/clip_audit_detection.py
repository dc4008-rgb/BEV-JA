#!/usr/bin/env python3
"""One-run raw detection audit for fixed-final500 outputs; no model or image IO.

Storage checks reuse clip_audit_depth helpers. Scientific functions are extracted
unchanged from SHA-bound frozen evaluator.py and R2 evaluate.py. The old raw
audit's CPU/GPU IoU allowance remains separate from exact AP/TNR/bootstrap.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import math
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import clip_audit_depth as storage

MODELS = ('fasterrcnn', 'ssd300', 'maskrcnn', 'fcos')
STATES = ('zero', 'random', 'final')
DOMAINS = ('kitti', 'nuscenes', 'combined')
IOU_AUDIT_ATOL = 1e-7  # Existing meeting_full_repeat audit_results.py policy.
SEED = 20260919


def functions(path, names, namespace):
    nodes = [node for node in ast.parse(Path(path).read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError('required frozen function definitions are missing/ambiguous')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def prediction_valid(value):
    import torch
    if not isinstance(value, dict) or set(value) != {'boxes', 'labels', 'scores'}:
        raise ValueError('raw prediction must contain all boxes/labels/scores')
    boxes, labels, scores = value['boxes'], value['labels'], value['scores']
    if (boxes.dtype != torch.float32 or scores.dtype != torch.float32 or labels.dtype != torch.int64
            or boxes.shape != (scores.numel(), 4) or labels.shape != scores.shape or scores.ndim != 1
            or not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all())
            or not bool(((scores >= 0) & (scores <= 1)).all())):
        raise ValueError('raw prediction dtype/shape/finite/range check failed')


def compare_rows(rebuilt, saved, counts, tolerated):
    candidate_fields = ('clean_candidates', 'adversarial_candidates', 'clean_car_candidates', 'adversarial_car_candidates')
    storage.exact({key: value for key, value in rebuilt.items() if key not in candidate_fields},
                  {key: value for key, value in saved.items() if key not in candidate_fields}, 'row_identity', counts)
    for field in candidate_fields:
        actual, expected = saved[field], rebuilt[field]
        storage.exact(len(expected), len(actual), field + '.count', counts)
        for index, (cpu, gpu) in enumerate(zip(expected, actual)):
            storage.exact(set(cpu), set(gpu), field + '.fields', counts)
            for key in ('score', 'label'):
                storage.exact(cpu[key], gpu[key], field + '.' + key, counts)
            left, right = cpu['iou'], gpu['iou']
            counts['iou_checks'] += 1
            if left == right:
                continue
            if (not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > IOU_AUDIT_ATOL
                    or (left > 0) != (right > 0) or (left >= .5) != (right >= .5)):
                raise ValueError('CPU/GPU IoU or candidate/AP50 threshold mismatch')
            tolerated.append({'model': rebuilt['model'], 'sample_id': rebuilt['sample_id'],
                              'state': rebuilt['state'], 'field': field, 'index': index,
                              'saved_gpu': right, 'recomputed_cpu': left, 'absolute_difference': abs(left - right)})


def audit(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.run.resolve()
    summary = storage.read_json(root / 'summary.json')
    if (summary.get('schema') != 'r2-joint-clip120-eight-model-development-evaluation-v1'
            or summary.get('status') != 'development_validation_complete'
            or summary.get('test_data_opened') is not False or summary.get('scene_count') != 40
            or summary.get('updates') != 500 or summary.get('training_budget_updates') != 500
            or summary.get('evaluation_forward_total') != 1280 or summary.get('checkpoint_selection') != 'fixed_final500'):
        raise ValueError('incomplete or incompatible eight-model evaluation')
    counts = {'artifact_sha_checks': 0, 'scalar_checks': 0, 'raw_tensor_checks': 0,
              'iou_checks': 0, 'rows_checked': 0, 'summaries_checked': 0}
    tolerated = []
    bindings, sources = summary['artifact_bindings'], summary['source_bindings']
    if sources['r2_manifest_sha256'] != storage.R2_MANIFEST or sources['child_manifest_sha256'] != storage.CHILD_MANIFEST:
        raise ValueError('frozen R2/child identities differ')
    child_manifest = storage.read_json(storage.bound(args.child, 'PACKAGE_MANIFEST.json', storage.CHILD_MANIFEST))
    source_index = {entry['path']: entry for entry in child_manifest['files']}
    r2_manifest = storage.read_json(storage.bound(args.r2_source, 'PACKAGE_MANIFEST.json', storage.R2_MANIFEST))
    r2_index = {entry['path']: entry for entry in r2_manifest['files']}
    metric_name = 'parent_v17/04_统一评估/evaluator.py'
    csv_name = 'parent_v17/01_公共数据集/runtime_v1/train_validation_manifest.csv'
    metric_path = storage.bound(args.child, metric_name, source_index[metric_name]['sha256'])
    storage.exact(source_index[metric_name]['sha256'], sources['depth_evaluator_sha256'], 'original_metric_source', counts)
    csv_path = storage.bound(args.child, csv_name, source_index[csv_name]['sha256'])
    storage.exact(source_index[csv_name]['sha256'], sources['training_manifest_sha256'], 'split_manifest_sha', counts)
    r2_eval = storage.bound(args.r2_source, 'evaluate.py', r2_index['evaluate.py']['sha256'])
    storage.exact(storage.sha(r2_eval), sources['r2_evaluate_sha256'], 'r2_evaluator_sha', counts)
    wrapper = args.evaluation_source or Path(__file__).with_name('clip_evaluate.py')
    training_root = args.training_run or Path(summary['run'])
    identity_bindings = storage.bind_run_model_wrapper(summary, args.child, source_index,
                            args.training_package_manifest_sha256, args.training_source, training_root, wrapper)
    storage.bound(args.training_source, 'clip_audit_detection.py', storage.sha(Path(__file__)))
    storage.bound(args.training_source, 'clip_audit_depth.py', storage.sha(Path(storage.__file__)))
    training_summary = storage.read_json(storage.bound(training_root, 'summary.json', summary['run_summary_sha256']))
    method, trained = identity_bindings['method'], summary['training_models']
    storage.exact(method, summary['method'], 'actual_method', counts)
    storage.exact(training_summary['candidate_id'], summary['candidate_id'], 'actual_candidate_identity', counts)
    storage.exact(list(trained), summary['training_models'], 'actual_training_models', counts)
    storage.exact(training_summary['training_manifest_sha256'], sources['training_manifest_sha256'], 'training_split_binding', counts)
    model_name = 'parent_v17/02_模型/model_manifest.json'
    storage.bound(args.child, model_name, source_index[model_name]['sha256'])
    storage.exact(source_index[model_name]['sha256'], sources['model_manifest_sha256'], 'frozen_model_manifest', counts)
    storage.exact(training_summary['model_manifest_sha256'], sources['model_manifest_sha256'], 'training_model_binding', counts)
    with csv_path.open(newline='', encoding='utf-8') as stream:
        frozen_rows = sorted((row for row in csv.DictReader(stream) if row['split'] == 'validation'),
                             key=lambda row: (row['dataset'], row['sample_id']))
    scenes = [(row['dataset'], row['sample_id']) for row in frozen_rows]
    if len(scenes) != 40 or len(set(scenes)) != 40 or Counter(dataset for dataset, _ in scenes) != {'kitti': 20, 'nuscenes': 20}:
        raise ValueError('frozen validation must contain exactly 20+20 scenes')
    storage.exact([{'sample_id': row['sample_id'], 'dataset': row['dataset'], 'split': 'validation'} for row in frozen_rows],
                  summary['validation_scenes'], 'frozen_scenes', counts)
    for name in ('per_sample.json', 'geometry_audit.json', 'raw_predictions/MANIFEST.json', 'detection_summary.json', 'CLIP_PLAN.json'):
        storage.bound(root, name, bindings[name]); counts['artifact_sha_checks'] += 1
    storage.bound(root, 'CLIP_PLAN.json', summary['gradient_clipping']['grid_sha256'])
    if bindings.get('CLIP_PLAN.json') != summary['gradient_clipping']['grid_sha256']:
        raise ValueError('evaluation clipping plan artifact binding differs')
    saved_summary = storage.read_json(storage.bound(root, 'detection_summary.json', summary['detection_summary_sha256']))
    for field, expected in {'candidate_id': summary['candidate_id'], 'method': method, 'updates': 500,
                            'records': 480, 'evaluation_forward_total': 640, 'summary_rows': 48,
                            'evaluation_models': list(MODELS), 'bootstrap_resamples': 1000, 'seed': SEED,
                            'test_data_opened': False, 'checkpoint_selection': 'fixed_final500'}.items():
        storage.exact(expected, saved_summary[field], 'detector_summary.' + field, counts)
    storage.exact(summary['run_summary_sha256'], saved_summary['run_summary_sha256'], 'detector_training_binding', counts)
    engine_name = 'parent_v17/05_运行/evaluate_road_comparison.py'
    storage.exact(source_index[engine_name]['sha256'], saved_summary['parent_metric_engine_source_sha256'], 'detector_engine_binding', counts)
    metric = functions(metric_path, {'box_iou', 'detection_candidates', 'average_precision', 'tnr_detection',
                                     'percentile_interval', 'detection_summary'}, {'torch': torch, 'np': np})
    zero_to_final = functions(r2_eval, {'zero_to_final'}, {'MODELS': MODELS, 'SEED': SEED}).zero_to_final
    raw_manifest = storage.read_json(root / 'raw_predictions/MANIFEST.json')
    entries = raw_manifest['entries']
    storage.exact([(model, *scene) for model in MODELS for scene in scenes],
                  [(entry['model'], entry['dataset'], entry['sample_id']) for entry in entries], 'raw_coverage_order', counts)
    storage.exact(640, raw_manifest['model_forward_count'], 'raw_forward_count', counts)
    saved_rows = storage.read_json(root / 'per_sample.json')
    saved_index = {(row['model'], row['dataset'], row['sample_id'], row['state']): row for row in saved_rows}
    if len(saved_rows) != 480 or len(saved_index) != 480:
        raise ValueError('per-sample rows must be unique and cover 480 records')
    geometry = storage.read_json(root / 'geometry_audit.json')['checks']
    geometry_index = {(row['model'], row['dataset'], row['sample_id'], row['state']): row for row in geometry}
    if len(geometry) != 480 or len(geometry_index) != 480:
        raise ValueError('GT geometry rows must cover all 480 records')
    road_quads = {(row['dataset'], row['sample_id']): [[float(x), float(y)] for x, y in storage.json.loads(row['road_quad_norm'])]
                  for row in frozen_rows}
    gt_boxes = {}
    rebuilt = []
    for entry in entries:
        payload = torch.load(storage.bound(root, entry['path'], entry['sha256']), map_location='cpu', weights_only=True)
        counts['artifact_sha_checks'] += 1
        model, dataset, sample_id = entry['model'], entry['dataset'], entry['sample_id']
        for field, expected in (('model', model), ('dataset', dataset), ('sample_id', sample_id), ('split', 'validation')):
            storage.exact(expected, payload[field], 'raw.' + field, counts)
        box = payload['target_box']
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box) or not (0 <= box[0] < box[2] <= 1024 and 0 <= box[1] < box[3] <= 320):
            raise ValueError('invalid inserted-target GT box')
        scene_key = (dataset, sample_id)
        if scene_key in gt_boxes:
            storage.exact(gt_boxes[scene_key], box, 'same_GT_across_models', counts)
        gt_boxes[scene_key] = box
        for state in ('clean', *STATES):
            prediction_valid(payload[state])
            for key, value in payload[state].items():
                storage.exact(storage.tensor_sha(value), entry['tensor_sha256'][state][key], 'raw_tensor_sha', counts)
                counts['raw_tensor_checks'] += 1
        clean = metric.detection_candidates(payload['clean'], box, (3, 6, 8))
        clean_car = metric.detection_candidates(payload['clean'], box, (3,))
        for state in STATES:
            key = (model, dataset, sample_id, state)
            gt = geometry_index[key]
            storage.exact(box, gt['target_box'], 'raw_GT_vs_geometry_GT', counts)
            # Serialization may store integral/floating coordinates; compare numeric quad values explicitly.
            if gt['road_quad'] != torch.as_tensor(road_quads[scene_key]).tolist():
                raise ValueError('GT road quadrilateral differs from frozen scene row')
            if gt['split'] != 'validation' or not gt['checks'] or not all(value is True for value in gt['checks'].values()):
                raise ValueError('saved geometry invariants did not all pass')
            row = {'method': method, 'model': model, 'sample_id': sample_id, 'dataset': dataset,
                   'split': 'validation', 'state': state, 'clean_candidates': clean, 'clean_car_candidates': clean_car,
                   'adversarial_candidates': metric.detection_candidates(payload[state], box, (3, 6, 8)),
                   'adversarial_car_candidates': metric.detection_candidates(payload[state], box, (3,))}
            compare_rows(row, saved_index[key], counts, tolerated)
            counts['rows_checked'] += 1
            rebuilt.append(row)
    summaries = []
    for model in MODELS:
        for state in STATES:
            for domain in DOMAINS:
                subset = [row for row in rebuilt if row['model'] == model and row['state'] == state
                          and (domain == 'combined' or row['dataset'] == domain)]
                value = metric.detection_summary(subset, 1000, SEED, allow_zero_reference=True)
                summaries.append(dict(value, model=model, state=state, dataset=domain))
    zero = zero_to_final(SimpleNamespace(detection_summary=metric.detection_summary), rebuilt)
    storage.exact(summaries, saved_summary['summaries'], 'clean_reference_summaries', counts)
    storage.exact(zero, saved_summary['zero_to_final'], 'zero_to_final_summaries', counts)
    counts['summaries_checked'] = len(summaries) + len(zero)
    storage.write_json(args.output / 'RECOMPUTED.json', {'summaries': summaries, 'zero_to_final': zero,
                                                      'raw_reconstruction_rows': 480, 'raw_forward_outputs': 640})
    return {'status': 'detection_raw_audit_passed', 'counts': counts, 'method': method,
            'updates': 500, 'identity_bindings': identity_bindings,
            'selection_context': summary['selection_context'], 'exploratory_posthoc': True,
            'candidate_id': summary['candidate_id'], 'evaluation_summary_sha256': storage.sha(root / 'summary.json'),
            'training_summary_sha256': summary['run_summary_sha256'], 'evaluation_wrapper_sha256': storage.sha(wrapper),
            'metric_source_sha256': storage.sha(metric_path), 'r2_evaluator_source_sha256': storage.sha(r2_eval),
            'audit_source_sha256': storage.sha(Path(__file__)), 'storage_helpers_sha256': storage.sha(Path(storage.__file__)),
            'tolerated_iou_count': len(tolerated), 'tolerated_iou': tolerated,
            'tolerance_policy': {'scores_labels_candidate_membership_AP50_TP_metrics': 'exact',
                                 'CPU_GPU_reconstructed_IoU_atol': IOU_AUDIT_ATOL,
                                 'IoU_positive_and_AP50_threshold_decisions': 'exact',
                                 'training_repeatability_tolerance': 'not applicable; not relaxed'},
            'no_models_loaded_or_called': True, 'no_images_opened': True, 'test_data_opened': False,
            'checkpoint_selection': 'fixed_final500', 'summary_count': 48,
            'scope': 'raw boxes/labels/scores plus bound inserted-target geometry to 480 rows and 48 summaries',
            'metric_execution': 'unchanged SHA-bound original function definitions; no result cache'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r2-source', 'child', 'run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--training-package-manifest-sha256', required=True,
                        help='Required actual SHA of the newly sealed cap120 followup PACKAGE_MANIFEST.json')
    parser.add_argument('--training-source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--training-run', type=Path, help='Optional relocated training directory; summary SHA must still match')
    parser.add_argument('--evaluation-source', type=Path, help='Default: clip_evaluate.py beside this script')
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise ValueError('live hostname mismatch')
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError('fresh output with existing parent required')
    if any(path.is_symlink() for path in (args.output, *args.output.absolute().parents)):
        raise ValueError('output symlinks forbidden')
    args.output = args.output.resolve()
    if args.run.resolve() == args.output or args.run.resolve() in args.output.parents:
        raise ValueError('audit output must be outside evaluation evidence')
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        result = audit(args)
        result['seconds'] = time.monotonic() - started
        storage.write_json(args.output / 'summary.json', result)
        print(storage.json.dumps({key: result[key] for key in ('status', 'counts', 'tolerated_iou_count')}, ensure_ascii=False), flush=True)
    except BaseException as error:
        storage.write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__,
                                                        'error': str(error), 'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
