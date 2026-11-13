#!/usr/bin/env python3
"""Fixed final100, 40-scene four-detector development evaluation; all raw boxes saved."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
from compare import bound_file, load_training, read_json, sha, tensor_sha, tree_difference, write_json

MODELS = ('fasterrcnn', 'ssd300', 'maskrcnn', 'fcos')
SEED = 20260919


def load_engine(candidate_root):
    path = Path(candidate_root).resolve() / 'training/inputs.py'
    spec = importlib.util.spec_from_file_location('_meeting_eval_v29_inputs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.IDENTITY != 'normalized-v29-c0-u1-lr005-20260922':
        raise ValueError('the frozen lr005 parent is required')
    module.verify_inputs(require_sealed=True)
    return module.parent_evaluator(), module.PARENT


def configure():
    import numpy as np
    import torch
    if os.environ.get('PYTHONHASHSEED') != str(SEED):
        raise ValueError('PYTHONHASHSEED must be set before launching evaluation')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def checked_run(run, engine, parent):
    import torch
    summary, entries = load_training(run)
    for field, path in (
            ('training_manifest_sha256', parent / '01_公共数据集/runtime_v1/train_validation_manifest.csv'),
            ('model_manifest_sha256', parent / '02_模型/model_manifest.json')):
        if summary.get(field) != sha(path):
            raise ValueError('training/evaluation input mismatch: ' + field)
    patches = {}
    for state, name in (('zero', 'zero_step_patch.pt'), ('final', 'final_patch.pt')):
        path = bound_file(run, name, summary.get('patch_bindings', {}).get(name))
        patches[state] = engine.load_patch(path)
    final100 = bound_file(run, 'checkpoint_0100.pt', summary.get('checkpoint_bindings', {}).get('checkpoint_0100.pt'))
    if tree_difference(patches['final'], engine.load_patch(final100)):
        raise ValueError('only fixed final100 is eligible')
    checkpoint = torch.load(final100, map_location='cpu', weights_only=True)
    if checkpoint.get('metadata', {}).get('updates') != 100:
        raise ValueError('checkpoint100 update count differs')
    patches['random'] = engine.common_random_patch()
    return summary, patches


def zero_to_final(engine, rows):
    """Unchanged v29 zero-reference arithmetic and bootstrap arguments."""
    indexed = {(r['model'], r['dataset'], r['sample_id'], r['state']): r for r in rows}
    derived = []
    for row in rows:
        if row['state'] == 'final':
            zero = indexed[(row['model'], row['dataset'], row['sample_id'], 'zero')]
            derived.append(dict(row, clean_candidates=zero['adversarial_candidates'],
                                clean_car_candidates=zero['adversarial_car_candidates']))
    output = []
    for model in MODELS:
        for domain in ('kitti', 'nuscenes', 'combined'):
            subset = [r for r in derived if r['model'] == model and (domain == 'combined' or r['dataset'] == domain)]
            value = engine.detection_summary(subset, 1000, SEED, allow_zero_reference=True)
            value.update(model=model, dataset=domain, reference_state='zero', evaluated_state='final',
                         tnr_undefined_reason=None if value['tnr_defined'] else 'reference_AP50_is_zero')
            output.append(value)
    return output


def evaluate(args, engine, records, summary, patches, device):
    import torch
    selected = engine.validation_rows(records)
    car, mask = engine.load_car(args.car, args.car_mask, device)
    initial = engine.build_context(engine.calibration_records(records)[0], car, mask, device)
    expected_zero = engine.canonicalize_rgb(engine.extract_road_patch(initial.background, initial.road_quad))
    if tree_difference(patches['zero'], expected_zero):
        raise ValueError('zero differs from common unoptimized first training road crop')
    del initial, expected_zero
    engine.save_patch(args.output / 'random_control_patch.pt', patches['random'], {'state': 'unoptimized_random_control', 'seed': SEED})
    control_hashes = {k: tensor_sha(v) for k, v in patches.items()}
    patches = {key: value.to(device) for key, value in patches.items()}
    (args.output / 'raw_predictions').mkdir()
    rows, geometry, inputs, summaries, raw_entries = [], [], [], [], []
    counts = dict.fromkeys(MODELS, 0)
    first = {domain: next(r.sample_id for r in selected if r.dataset == domain) for domain in ('kitti', 'nuscenes')}
    for model_index, model in enumerate(MODELS):
        adapter = engine.build_offline_detector(model, device, args.model_manifest, args.weights_root)
        by_state = {state: [] for state in engine.STATES}
        try:
            with torch.no_grad():
                for scene_index, record in enumerate(selected):
                    context = engine.build_context(record, car, mask, device)
                    clean = {key: value for key, value in adapter.predict(context.benign)[0].items() if key in ('boxes', 'labels', 'scores')}
                    counts[model] += 1
                    raw = {'sample_id': record.sample_id, 'dataset': record.dataset, 'split': 'validation', 'model': model,
                           'target_box': list(context.object_box), 'clean': engine.cpu_predictions(clean)}
                    raw_inputs = {'sample_id': record.sample_id, 'clean': context.benign.cpu(), 'object_mask': context.object_mask.cpu(),
                                  'road_quad': torch.as_tensor(context.road_quad).cpu()} if model_index == 0 and record.sample_id == first[record.dataset] else None
                    if record.sample_id == first[record.dataset]:
                        engine.save_preview(args.output / (model + '_' + record.dataset + '_clean.png'), context.benign, clean,
                                            context.object_box, model + ' | ' + record.sample_id + ' | clean')
                    for state in engine.STATES:
                        render = engine.RoadRGBView(patches[state]).render(context)
                        invariant = engine.invariant_record(context, render, record, model, state)
                        geometry.append(invariant)
                        inputs.append(dict({k: invariant[k] for k in ('sample_id', 'dataset', 'split', 'model', 'state')},
                                           clean_input_sha256=tensor_sha(context.benign), rendered_input_sha256=tensor_sha(render.image),
                                           support_tensor_sha256=tensor_sha(render.support)))
                        prediction = {key: value for key, value in adapter.predict(render.image)[0].items() if key in ('boxes', 'labels', 'scores')}
                        counts[model] += 1
                        raw[state] = engine.cpu_predictions(prediction)
                        row = {'method': 'ours_meeting_six', 'model': model, 'sample_id': record.sample_id, 'dataset': record.dataset,
                               'split': 'validation', 'state': state,
                               'clean_candidates': engine.detection_candidates(clean, context.object_box, engine.VEHICLE_CLASS_IDS),
                               'adversarial_candidates': engine.detection_candidates(prediction, context.object_box, engine.VEHICLE_CLASS_IDS),
                               'clean_car_candidates': engine.detection_candidates(clean, context.object_box, engine.CAR_CLASS_IDS),
                               'adversarial_car_candidates': engine.detection_candidates(prediction, context.object_box, engine.CAR_CLASS_IDS)}
                        rows.append(row)
                        by_state[state].append(row)
                        if raw_inputs is not None:
                            raw_inputs[state] = render.image.cpu()
                            raw_inputs[state + '_support'] = render.support.cpu()
                        if record.sample_id == first[record.dataset]:
                            engine.save_preview(args.output / (model + '_' + record.dataset + '_' + state + '.png'), render.image,
                                                prediction, context.object_box, model + ' | ' + record.sample_id + ' | ' + state)
                    name = 'raw_predictions/%s_%03d.pt' % (model, scene_index)
                    torch.save(raw, args.output / name)
                    raw_entries.append({'model': model, 'sample_id': record.sample_id, 'dataset': record.dataset,
                                        'path': name, 'sha256': sha(args.output / name),
                                        'tensor_sha256': {state: {k: tensor_sha(v) for k, v in raw[state].items()}
                                                          for state in ('clean', *engine.STATES)}})
                    if raw_inputs is not None:
                        torch.save(raw_inputs, args.output / (record.dataset + '_inputs.pt'))
                for state in engine.STATES:
                    for domain in ('kitti', 'nuscenes', 'combined'):
                        subset = [r for r in by_state[state] if domain == 'combined' or r['dataset'] == domain]
                        value = engine.detection_summary(subset, 1000, SEED, allow_zero_reference=True)
                        summaries.append(dict(value, model=model, state=state, dataset=domain))
        finally:
            del adapter
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if len(rows) != 480 or len(geometry) != 480 or len(inputs) != 480 or counts != dict.fromkeys(MODELS, 160):
        raise RuntimeError('frozen evaluation coverage changed')
    if Counter(entry['model'] for entry in raw_entries) != dict.fromkeys(MODELS, 40):
        raise RuntimeError('missing raw predictions')
    write_json(args.output / 'per_sample.json', rows)
    write_json(args.output / 'geometry_audit.json', {'status': 'pass', 'unique_scenes': 40, 'checked_model_scene_states': 480, 'checks': geometry})
    write_json(args.output / 'rendered_input_hashes.json', {'schema': 'meeting-evaluation-input-sha-v1', 'rows': inputs,
                'hash_rule': 'SHA256(canonical dtype/shape then original contiguous dtype bytes)', 'additional_model_forwards': 0})
    write_json(args.output / 'raw_predictions/MANIFEST.json', {'schema': 'meeting-evaluation-raw-predictions-v1', 'entries': raw_entries,
                'model_forward_count': 640, 'fields': ['boxes', 'labels', 'scores'], 'all_40_scenes_all_four_models_all_four_states': True})
    return {'status': 'development_validation_complete', 'candidate_id': summary['candidate_id'],
            'protocol_sha256': summary['protocol_sha256'], 'package_manifest_sha256': summary['package_manifest_sha256'],
            'updates': 100, 'records': 480, 'evaluation_forward_total': 640, 'evaluation_model_scene_forwards': counts,
            'summaries': summaries, 'zero_to_final': zero_to_final(engine, rows), 'summary_rows': 48,
            'evaluation_models': list(MODELS), 'evaluation_target_classes': [3, 6, 8], 'evaluation_car_auxiliary_classes': [3],
            'evaluation_model_roles': {m: ('source' if m in MODELS[:2] else 'transfer') for m in MODELS},
            'control_tensor_sha256': control_hashes, 'parent_metric_engine_source_sha256': sha(Path(engine.__file__)),
            'test_data_opened': False, 'formal_test_allowed': False, 'automatic_promotion': False,
            'checkpoint_selection': 'fixed_final100', 'run_summary_sha256': sha(args.run_output / 'summary.json'),
            'seed': SEED, 'metric': 'tnr_det=(target_vehicle_AP50_clean-target_vehicle_AP50_adv)/target_vehicle_AP50_clean',
            'raw_prediction_count': 640, 'bootstrap_resamples': 1000, 'source_images_full_raw_saved': 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--run-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--source-prefix', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--package-manifest-sha256', required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise ValueError('live hostname mismatch')
    args.run_output, args.output = args.run_output.resolve(), args.output.resolve()
    if args.output.exists() or args.output == args.run_output or args.run_output in args.output.parents:
        raise FileExistsError('evaluation needs a fresh directory outside training')
    if (args.source_prefix is None) != (args.source_root is None):
        raise ValueError('source mapping requires both prefix and root')
    args.output.mkdir(parents=True)
    try:
        configure()
        import torch
        import torchvision
        from importlib.metadata import version
        if (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0], version('diffusers')) != ('2.8.0', '0.23.0', '0.38.0'):
            raise ValueError('frozen runtime versions differ')
        engine, parent = load_engine(args.candidate_root)
        summary, patches = checked_run(args.run_output, engine, parent)
        if summary['package_manifest_sha256'] != args.package_manifest_sha256:
            raise ValueError('training package SHA binding differs')
        args.training_manifest = parent / '01_公共数据集/runtime_v1/train_validation_manifest.csv'
        args.model_manifest = parent / '02_模型/model_manifest.json'
        args.car, args.car_mask = parent / '01_公共数据集/car_00.jpg', parent / '01_公共数据集/car_00_mask.jpg'
        records = engine.training_records(args.training_manifest, args.source_prefix, args.source_root)
        checks = engine.preflight(records, MODELS, args.model_manifest, args.car, args.car_mask, args.weights_root)
        write_json(args.output / 'input_preflight.json', checks)
        if not checks['inputs_ready'] or not torch.cuda.is_available():
            raise RuntimeError('original assets and CUDA required')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        started = time.monotonic()
        result = evaluate(args, engine, records, summary, patches, device)
        torch.cuda.synchronize(device)
        result['seconds'] = time.monotonic() - started
        result['hostname'] = socket.gethostname()
        result['artifact_bindings'] = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
        write_json(args.output / 'summary.json', result)
        print(json.dumps({'status': result['status'], 'forwards': 640, 'summary_rows': 48}, ensure_ascii=False))
    except BaseException as error:
        write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
