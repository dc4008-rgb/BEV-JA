"""Ours5 first-two ablation full200 evaluation, derived from executed full8 code.

The original rendering, detector/depth calls, native/CAS storage and numerical
aggregation functions are preserved. New cases keep their actual method and
training provenance. Standard YOLOv5 is an evaluation target.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[_name] = '1'
IDENTITY = 'ours5-first2-ablations-fixed200-eight-target-YOLOv5-v1'
EVALUATION_SCHEMA = 'ours5-ablation-full200-eight-target-evaluation-v1'
SEED = 20260919
DETECTION_MODELS = ('fasterrcnn', 'ssd300', 'yolov5s', 'fcos')
DEPTH_MODELS = ('monodepth2', 'dpt_large', 'depth_anything_v1', 'zoedepth')
SEMANTICS = {'monodepth2': 'mono_scale_ambiguous_depth', 'dpt_large': 'relative_inverse_depth', 'depth_anything_v1': 'relative_inverse_depth', 'zoedepth': 'metric_depth'}
STATES = ('clean', 'zero', 'random', 'final')
COMPARISONS = (('clean', 'zero'), ('clean', 'random'), ('clean', 'final'), ('zero', 'final'))
METRIC_SHA256 = '50bcf1aad80ae902d6e660017b1614d83c9a8a085b6a6cbe02a5402e31752c81'
from artifact_io import CAS_STORAGE as RAW_STORAGE, load_torch_lossless, save_torch_lossless, checked_pool, tensor_blob_bindings, materialize_tensors
TENSOR_POOL = None

def save_raw(path, payload):
    """Losslessly externalize exact computed tensors; all victim calls execute."""
    if TENSOR_POOL is None:
        raise ValueError('explicit checked tensor pool required')
    return save_torch_lossless(plain(path), payload, pool_root=TENSOR_POOL)

def load_raw(path):
    """Native train .pt stays direct; CAS raw requires the explicit checked pool."""
    p = plain(path)
    if str(p).endswith('.pt.gz'):
        if TENSOR_POOL is None:
            raise ValueError('checked tensor pool required for raw decode')
        return load_torch_lossless(p, pool_root=TENSOR_POOL)
    return load_torch_lossless(p)

def collect_pool_bindings(output):
    result = {}
    for p in sorted(Path(output).rglob('*.pt.gz')):
        for name, binding in tensor_blob_bindings(load_torch_lossless(p)).items():
            if name in result and result[name] != binding:
                raise ValueError('inconsistent CAS binding')
            result[name] = binding
    return result

def bind_tensor_pool(path):
    global TENSOR_POOL
    path = checked_pool(plain(path))
    if TENSOR_POOL is not None and TENSOR_POOL != path:
        raise ValueError('cannot change tensor pool within evaluation process')
    TENSOR_POOL = path
    return path

def verify_pool_retention(output, summary):
    if TENSOR_POOL is None or str(TENSOR_POOL) != summary.get('tensor_pool_root'):
        raise ValueError('exact retained pool root required')
    bindings = collect_pool_bindings(output)
    exact(bindings, summary.get('tensor_pool_bindings'), 'all exact computed tensor blob references')
    for binding in bindings.values():
        materialize_tensors({'__advrm_lossless_tensor_blob_v1__': binding}, TENSOR_POOL)
    return dict(status='all_bound_tensor_blobs_verified', retained_blobs=len(bindings), retained_blob_bytes=sum((b['bytes'] for b in bindings.values())))

def plain(value):
    path = Path(value).absolute()
    if '..' in path.parts or any((p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('symlink/traversal forbidden')
    return path

def exact(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('Shared own-case mismatch: ' + label)

def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value)):
        raise ValueError('explicit lowercase SHA256 required')
    return value

def read_json(path):

    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def checked_file(root, relative, binding=None):
    root, relative = (Path(root), Path(relative))
    if relative.is_absolute() or '..' in relative.parts or (not relative.parts):
        raise ValueError('unsafe artifact path')
    path = root / relative
    if any((part.is_symlink() for part in (path, *path.absolute().parents))) or not path.is_file():
        raise ValueError('missing/symlink artifact: ' + str(path))
    if binding is not None:
        if not isinstance(binding, dict) or set(binding) != {'sha256', 'bytes'}:
            raise ValueError('artifact binding must contain sha256 and bytes')
        if type(binding['bytes']) is not int or binding['bytes'] < 0:
            raise ValueError('invalid artifact byte count')
        if path.stat().st_size != binding['bytes'] or sha(path) != require_sha(binding['sha256']):
            raise ValueError('artifact bytes/SHA mismatch: ' + str(path))
    return path

def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def load_metric_functions(path):
    """Execute only the exact frozen metric definitions, without model imports."""
    import math
    import statistics
    import numpy as np
    import torch
    import torch.nn.functional as F
    if sha(path) != METRIC_SHA256:
        raise ValueError('frozen metric source SHA differs')
    names = {'average_precision', 'tnr_detection', 'detection_summary', 'depth_metrics', 'depth_summary', 'percentile_interval'}
    nodes = [node for node in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise ValueError('missing or ambiguous frozen metric definitions')
    namespace = dict(math=math, statistics=statistics, np=np, torch=torch, F=F)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})

def domains(rows):
    return tuple((domain for domain in ('kitti', 'nuscenes', 'combined') if domain == 'combined' or any((row['dataset'] == domain for row in rows))))

def summarize_detection(metrics, rows):
    """Combined AP ranks candidates globally across all samples, not domain means."""
    output = []
    for model in DETECTION_MODELS:
        for reference, evaluated in COMPARISONS:
            by_state = {(row['dataset'], row['sample_id'], row['state']): row for row in rows if row['model'] == model}
            selected = []
            for row in rows:
                if row['model'] != model or row['state'] != evaluated:
                    continue
                value = dict(row)
                if reference == 'zero':
                    zero = by_state[row['dataset'], row['sample_id'], 'zero']
                    value.update(clean_candidates=zero['adversarial_candidates'], clean_car_candidates=zero['adversarial_car_candidates'])
                selected.append(value)
            for domain in domains(selected):
                subset = [row for row in selected if domain == 'combined' or row['dataset'] == domain]
                result = metrics.detection_summary(subset, 1000, SEED, allow_zero_reference=True)
                result.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated, aggregation='global_candidate_score_ranking_with_one_target_per_sample', tnr_undefined_reason=None if result['tnr_defined'] else 'reference_AP50_is_zero')
                output.append(result)
    return output

def summarize_depth(metrics, rows):
    output = []
    for model in DEPTH_MODELS:
        for reference, evaluated in COMPARISONS:
            selected = [row for row in rows if row['model'] == model and row['reference_state'] == reference and (row['evaluated_state'] == evaluated)]
            for domain in domains(selected):
                subset = [row for row in selected if domain == 'combined' or row['dataset'] == domain]
                result = metrics.depth_summary(subset, 1000, SEED)
                result.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated, depth_representation=SEMANTICS[model], aggregation='per_sample_macro_mean_primary', reference='same_model_prediction_on_' + reference + '_scene')
                output.append(result)
    return output

def row_binding(case, state, random_binding):
    binding = {key: value for key, value in case['binding'].items() if key != 'patches'}
    binding['patch_binding'] = None if state == 'clean' else random_binding if state == 'random' else case['binding']['patches'][state]
    return binding

def evaluate_detection(args, cases, engine, metrics, r2, device, random_patch, random_binding):
    import torch
    car, mask = engine.load_car(args.car, args.car_mask, device)
    (args.output / 'raw_predictions').mkdir()
    rows, entries, geometry, inputs = ([], [], [], [])
    counts = dict.fromkeys(DETECTION_MODELS, 0)
    for model in DETECTION_MODELS:
        adapter = build_evaluation_detector(args, engine, model, device)
        try:
            with torch.no_grad():
                for index, case in enumerate(cases):
                    record = case['record']
                    context = engine.build_context(record, car, mask, device)
                    clean = {key: value for key, value in adapter.predict(context.benign)[0].items() if key in ('boxes', 'labels', 'scores')}
                    counts[model] += 1
                    clean_hash = r2.tensor_sha(context.benign)
                    payload = dict(case['binding'], model=model, split=record.split, target_box=list(context.object_box), clean=engine.cpu_predictions(clean), input_sha256={'clean': clean_hash})
                    inputs.append(dict(row_binding(case, 'clean', random_binding), model=model, state='clean', clean_input_sha256=clean_hash, rendered_input_sha256=clean_hash, support_tensor_sha256=None))
                    for state in STATES[1:]:
                        patch = random_patch if state == 'random' else case['patches'][state]
                        render = engine.RoadRGBView(patch.to(device)).render(context)
                        invariant = engine.invariant_record(context, render, record, model, state)
                        geometry.append(dict(invariant, sample_binding=row_binding(case, state, random_binding)))
                        image_hash = r2.tensor_sha(render.image)
                        inputs.append(dict(row_binding(case, state, random_binding), model=model, state=state, clean_input_sha256=clean_hash, rendered_input_sha256=image_hash, support_tensor_sha256=r2.tensor_sha(render.support)))
                        prediction = {key: value for key, value in adapter.predict(render.image)[0].items() if key in ('boxes', 'labels', 'scores')}
                        counts[model] += 1
                        payload[state] = engine.cpu_predictions(prediction)
                        payload['input_sha256'][state] = image_hash
                        row = dict(row_binding(case, state, random_binding), method=case['binding']['method_id'], model=model, split=record.split, state=state, used_in_training=detector_used_in_training(case['binding'], model), clean_candidates=engine.detection_candidates(clean, context.object_box, engine.VEHICLE_CLASS_IDS), adversarial_candidates=engine.detection_candidates(prediction, context.object_box, engine.VEHICLE_CLASS_IDS), clean_car_candidates=engine.detection_candidates(clean, context.object_box, engine.CAR_CLASS_IDS), adversarial_car_candidates=engine.detection_candidates(prediction, context.object_box, engine.CAR_CLASS_IDS))
                        rows.append(row)
                    name = 'raw_predictions/%s_%03d.pt.gz' % (model, index)
                    save_raw(args.output / name, payload)
                    entries.append(dict(case['binding'], model=model, path=name, sha256=sha(args.output / name), tensor_sha256={state: {key: r2.tensor_sha(value) for key, value in payload[state].items()} for state in STATES}))
                    print('SAME_SAMPLE_DETECTION %s %d/%d %s' % (model, index + 1, len(cases), record.sample_id), flush=True)
        finally:
            del adapter
            torch.cuda.empty_cache()
    count = len(cases)
    if len(rows) != count * 12 or len(entries) != count * 4 or counts != dict.fromkeys(DETECTION_MODELS, count * 4):
        raise ValueError('incomplete same-sample detector coverage')
    write_json(args.output / 'per_sample.json', rows)
    write_json(args.output / 'geometry_audit.json', {'status': 'pass', 'checks': geometry})
    write_json(args.output / 'rendered_input_hashes.json', {'rows': inputs})
    write_json(args.output / 'raw_predictions/MANIFEST.json', {'schema': 'sample-specific-detection-raw-v1', 'entries': entries, 'model_forward_count': count * 16, 'states': list(STATES)})
    result = {'status': 'same_sample_attack_evaluation_complete', 'model_forward_count': count * 16, 'records': len(rows), 'summaries': summarize_detection(metrics, rows), 'evaluation_model_scene_forwards': counts, 'metric_source_sha256': METRIC_SHA256, 'metric': 'tnr_det=(target_vehicle_AP50_clean-target_vehicle_AP50_final)/target_vehicle_AP50_clean'}
    write_json(args.output / 'detection_summary.json', result)
    return result

def evaluate_depth(args, cases, engine, metrics, r2, device, random_patch, random_binding):
    import torch
    from model_adapters import build_depth_adapter
    car, mask = engine.load_car(args.car, args.car_mask, device)
    (args.output / 'depth_raw/masks').mkdir(parents=True)
    reference = {(row['sample_id'], row['state']): row for row in read_json(args.output / 'rendered_input_hashes.json')['rows'] if row['model'] == DETECTION_MODELS[0]}
    rows, entries, geometry, inputs, masks = ([], [], [], [], {})
    counts = dict.fromkeys(DEPTH_MODELS, 0)
    for model in DEPTH_MODELS:
        adapter = build_depth_adapter(model, device, model_manifest=args.model_manifest)
        try:
            with torch.no_grad():
                for index, case in enumerate(cases):
                    record = case['record']
                    context = engine.build_context(record, car, mask, device)
                    mask_hash = r2.tensor_sha(context.object_mask)
                    if record.sample_id not in masks:
                        name = 'depth_raw/masks/%03d.pt.gz' % index
                        save_raw(args.output / name, context.object_mask.detach().cpu())
                        masks[record.sample_id] = dict(path=name, sha256=sha(args.output / name), tensor_sha256=mask_hash)
                    elif masks[record.sample_id]['tensor_sha256'] != mask_hash:
                        raise ValueError('object ROI changed across depth models')
                    payload = dict(case['binding'], schema='sample-specific-depth-raw-output-v1', model=model, split=record.split, semantics=SEMANTICS[model], object_mask=masks[record.sample_id], states={}, input_sha256={}, support_sha256={})
                    for state in STATES:
                        if state == 'clean':
                            image = context.benign
                        else:
                            patch = random_patch if state == 'random' else case['patches'][state]
                            render = engine.RoadRGBView(patch.to(device)).render(context)
                            image = render.image
                            geometry.append(dict(engine.invariant_record(context, render, record, model, state), sample_binding=row_binding(case, state, random_binding)))
                            support_hash = r2.tensor_sha(render.support)
                            if support_hash != reference[record.sample_id, state]['support_tensor_sha256']:
                                raise ValueError('depth/detection support differs')
                            payload['support_sha256'][state] = support_hash
                        image_hash = r2.tensor_sha(image)
                        if image_hash != reference[record.sample_id, state]['rendered_input_sha256']:
                            raise ValueError('depth/detection rendered input differs')
                        prediction = adapter.predict_with_diagnostics(image)
                        counts[model] += 1
                        if prediction.metadata.get('evaluation_semantics') != SEMANTICS[model]:
                            raise ValueError('depth evaluation native semantics differ')
                        value = prediction.stages.get('evaluation_output')
                        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                            raise ValueError('missing floating evaluation_output')
                        value = value.detach().cpu().contiguous()
                        if value.ndim == 3:
                            value = value.unsqueeze(1)
                        if value.shape != context.object_mask.shape or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                            raise ValueError('invalid native depth evaluation_output')
                        payload['states'][state] = value
                        payload['input_sha256'][state] = image_hash
                        inputs.append(dict(row_binding(case, state, random_binding), model=model, state=state, split=record.split, input_sha256=image_hash))
                        del prediction
                    for ref, evaluated in COMPARISONS:
                        row = dict(row_binding(case, evaluated, random_binding), method=case['binding']['method_id'], model=model, split=record.split, reference_state=ref, evaluated_state=evaluated, reference_patch_binding=None if ref == 'clean' else case['binding']['patches'][ref], used_in_training=depth_used_in_training(case['binding'], model))
                        row.update(metrics.depth_metrics(payload['states'][evaluated], payload['states'][ref], context.object_mask.detach().cpu(), semantics=SEMANTICS[model]))
                        rows.append(row)
                    name = 'depth_raw/%s_%03d.pt.gz' % (model, index)
                    save_raw(args.output / name, payload)
                    entries.append(dict(case['binding'], model=model, path=name, sha256=sha(args.output / name), object_mask=masks[record.sample_id], semantics=SEMANTICS[model], tensor_sha256={state: r2.tensor_sha(payload['states'][state]) for state in STATES}))
                    print('SAME_SAMPLE_DEPTH %s %d/%d %s' % (model, index + 1, len(cases), record.sample_id), flush=True)
        finally:
            del adapter
            torch.cuda.empty_cache()
    count = len(cases)
    if len(rows) != count * 16 or len(entries) != count * 4 or counts != dict.fromkeys(DEPTH_MODELS, count * 4):
        raise ValueError('incomplete same-sample depth coverage')
    write_json(args.output / 'depth_per_sample.json', rows)
    write_json(args.output / 'depth_geometry_audit.json', {'status': 'pass', 'checks': geometry})
    write_json(args.output / 'depth_input_hashes.json', {'rows': inputs, 'all_equal_to_detection_inputs': True})
    write_json(args.output / 'depth_raw/MANIFEST.json', {'schema': 'sample-specific-depth-raw-manifest-v1', 'entries': entries, 'states': list(STATES), 'unique_scenes': count, 'model_forward_count': count * 16, 'metric_input_stage': 'evaluation_output', 'comparisons': [list(pair) for pair in COMPARISONS], 'test_data_opened': False})
    result = {'status': 'same_sample_attack_evaluation_complete', 'model_forward_count': count * 16, 'records': len(rows), 'summaries': summarize_depth(metrics, rows), 'evaluation_model_scene_forwards': counts, 'metric_source_sha256': METRIC_SHA256, 'metric': 'native clean-reference prediction shift; no ground-truth depth error', 'no_cross_model_raw_depth_or_signed_inverse_mean': True}
    write_json(args.output / 'depth_summary.json', result)
    return result

def detector_used_in_training(binding, model):
    if model not in DETECTION_MODELS:
        raise ValueError('unknown evaluated detector')
    return model in role_metadata(binding)['source_models']

def depth_used_in_training(binding, model):
    if model not in DEPTH_MODELS:
        raise ValueError('unknown evaluated depth model')
    return model in role_metadata(binding)['source_models']


def build_evaluation_detector(args, engine, model, device):
    if model != 'yolov5s':
        return engine.build_offline_detector(model, device, args.model_manifest, args.weights_root)
    from yolov5_target import load_local_target, SharedTargetAdapter
    config = Path(args.output) / 'target_loader_config'
    config.mkdir(exist_ok=False)
    target = load_local_target(args.yolov5_model_binding, device, config)
    args.actual_YOLO_target_initialization_observation = target.initialization_observation
    adapter = SharedTargetAdapter(target)
    args.actual_YOLO_pre_NMS_anchor_metadata = adapter.decoded_anchor_records
    return adapter


METHOD_VARIANTS = {'ours5_b3_full_nuscenes100': 'custom'}
SOURCE_DETECTORS = ('fasterrcnn', 'ssd300')
SOURCE_DEPTHS = ('monodepth2', 'dpt_large')


def role_metadata(binding):
    method = binding.get('method_id')
    if (method not in METHOD_VARIANTS
            or binding.get('ablation_variant') != METHOD_VARIANTS[method]
            or binding.get('updates_per_sample') != 125):
        raise ValueError('actual approved Ours5 ablation identity and125 updates required')
    return dict(
        source_detector_models=list(SOURCE_DETECTORS),
        source_depth_models=list(SOURCE_DEPTHS),
        source_models=list(SOURCE_DETECTORS + SOURCE_DEPTHS),
        transfer_models=['yolov5s', 'fcos', 'depth_anything_v1', 'zoedepth'],
        model_roles={
            **{m: 'source_whitebox' if m in SOURCE_DETECTORS else 'cross_detector_transfer'
               for m in DETECTION_MODELS},
            **{m: 'source_whitebox' if m in SOURCE_DEPTHS else 'cross_depth_transfer'
               for m in DEPTH_MODELS}},
        all_victims_are_transfer=False)


def evaluate_ablation_cases(args, cases, engine, metrics, fingerprints,
                            device, random_patch, random_binding):
    if (len(cases) != 100
            or len({c['sample']['sample_id'] for c in cases}) != 100
            or Counter(c['sample']['dataset'] for c in cases) != {'nuscenes': 100}
            or Counter(c['sample']['source_split'] for c in cases) != {'train': 80, 'validation': 20}
            or len({c['binding']['method_id'] for c in cases}) != 1):
        raise ValueError('one actual method and nuScenes-only ordered100 own-input cases required')
    for case in cases:
        role_metadata(case['binding'])
        if set(case['patches']) != {'zero', 'final'}:
            raise ValueError('actual own-input zero/final patch pair required')
    detection = evaluate_detection(args, cases, engine, metrics, fingerprints,
                                   device, random_patch, random_binding)
    if len(args.actual_YOLO_pre_NMS_anchor_metadata) != 400:
        raise ValueError('complete400 YOLO anchor metadata observations required')
    write_json(Path(args.output) / 'YOLO_PRE_NMS_ANCHOR_METADATA.json',
               {'scope': 'D_ONLY_NOT_NATIVE_ANCHOR_BODY',
                'rows': args.actual_YOLO_pre_NMS_anchor_metadata})
    depth = evaluate_depth(args, cases, engine, metrics, fingerprints,
                           device, random_patch, random_binding)
    if detection['model_forward_count'] != 1600 or depth['model_forward_count'] != 1600:
        raise ValueError('all eight targets and four states must actually execute')
    return dict(
        schema=EVALUATION_SCHEMA, method_id=cases[0]['binding']['method_id'],
        ablation_variant=cases[0]['binding']['ablation_variant'],
        sample_count=100, updates_per_sample=125,
        detection=detection, depth=depth, model_forward_count=3200,
        new_optimizer_updates=0,
        model_forward_count_scope='benchmark_scene_forwards_only',
        actual_YOLO_target_initialization_observation=args.actual_YOLO_target_initialization_observation,
        evaluated_detectors=list(DETECTION_MODELS), evaluated_depth_models=list(DEPTH_MODELS),
        roles=role_metadata(cases[0]['binding']))
