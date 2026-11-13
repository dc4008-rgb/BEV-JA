#!/usr/bin/env python3
"""Evaluate each attack input with its own fixed-final Patch, never a shared final.

This is same-sample attack evaluation, not held-out cross-scene validation.
GPU execution is required for scientific inference. CPU tests check routing,
identity gates and the frozen metric aggregation, not model attack efficacy.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from artifact_io import (CAS_STORAGE as RAW_STORAGE, load_torch_lossless, save_torch_lossless,
                         tensor_blob_bindings, state_fingerprint, checked_pool)

sys.dont_write_bytecode = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[_name] = '1'

IDENTITY = 'ours_complete_eot1_parallel200_v1'
SEED = 20260919
DETECTION_MODELS = ('fasterrcnn', 'ssd300', 'maskrcnn', 'fcos')
DEPTH_MODELS = ('monodepth2', 'dpt_large', 'depth_anything_v1', 'zoedepth')
SEMANTICS = {'monodepth2': 'mono_scale_ambiguous_depth', 'dpt_large': 'relative_inverse_depth',
             'depth_anything_v1': 'relative_inverse_depth', 'zoedepth': 'metric_depth'}
STATES = ('clean', 'zero', 'random', 'final')
COMPARISONS = (('clean', 'zero'), ('clean', 'random'), ('clean', 'final'), ('zero', 'final'))
METRIC_SHA256 = '50bcf1aad80ae902d6e660017b1614d83c9a8a085b6a6cbe02a5402e31752c81'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
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
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe artifact path')
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.absolute().parents)) or not path.is_file():
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
    names = {'average_precision', 'tnr_detection', 'detection_summary',
             'depth_metrics', 'depth_summary', 'percentile_interval'}
    nodes = [node for node in ast.parse(Path(path).read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise ValueError('missing or ambiguous frozen metric definitions')
    namespace = dict(math=math, statistics=statistics, np=np, torch=torch, F=F)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def validate_summary(plan, sample, summary):
    from sample_contract import planned_budget, CONTROLLER_CONFIG
    expected = dict(identity=IDENTITY, status='train_optimization_complete',
                    plan_sha256=plan['_plan_sha256'], source_bindings=plan['source_bindings'],
                    executed_updates=plan['budget']['updates_per_sample'],
                    checkpoint_selection='fixed_final', test_data_opened=False,
                    validation_data_opened=sample['source_split'] == 'validation',
                    optimization_on_original_validation=sample['source_split'] == 'validation')
    expected.update({key: sample[key] for key in
                     ('sample_id', 'case_id', 'dataset', 'source_split', 'source_sha256')})
    for key, value in expected.items():
        if type(summary.get(key)) is not type(value) or summary[key] != value:
            raise ValueError('sample training summary binding differs: ' + key)
    if summary.get('scientific_config') != plan['scientific_config']:
        raise ValueError('sample scientific configuration differs')
    if summary.get('controller_config') != CONTROLLER_CONFIG:
        raise ValueError('sample dual controller differs')
    for key, value in {'task_autograd_calls': 2 * plan['budget']['updates_per_sample'],
                       'combined_scalar_autograd_calls': 0,
                       'all_predeclared_scenes_retained': True,
                       'controller_rule': 'periodic_gap_gated_foa_outer_reset'}.items():
        if type(summary.get(key)) is not type(value) or summary.get(key) != value:
            raise ValueError('dual task execution summary differs: ' + key)
    if summary.get('training_depth_models') != plan['scientific_config']['depth_models']:
        raise ValueError('actual depth training roles differ')
    updates = plan['budget']['updates_per_sample']
    expected_budget = planned_budget(updates)
    if any(summary.get('budget', {}).get(key) != expected_budget for key in ('planned', 'attempted', 'completed')):
        raise ValueError('actual per-sample call budget is incomplete or differs')
    for state, name in (('zero', 'zero_step_patch.pt'), ('final', 'final_patch.pt')):
        if name not in summary.get('patch_bindings', {}):
            raise ValueError('missing own-sample patch binding: ' + state)
    checkpoint = 'checkpoint_%04d.pt' % updates
    if checkpoint not in summary.get('checkpoint_bindings', {}):
        raise ValueError('missing fixed-final checkpoint binding')
    return summary


def validate_state_manifest(plan, sample, manifest):
    updates = plan['budget']['updates_per_sample']
    if (manifest.get('schema') != 'sample-specific-state-evidence-v1' or manifest.get('identity') != IDENTITY
            or manifest.get('status') != 'complete' or manifest.get('executed_updates') != updates
            or manifest.get('plan_sha256') != plan['_plan_sha256']
            or manifest.get('source_bindings') != plan['source_bindings']):
        raise ValueError('sample-specific state evidence binding/completion differs')
    if any(manifest.get(key) != sample[key] for key in ('sample_id', 'case_id', 'dataset', 'source_split', 'source_sha256')):
        raise ValueError('state evidence belongs to another sample')
    expected = [('initialization', 0)]
    for outer in range(updates // 25):
        expected.append(('outer_start', outer * 25))
        expected.extend((phase, step) for step in range(outer * 25 + 1, outer * 25 + 26)
                        for phase in ('before_denoise', 'pre_backward', 'pre_update', 'post_update'))
        expected.append(('outer_end', (outer + 1) * 25))
    entries = manifest.get('entries', [])
    if [(entry.get('phase'), entry.get('global_step')) for entry in entries] != expected:
        raise ValueError('incomplete/foreign sample-specific phase sequence')
    for entry in entries:
        require_sha(entry.get('state_fingerprint'))
        if entry.get('recording_preserved_rng') is not True:
            raise ValueError('state recording consumed scientific RNG')
    if manifest.get('resume_enabled') is not False:
        raise ValueError('new state evidence must not claim resume')
    retained = [entry for entry in entries if entry['phase'] == 'initialization' or
                (entry['phase'] == 'outer_end' and entry['global_step'] == updates)]
    boundaries = manifest.get('outer_boundary_checkpoints', [])
    keys = ('phase', 'global_step', 'state_fingerprint', 'path', 'sha256', 'bytes', 'retention_kind')
    if len(boundaries) != 2 or len(retained) != len(boundaries):
        raise ValueError('missing own-sample initialization/outer-boundary state')
    for entry, boundary in zip(retained, boundaries):
        if any(entry.get(key) != boundary.get(key) or key not in entry for key in keys):
            raise ValueError('retained state pointer differs from phase evidence')
    math_steps = [step for step in (1, 25, 500) if step <= updates]
    if manifest.get('fixed_update_math_steps') != math_steps:
        raise ValueError('fixed arithmetic checkpoints differ')
    probes = manifest.get('fixed_update_math_checks', [])
    if [(entry.get('phase'), entry.get('global_step')) for entry in probes] != [('pre_update', step) for step in math_steps]:
        raise ValueError('incomplete fixed task-gradient math evidence')
    for probe in probes:
        matching = [entry for entry in entries if entry['phase'] == 'pre_update' and entry['global_step'] == probe['global_step']]
        if len(matching) != 1 or any(probe.get(key) != matching[0].get(key) or key not in probe for key in keys):
            raise ValueError('math checkpoint differs from exact phase entry')
        if probe['retention_kind'] != 'fixed_update_math_probe':
            raise ValueError('math checkpoint retention role differs')
    return manifest


def validate_patch_metadata(payload, plan, sample, updates):
    metadata = payload.get('metadata', {})
    expected = {key: sample[key] for key in ('sample_id', 'case_id', 'source_sha256')}
    expected.update(plan_sha256=plan['_plan_sha256'], updates=updates)
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError('patch belongs to a different sample/budget: ' + key)


def load_cases(plan, runs_root, records, engine, r2, record_for_sample):
    """Fail on any missing/foreign case before any scientific inference."""
    import torch
    from case_owners200 import resolve_case_owner, case_origin_binding
    cases = []
    for sample in plan['samples']:
        owner_plan = resolve_case_owner(plan, sample)
        case_id = sample['case_id']
        if not isinstance(case_id, str) or Path(case_id).name != case_id or case_id in ('', '.', '..'):
            raise ValueError('unsafe case directory')
        run = Path(runs_root) / 'samples' / case_id / 'train'
        summary_path = checked_file(run, 'summary.json')
        summary = validate_summary(owner_plan, sample, read_json(summary_path))
        state_path = checked_file(run, 'state_evidence/MANIFEST.json')
        if sha(state_path) != require_sha(summary.get('state_evidence_manifest_sha256')):
            raise ValueError('state evidence manifest SHA differs')
        state_manifest = validate_state_manifest(owner_plan, sample, read_json(state_path))
        retained_states = {}
        for entry in [*state_manifest.get('outer_boundary_checkpoints', []), *state_manifest['fixed_update_math_checks']]:
            state_file = checked_file(run, entry['path'], {key: entry[key] for key in ('sha256', 'bytes')})
            payload = load_torch_lossless(state_file)
            if (payload.get('phase') != entry['phase'] or payload.get('global_step') != entry['global_step'] or
                    state_fingerprint(payload.get('state')) != entry['state_fingerprint']):
                raise ValueError('retained state payload/fingerprint differs')
            for key in ('sample_id', 'case_id', 'source_sha256'):
                if payload.get('metadata', {}).get(key) != sample[key]:
                    raise ValueError('retained state belongs to another sample')
            if payload.get('metadata', {}).get('plan_sha256') != owner_plan['_plan_sha256']:
                raise ValueError('retained state belongs to another plan')
            if payload.get('metadata', {}).get('identity') != IDENTITY:
                raise ValueError('retained state belongs to another method')
            if entry['retention_kind'] == 'initial_or_final_state':
                retained_states[entry['phase']] = (entry, payload['state'])
        record = record_for_sample(plan, sample, records)
        if sha(record.source_path) != sample['source_sha256']:
            raise ValueError('sample source image SHA differs')
        patches, bindings = {}, {}
        updates = plan['budget']['updates_per_sample']
        for state, name, count in (('zero', 'zero_step_patch.pt', 0), ('final', 'final_patch.pt', updates)):
            path = checked_file(run, name, summary['patch_bindings'][name])
            payload = torch.load(path, map_location='cpu', weights_only=True)
            validate_patch_metadata(payload, owner_plan, sample, count)
            patches[state] = engine.load_patch(path)
            bindings[state] = dict(path=str(path.resolve()), **summary['patch_bindings'][name],
                                   tensor_sha256=r2.tensor_sha(patches[state]), sample_id=sample['sample_id'],
                                   case_id=case_id, source_sha256=sample['source_sha256'])
        checkpoint_name = 'checkpoint_%04d.pt' % updates
        checkpoint_path = checked_file(run, checkpoint_name, summary['checkpoint_bindings'][checkpoint_name])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        validate_patch_metadata(checkpoint, owner_plan, sample, updates)
        if not torch.equal(patches['final'], engine.load_patch(checkpoint_path)):
            raise ValueError('own final differs from own fixed-final checkpoint')
        entry, initialization = retained_states['initialization']
        raw_path = checked_file(run, 'input_raw512.pt', summary['patch_bindings']['input_raw512.pt'])
        raw_payload = load_torch_lossless(raw_path)
        validate_patch_metadata(raw_payload, owner_plan, sample, 0)
        if not torch.equal(raw_payload['raw_generator_rgb'], initialization['initial_patch']):
            raise ValueError('initial raw512 state differs from bound own input')
        if not torch.equal(initialization['initial_small'], patches['zero']):
            raise ValueError('zero is not the retained own initial canonical patch')
        final_payload = torch.load(run / 'final_patch.pt', map_location='cpu', weights_only=True)
        if not torch.equal(final_payload['raw_generator_rgb'], retained_states['outer_end'][1]['raw_patch']):
            raise ValueError('fixed final raw512 differs from retained final optimization state')
        trace_path = checked_file(run, 'attack_trace.jsonl')
        if sha(trace_path) != require_sha(summary.get('trace_sha256')):
            raise ValueError('sample update trace SHA differs')
        trace = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line]
        if len(trace) != updates or any(row.get('global_step') != index or
                row.get('sample_id') != sample['sample_id'] or row.get('dataset') != sample['dataset'] or
                row.get('split') != sample['source_split'] or
                row.get('autograd_calls_this_step') != 2 for index, row in enumerate(trace, 1)):
            raise ValueError('update trace is not complete same-sample optimization')
        cases.append(dict(sample=sample, record=record, run=run, patches=patches,
                          binding=dict(sample_id=sample['sample_id'], case_id=case_id,
                                       dataset=sample['dataset'], source_split=sample['source_split'],
                                       source_sha256=sample['source_sha256'], plan_sha256=owner_plan['_plan_sha256'],
                                       evaluation_plan_sha256=plan['_plan_sha256'],
                                       training_origin=case_origin_binding(plan, sample),
                                       train_summary_sha256=sha(summary_path), patches=bindings)))
    if len(cases) != len(plan['samples']) or len({case['sample']['sample_id'] for case in cases}) != len(cases):
        raise ValueError('sample coverage/identity is not one-to-one')
    return cases


def domains(rows):
    return tuple(domain for domain in ('kitti', 'nuscenes', 'combined')
                 if domain == 'combined' or any(row['dataset'] == domain for row in rows))


def summarize_detection(metrics, rows):
    """Combined AP ranks candidates globally across all samples, not domain means."""
    output = []
    for model in DETECTION_MODELS:
        for reference, evaluated in COMPARISONS:
            by_state = {(row['dataset'], row['sample_id'], row['state']): row
                        for row in rows if row['model'] == model}
            selected = []
            for row in rows:
                if row['model'] != model or row['state'] != evaluated:
                    continue
                value = dict(row)
                if reference == 'zero':
                    zero = by_state[(row['dataset'], row['sample_id'], 'zero')]
                    value.update(clean_candidates=zero['adversarial_candidates'],
                                 clean_car_candidates=zero['adversarial_car_candidates'])
                selected.append(value)
            for domain in domains(selected):
                subset = [row for row in selected if domain == 'combined' or row['dataset'] == domain]
                result = metrics.detection_summary(subset, 1000, SEED, allow_zero_reference=True)
                result.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated,
                              aggregation='global_candidate_score_ranking_with_one_target_per_sample',
                              tnr_undefined_reason=None if result['tnr_defined'] else 'reference_AP50_is_zero')
                output.append(result)
    return output


def summarize_depth(metrics, rows):
    output = []
    for model in DEPTH_MODELS:
        for reference, evaluated in COMPARISONS:
            selected = [row for row in rows if row['model'] == model and row['reference_state'] == reference
                        and row['evaluated_state'] == evaluated]
            for domain in domains(selected):
                subset = [row for row in selected if domain == 'combined' or row['dataset'] == domain]
                result = metrics.depth_summary(subset, 1000, SEED)
                result.update(model=model, dataset=domain, reference_state=reference, evaluated_state=evaluated,
                              depth_representation=SEMANTICS[model], aggregation='per_sample_macro_mean_primary',
                              reference='same_model_prediction_on_' + reference + '_scene')
                output.append(result)
    return output


def row_binding(case, state, random_binding):
    binding = {key: value for key, value in case['binding'].items() if key != 'patches'}
    binding['patch_binding'] = (None if state == 'clean' else random_binding if state == 'random'
                                else case['binding']['patches'][state])
    return binding


def evaluate_detection(args, cases, engine, metrics, r2, device, random_patch, random_binding):
    import torch
    car, mask = engine.load_car(args.car, args.car_mask, device)
    (args.output / 'raw_predictions').mkdir()
    rows, entries, geometry, inputs = [], [], [], []
    counts = dict.fromkeys(DETECTION_MODELS, 0)
    for model in DETECTION_MODELS:
        adapter = engine.build_offline_detector(model, device, args.model_manifest, args.weights_root)
        try:
            with torch.no_grad():
                for index, case in enumerate(cases):
                    record = case['record']
                    context = engine.build_context(record, car, mask, device)
                    if model == DETECTION_MODELS[0]:
                        expected_zero = engine.canonicalize_rgb(engine.extract_road_patch(context.background, context.road_quad))
                        if not torch.equal(case['patches']['zero'].to(device), expected_zero):
                            raise ValueError('zero is not this sample own unpatched road crop')
                    clean = {key: value for key, value in adapter.predict(context.benign)[0].items()
                             if key in ('boxes', 'labels', 'scores')}
                    counts[model] += 1
                    clean_hash = r2.tensor_sha(context.benign)
                    payload = dict(case['binding'], model=model, split=record.split, target_box=list(context.object_box),
                                   clean=engine.cpu_predictions(clean), input_sha256={'clean': clean_hash})
                    inputs.append(dict(row_binding(case, 'clean', random_binding), model=model, state='clean',
                                       clean_input_sha256=clean_hash, rendered_input_sha256=clean_hash,
                                       support_tensor_sha256=None))
                    for state in STATES[1:]:
                        patch = random_patch if state == 'random' else case['patches'][state]
                        render = engine.RoadRGBView(patch.to(device)).render(context)
                        invariant = engine.invariant_record(context, render, record, model, state)
                        geometry.append(dict(invariant, sample_binding=row_binding(case, state, random_binding)))
                        image_hash = r2.tensor_sha(render.image)
                        inputs.append(dict(row_binding(case, state, random_binding), model=model, state=state,
                                           clean_input_sha256=clean_hash, rendered_input_sha256=image_hash,
                                           support_tensor_sha256=r2.tensor_sha(render.support)))
                        prediction = {key: value for key, value in adapter.predict(render.image)[0].items()
                                      if key in ('boxes', 'labels', 'scores')}
                        counts[model] += 1
                        payload[state] = engine.cpu_predictions(prediction)
                        payload['input_sha256'][state] = image_hash
                        row = dict(row_binding(case, state, random_binding), method=IDENTITY, model=model,
                                   split=record.split, state=state, used_in_training=model in DETECTION_MODELS[:2],
                                   clean_candidates=engine.detection_candidates(clean, context.object_box, engine.VEHICLE_CLASS_IDS),
                                   adversarial_candidates=engine.detection_candidates(prediction, context.object_box, engine.VEHICLE_CLASS_IDS),
                                   clean_car_candidates=engine.detection_candidates(clean, context.object_box, engine.CAR_CLASS_IDS),
                                   adversarial_car_candidates=engine.detection_candidates(prediction, context.object_box, engine.CAR_CLASS_IDS))
                        rows.append(row)
                    name = 'raw_predictions/%s_%03d.pt.gz' % (model, index)
                    save_torch_lossless(args.output / name, payload, pool_root=args.tensor_pool)
                    entries.append(dict(case['binding'], model=model, path=name, sha256=sha(args.output / name),
                                        tensor_sha256={state: {key: r2.tensor_sha(value) for key, value in payload[state].items()}
                                                       for state in STATES}))
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
    write_json(args.output / 'raw_predictions/MANIFEST.json', {'schema': 'sample-specific-detection-raw-v1',
               'entries': entries, 'model_forward_count': count * 16, 'states': list(STATES)})
    result = {'status': 'same_sample_attack_evaluation_complete', 'model_forward_count': count * 16,
              'records': len(rows), 'summaries': summarize_detection(metrics, rows),
              'evaluation_model_scene_forwards': counts, 'metric_source_sha256': METRIC_SHA256,
              'metric': 'tnr_det=(target_vehicle_AP50_clean-target_vehicle_AP50_final)/target_vehicle_AP50_clean'}
    write_json(args.output / 'detection_summary.json', result)
    return result


def evaluate_depth(args, cases, engine, metrics, r2, device, random_patch, random_binding):
    import torch
    from model_adapters import build_depth_adapter
    car, mask = engine.load_car(args.car, args.car_mask, device)
    (args.output / 'depth_raw/masks').mkdir(parents=True)
    reference = {(row['sample_id'], row['state']): row
                 for row in read_json(args.output / 'rendered_input_hashes.json')['rows']
                 if row['model'] == DETECTION_MODELS[0]}
    rows, entries, geometry, inputs, masks = [], [], [], [], {}
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
                        save_torch_lossless(args.output / name, context.object_mask.detach().cpu(), pool_root=args.tensor_pool)
                        masks[record.sample_id] = dict(path=name, sha256=sha(args.output / name), tensor_sha256=mask_hash)
                    elif masks[record.sample_id]['tensor_sha256'] != mask_hash:
                        raise ValueError('object ROI changed across depth models')
                    payload = dict(case['binding'], schema='sample-specific-depth-raw-output-v1', model=model,
                                   split=record.split, semantics=SEMANTICS[model], object_mask=masks[record.sample_id],
                                   states={}, input_sha256={}, support_sha256={})
                    for state in STATES:
                        if state == 'clean':
                            image = context.benign
                        else:
                            patch = random_patch if state == 'random' else case['patches'][state]
                            render = engine.RoadRGBView(patch.to(device)).render(context)
                            image = render.image
                            geometry.append(dict(engine.invariant_record(context, render, record, model, state),
                                                 sample_binding=row_binding(case, state, random_binding)))
                            support_hash = r2.tensor_sha(render.support)
                            if support_hash != reference[(record.sample_id, state)]['support_tensor_sha256']:
                                raise ValueError('depth/detection support differs')
                            payload['support_sha256'][state] = support_hash
                        image_hash = r2.tensor_sha(image)
                        if image_hash != reference[(record.sample_id, state)]['rendered_input_sha256']:
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
                        inputs.append(dict(row_binding(case, state, random_binding), model=model, state=state,
                                           split=record.split, input_sha256=image_hash))
                        del prediction
                    for ref, evaluated in COMPARISONS:
                        row = dict(row_binding(case, evaluated, random_binding), method=IDENTITY, model=model,
                                   split=record.split, reference_state=ref, evaluated_state=evaluated,
                                   reference_patch_binding=None if ref == 'clean' else case['binding']['patches'][ref],
                                   used_in_training=model in args.training_depth_models)
                        row.update(metrics.depth_metrics(payload['states'][evaluated], payload['states'][ref],
                                                         context.object_mask.detach().cpu(), semantics=SEMANTICS[model]))
                        rows.append(row)
                    name = 'depth_raw/%s_%03d.pt.gz' % (model, index)
                    save_torch_lossless(args.output / name, payload, pool_root=args.tensor_pool)
                    entries.append(dict(case['binding'], model=model, path=name, sha256=sha(args.output / name),
                                        object_mask=masks[record.sample_id], semantics=SEMANTICS[model],
                                        tensor_sha256={state: r2.tensor_sha(payload['states'][state]) for state in STATES}))
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
    write_json(args.output / 'depth_raw/MANIFEST.json', {'schema': 'sample-specific-depth-raw-manifest-v1',
               'entries': entries, 'states': list(STATES), 'unique_scenes': count, 'model_forward_count': count * 16,
               'metric_input_stage': 'evaluation_output', 'comparisons': [list(pair) for pair in COMPARISONS],
               'test_data_opened': False})
    result = {'status': 'same_sample_attack_evaluation_complete', 'model_forward_count': count * 16,
              'records': len(rows), 'summaries': summarize_depth(metrics, rows),
              'evaluation_model_scene_forwards': counts, 'metric_source_sha256': METRIC_SHA256,
              'metric': 'native clean-reference prediction shift; no ground-truth depth error',
              'no_cross_model_raw_depth_or_signed_inverse_mean': True}
    write_json(args.output / 'depth_summary.json', result)
    return result


def collect_pool_bindings(output):
    result = {}
    for path in sorted(Path(output).rglob('*.pt.gz')):
        stored = load_torch_lossless(path)
        for name, binding in tensor_blob_bindings(stored).items():
            if name in result and result[name] != binding:
                raise ValueError('inconsistent raw tensor-pool binding')
            result[name] = binding
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'runs-root', 'output', 'child', 'r2-source', 'v2-source', 'clip-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--weights-root', type=Path)
    parser.add_argument('--source-prefix', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--tensor-pool', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    from sample_contract import load_plan, record_for_sample
    from source_bindings import verify_sources
    plan = load_plan(args.plan)
    args.tensor_pool = checked_pool(args.tensor_pool)
    args.training_depth_models = tuple(plan['scientific_config']['depth_models'])
    if socket.gethostname() != args.expected_hostname:
        raise ValueError('live hostname mismatch')
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError('fresh evaluation output with an existing parent required')
    if any(path.is_symlink() for path in (args.output, *args.output.absolute().parents)):
        raise ValueError('output symlinks forbidden')
    args.output, args.runs_root = args.output.resolve(), args.runs_root.resolve()
    if args.output == args.runs_root or args.output in args.runs_root.parents or args.runs_root in args.output.parents:
        raise ValueError('evaluation output must be outside training runs')
    if (args.source_prefix is None) != (args.source_root is None):
        raise ValueError('source remapping needs both paths')
    sources = verify_sources(plan, child=args.child, r2_source=args.r2_source,
                             v2_source=args.v2_source, clip_source=args.clip_source)
    sys.path.insert(0, str(sources['r2_source']))
    r2 = module_from_path('_sample_specific_r2_evaluate', sources['r2_source'] / 'evaluate.py')
    r2.configure()
    import torch
    import torchvision
    from importlib.metadata import version
    if (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0], version('diffusers')) != ('2.8.0', '0.23.0', '0.38.0'):
        raise ValueError('frozen runtime versions differ')
    engine, parent = r2.load_engine(sources['child'])
    metrics = load_metric_functions(parent / '04_统一评估/evaluator.py')
    args.model_manifest = parent / '02_模型/model_manifest.json'
    args.car, args.car_mask = parent / '01_公共数据集/car_00.jpg', parent / '01_公共数据集/car_00_mask.jpg'
    records = engine.training_records(parent / '01_公共数据集/runtime_v1/train_validation_manifest.csv',
                                      args.source_prefix, args.source_root)
    cases = load_cases(plan, args.runs_root, records, engine, r2, record_for_sample)
    # Only selected attack inputs are opened. Never use the old fixed40 validation selector.
    from road_runtime import checked_file as asset_checked_file
    selected = [case['record'] for case in cases]
    preflight = engine.preflight(selected, DETECTION_MODELS, args.model_manifest, args.car, args.car_mask, args.weights_root)
    clip_eval = module_from_path('_sample_specific_depth_asset_helpers', sources['clip_source'] / 'clip_evaluate.py')
    preflight['depth'] = clip_eval.depth_preflight(args.model_manifest, asset_checked_file)
    if not preflight['inputs_ready'] or not preflight['depth']['inputs_ready'] or not torch.cuda.is_available():
        raise RuntimeError('frozen evaluation assets and CUDA required')
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / 'input_preflight.json', preflight)
        write_json(args.output / 'case_bindings.json', {'cases': [case['binding'] for case in cases]})
        random_patch = engine.common_random_patch()
        random_path = args.output / 'random_control_patch.pt'
        engine.save_patch(random_path, random_patch, {'state': 'unoptimized_random_control', 'seed': SEED})
        random_binding = dict(path=str(random_path), sha256=sha(random_path), bytes=random_path.stat().st_size,
                              tensor_sha256=r2.tensor_sha(random_patch), role='common_random_control_only')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        detector = evaluate_detection(args, cases, engine, metrics, r2, device, random_patch, random_binding)
        depth = evaluate_depth(args, cases, engine, metrics, r2, device, random_patch, random_binding)
        torch.cuda.synchronize(device)
        with (args.output / 'PLAN.json').open('xb') as stream:
            stream.write(args.plan.read_bytes())
        result = {'schema': 'r2-joint-sample-specific-eight-model-evaluation-v1',
                  'identity': IDENTITY, 'status': 'same_sample_attack_evaluation_complete',
                  'evaluation_role': 'same_sample_optimization_and_evaluation_not_heldout_transfer',
                  'plan_sha256': plan['_plan_sha256'], 'ablation_variant': plan['ablation_variant'], 'source_bindings': plan['source_bindings'],
                  'sample_count': len(cases), 'sample_ids': [case['sample']['sample_id'] for case in cases],
                  'source_split_counts': dict(Counter(case['sample']['source_split'] for case in cases)),
                  'updates_per_sample': plan['budget']['updates_per_sample'],
                  'evaluation_forward_total': len(cases) * 32, 'states': list(STATES),
                  'evaluation_models': [*DETECTION_MODELS, *DEPTH_MODELS],
                  'source_models': [*DETECTION_MODELS[:2], *args.training_depth_models],
                  'transfer_models': [*DETECTION_MODELS[2:], *[name for name in DEPTH_MODELS if name not in args.training_depth_models]],
                  'controller_config': plan['scientific_config']['controller_config'], 'raw_storage': RAW_STORAGE,
                  'tensor_pool_root': str(args.tensor_pool.resolve()),
                  'tensor_pool_bindings': collect_pool_bindings(args.output),
                  'same_sample_patch_routing': True, 'fixed_final': True, 'metric_source_sha256': METRIC_SHA256,
                  'test_data_opened': False, 'heldout_generalization_claim_allowed': False,
                  'detection_summary_sha256': sha(args.output / 'detection_summary.json'),
                  'depth_summary_sha256': sha(args.output / 'depth_summary.json'),
                  'detection_summary_rows': len(detector['summaries']), 'depth_summary_rows': len(depth['summaries']),
                  'seconds': time.monotonic() - started, 'hostname': socket.gethostname(),
                  'runtime': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                              'diffusers': version('diffusers'), 'transformers': version('transformers')},
                  'artifact_bindings': {str(path.relative_to(args.output)): sha(path)
                                        for path in sorted(args.output.rglob('*')) if path.is_file()}}
        write_json(args.output / 'summary.json', result)
        print(json.dumps({'status': result['status'], 'samples': len(cases),
                          'forwards': result['evaluation_forward_total']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__,
                                                'error': str(error), 'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
