#!/usr/bin/env python3
"""Fixed final100 v29 U1/lr evaluation using the unchanged frozen v17 metric engine."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import time

if __package__:
    from .inputs import (ROOT, PARENT, IDENTITY, candidate_run_path, load_protocol,
                         parent_evaluator, sha, verify_inputs, LEARNING_RATE, VARIANT_LABEL, EVALUATION_MODELS)
else:
    from inputs import (ROOT, PARENT, IDENTITY, candidate_run_path, load_protocol,
                        parent_evaluator, sha, verify_inputs, LEARNING_RATE, VARIANT_LABEL, EVALUATION_MODELS)

PHASES = ('before_denoise', 'gradients', 'pre_update', 'post_update')
OBJECTIVES = ('U1',)
CORE_MODELS = ('fasterrcnn', 'ssd300')
MATHEMATICAL_OBJECTIVE = 'C0'
OBJECTIVE_DESCRIPTIONS = {'U1': 'Both detectors retain original post-NMS max(vehicle score * target IoU), vehicle classes 3/6/8.'}


def require_current_objective(value, expected=None):
    objective = value.get('update_variant')
    if ('ssd_objective' in value or objective not in OBJECTIVES or
            (expected is not None and objective != expected) or
            value.get('candidate_id') != IDENTITY+'-ours-det-only-'+objective):
        raise ValueError('current v29 unique-U1 C0 objective identity required; legacy SSD-only identity is rejected')
    return objective


def validate_objective_diagnostics(value, model):
    """Validate the fixed original C0/C1 branch and its recorded classes."""
    required = {'mathematical_objective':MATHEMATICAL_OBJECTIVE,
                'model_id':model,'training_target_classes':[3,6,8],
                'loss_branch':('C1_native_matched' if MATHEMATICAL_OBJECTIVE=='C1' and model=='ssd300' else 'C0_post_nms')}
    if MATHEMATICAL_OBJECTIVE=='C1' and model=='ssd300':
        required.update(objective='c1_native_gt_matched_pre_nms_vehicle_group_probability',
                        vehicle_classes=[3,6,8],direction='minimize',
                        matching_rule='native_ssd_matcher_including_best_anchor_for_gt',
                        matching_iou_threshold=.5, selection_uses_predictions_or_gradients=False,
                        cross_level_forced_matches=False, model_forward_equivalents=1,
                        native_adapter_predict_calls=1)
        if (not isinstance(value,dict) or type(value.get('matched_prediction_count')) is not int
                or value['matched_prediction_count'] <= 0):
            raise ValueError('fixed-loss diagnostics require native SSD positives: '+model)
    if (not isinstance(value,dict) or any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in required.items())):
        raise ValueError('fixed-loss diagnostics differ: '+model)


def validate_loss_observations(value, objective):
    if not isinstance(value,dict) or set(value) != set(CORE_MODELS):
        raise ValueError('both detector loss observations are required')
    for model, details in value.items():
        if not isinstance(details,dict):
            raise ValueError('invalid per-detector loss observations')
        for key in ('raw','normalized'):
            x = details.get(key)
            if type(x) not in (int,float) or not math.isfinite(x):
                raise ValueError('finite raw and normalized detector observations are required')
        if objective in OBJECTIVES:
            draws = details.get('objective_diagnostics_by_eot_draw')
            if not isinstance(draws,list) or len(draws) != 2:
                raise ValueError('both original EOT draws require fixed-loss evidence')
            for draw in draws:
                validate_objective_diagnostics(draw,model)


def read_json(path):
    def reject(value):
        raise ValueError('non-finite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def safe_file(base, relative):
    base, relative = Path(base), Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('unsafe evidence path')
    for i in range(len(relative.parts) + 1):
        if (base / Path(*relative.parts[:i])).is_symlink():
            raise ValueError('symlink evidence is forbidden')
    path = base / relative
    if not path.is_file():
        raise ValueError('missing evidence: ' + str(relative))
    return path


def bound_file(base, name, digest):
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('invalid SHA binding: ' + str(name))
    path = safe_file(base, name)
    if sha(path) != digest:
        raise ValueError('artifact SHA differs: ' + str(name))
    return path


def typed_tensor_sha(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(canonical_json({'dtype': str(value.dtype), 'shape': list(value.shape)}))
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_calibration(run, summary, protocol, records):
    value = read_json(bound_file(run, 'calibration.json', summary.get('calibration_sha256')))
    objective = require_current_objective(summary)
    require_current_objective(value,objective)
    if value.get('candidate_id') != summary['candidate_id']:
        raise ValueError('calibration candidate identity differs')
    if value.get('update_variant') != objective or value.get('protocol_sha256') != summary['protocol_sha256']:
        raise ValueError('calibration objective differs')
    required = {'schema': 'v29_fixed_train_loss_calibration_v1',
                'source': 'fixed_first_train_road_crop_before_vae_noise_or_optimization',
                'eot': None, 'victim_forward_calls': 8, 'autograd_calls': 0,
                'model_loss_attempts': 8, 'model_loss_completed': 8,
                'optimized_v20_states_used': False, 'test_data_opened': False,
                'validation_data_opened': False, 'updated_parameters': 0}
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in required.items()):
        raise ValueError('calibration scope/identity differs')
    ids = protocol['calibration']['sample_ids']
    if value.get('sample_ids') != ids or summary.get('calibration_sample_ids') != ids:
        raise ValueError('calibration must bind the four fixed train identities')
    samples = value.get('samples', [])
    if len(samples) != 4 or [x.get('sample_id') for x in samples] != ids:
        raise ValueError('calibration sample evidence differs')
    originals = {r.sample_id: r for r in records}
    for row in samples:
        record = originals[row['sample_id']]
        if (row.get('split') != 'train' or row.get('dataset') != record.dataset or row.get('eot') is not None or
                row.get('road_quad_norm') != record.road_quad_norm):
            raise ValueError('calibration used a different split/scene/EOT')
        if hasattr(record, 'source_sha256') and row.get('source_sha256') != record.source_sha256:
            raise ValueError('calibration source identity differs')
        if objective in OBJECTIVES:
            for model in CORE_MODELS:
                validate_objective_diagnostics(row.get('models',{}).get(model,{}).get('diagnostics'),model)
    normalizers = summary.get('normalizers')
    if (not isinstance(normalizers, dict) or set(normalizers) != set(CORE_MODELS) or
            set(value.get('models', {})) != set(CORE_MODELS) or
            any(set(row.get('models', {})) != set(CORE_MODELS) for row in samples) or
            value.get('normalizers') != normalizers):
        raise ValueError('calibration/summary normalizers differ')
    for model, normalizer in normalizers.items():
        if type(normalizer) not in (int, float) or not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError('invalid calibrated scale')
        raw = [row['models'][model]['raw_loss'] for row in samples]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in raw):
            raise ValueError('invalid raw calibration loss')
        absolute = [abs(x) for x in raw]
        mean = sum(absolute) / 4
        evidence = value['models'][model]
        if (evidence.get('raw_sample_absolute_losses') != absolute or
                evidence.get('raw_mean_absolute') != mean or evidence.get('floor') != 1e-6 or
                evidence.get('floor_applied') is not (mean < 1e-6) or
                evidence.get('normalizer') != normalizer or normalizer != max(mean, 1e-6)):
            raise ValueError('calibration arithmetic differs')
        if any(row['models'][model].get('absolute_loss') != abs(row['models'][model]['raw_loss']) for row in samples):
            raise ValueError('calibration absolute loss differs')
    return value


def validate_states(run, summary, trace):
    import torch
    manifest = read_json(bound_file(run, 'state_evidence/MANIFEST.json', summary.get('state_evidence_manifest_sha256')))
    require_current_objective(manifest,summary['update_variant'])
    if (manifest.get('identity') != IDENTITY or manifest.get('latent_lr') != LEARNING_RATE or
            manifest.get('learning_rate_label') != VARIANT_LABEL or
            manifest.get('schema') != 'v29-training-state-manifest-v1' or
            manifest.get('candidate_id') != summary['candidate_id'] or
            manifest.get('update_variant') != summary['update_variant'] or
            manifest.get('status') != 'complete' or type(manifest.get('executed_updates')) is not int or
            manifest.get('executed_updates') != 100 or manifest.get('resume_enabled') is not False):
        raise ValueError('full-state manifest is not a completed v29 run')
    ledger = bound_file(run, 'state_evidence/ledger.jsonl', manifest.get('ledger_sha256'))
    entries = manifest.get('entries', [])
    expected_order = [('initialization', 0, None)]
    for outer in range(1, 5):
        expected_order.append(('outer_start', 0, outer))
        expected_order.extend((phase, step, None) for step in range((outer-1)*25+1, outer*25+1) for phase in PHASES)
        expected_order.append(('outer_end', 0, outer))
    if len(entries) != 409:
        raise ValueError('full training requires 409 states including initialization and four outer rounds')
    if [json.loads(line) for line in ledger.read_text().splitlines()] != entries:
        raise ValueError('state ledger/manifest differs')
    previous, seen, step_entries, ordered_phases = None, set(), {}, []
    for index, entry in enumerate(entries, 1):
        unsigned = {k: v for k, v in entry.items() if k != 'entry_sha256'}
        if (type(entry.get('index')) is not int or entry['index'] != index or
                entry.get('previous_entry_sha256') != previous or
                entry.get('entry_sha256') != hashlib.sha256(canonical_json(unsigned)).hexdigest()):
            raise ValueError('state ledger hash chain differs')
        previous = entry['entry_sha256']
        name, step, phase = entry['path'], entry['global_step'], entry['phase']
        expected_phase, expected_step, expected_outer = expected_order[index-1]
        if (phase, step) != (expected_phase, expected_step):
            raise ValueError('full state sequence differs from the four fixed outer rounds')
        if name in seen or not name.startswith('state_evidence/') or type(step) is not int or not 0 <= step <= 100:
            raise ValueError('duplicate/invalid full-state evidence')
        seen.add(name)
        path = bound_file(run, name, entry['sha256'])
        if type(entry.get('bytes')) is not int or path.stat().st_size != entry['bytes']:
            raise ValueError('state evidence length differs')
        if (entry.get('recording_preserved_rng') is not True or
                entry.get('recording_rng_before_sha256') != entry.get('recording_rng_after_sha256')):
            raise ValueError('state recording changed scientific RNG')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        metadata = payload.get('metadata', {})
        require_current_objective(metadata,summary['update_variant'])
        if (metadata.get('identity') != IDENTITY or metadata.get('latent_lr') != LEARNING_RATE or
                metadata.get('learning_rate_label') != VARIANT_LABEL or
                payload.get('schema') != 'v29-training-state-v1' or not isinstance(payload.get('state'), dict) or
                metadata.get('candidate_id') != summary['candidate_id'] or
                metadata.get('update_variant') != summary['update_variant'] or
                type(metadata.get('global_step')) is not int or metadata.get('global_step') != step or
                metadata.get('phase') != phase or metadata.get('resume_enabled') is not False):
            raise ValueError('state payload identity differs')
        state = payload['state']
        if expected_outer is not None and (type(state.get('outer_round')) is not int or state['outer_round'] != expected_outer):
            raise ValueError('outer state round identity differs')
        if phase == 'initialization' and (canonical_json(state.get('config')) != canonical_json(summary['config']) or
                state.get('normalizers') != summary['normalizers'] or state.get('calibration_sha256') != summary['calibration_sha256']):
            raise ValueError('initial full-state configuration/calibration differs')
        if phase == 'before_denoise':
            if (state.get('calibration_sha256') != summary['calibration_sha256'] or
                    state.get('normalizers') != summary['normalizers'] or
                    any(state.get('planned', {}).get(k) != trace[step-1].get(k) for k in
                        ('global_step','outer_round','inner_step','sample_id','dataset','noise_seed','render_seed'))):
                raise ValueError('before-state plan/calibration differs')
        if phase in ('gradients','post_update'):
            field = 'losses_pre' if phase == 'gradients' else 'losses_post'
            validate_loss_observations(state.get(field),summary['update_variant'])
            if canonical_json(state[field]) != canonical_json(trace[step-1].get(field)):
                raise ValueError('saved detector objective observations differ from trace')
        if phase == 'pre_update':
            if (state.get('latent_lr') != LEARNING_RATE or
                    state.get('learning_rate_label') != VARIANT_LABEL):
                raise ValueError('saved pre-update learning rate differs from this child')
            pending_pre_update = state
        if phase == 'post_update':
            if __package__:
                from .update_rule import validate_saved_update
            else:
                from update_rule import validate_saved_update
            validate_saved_update(pending_pre_update, state, summary['update_variant'])
            if state.get('update_statistics') != trace[step-1].get('update_statistics'):
                raise ValueError('trace update statistics differ')
        del payload, state
        if phase in PHASES:
            if not 1 <= step <= 100 or (step, phase) in step_entries:
                raise ValueError('duplicate or invalid update phase')
            step_entries[step, phase] = {'path': name, 'sha256': entry['sha256']}
            ordered_phases.append((step, phase))
        elif step != 0 or phase not in ('initialization', 'outer_start', 'outer_end'):
            raise ValueError('unexpected completed-run state phase')
    if set(step_entries) != {(step, phase) for step in range(1, 101) for phase in PHASES}:
        raise ValueError('all 100 before/gradient/pre/post states are required')
    if ordered_phases != [(step, phase) for step in range(1, 101) for phase in PHASES]:
        raise ValueError('update states are not in the execution order')
    for step, row in enumerate(trace, 1):
        if row.get('state_bindings') != {phase: step_entries[step, phase] for phase in PHASES}:
            raise ValueError('trace is not bound to every full update state')
    actual_pt = {str(p.relative_to(run)) for p in (run / 'state_evidence').glob('*.pt')}
    if actual_pt != seen:
        raise ValueError('state evidence file set differs')
    return {'entries': len(entries), 'complete_update_phases': 400,
            'manifest_sha256': summary['state_evidence_manifest_sha256'], 'all_hashes_verified': True}


def validate_completed_run(run, records, training_manifest, model_manifest, *, root=ROOT, protocol=None):
    import torch
    protocol = load_protocol() if protocol is None else protocol
    engine = parent_evaluator()
    run = candidate_run_path(run, root)
    engine.validation_rows(records)
    summary = read_json(safe_file(run, 'summary.json'))
    from ours_stateful import StatefulOursConfig, build_execution_plan
    config = StatefulOursConfig(**protocol['stateful_sd35'])
    if config.latent_lr != LEARNING_RATE:
        raise ValueError('protocol latent learning rate differs from this child')
    objective = require_current_objective(summary)
    if (objective != 'U1' or summary.get('candidate_id') != IDENTITY + '-ours-det-only-' + objective or
            summary.get('task_variant') != 'det-only' or summary.get('status') != 'train_optimization_complete' or
            type(summary.get('executed_updates')) is not int or summary['executed_updates'] != 100 or
            type(summary.get('seed')) is not int or summary['seed'] != 20260919 or
            summary.get('training_models') != ['fasterrcnn', 'ssd300'] or summary.get('test_data_opened') is not False or
            summary.get('validation_data_opened') is not False or summary.get('formal_test_result') is not False or
            summary.get('resume_enabled') is not False):
        raise ValueError('only the completed v29 det-only U1 final100 run is eligible')
    if (summary.get('identity') != IDENTITY or summary.get('latent_lr') != LEARNING_RATE or
            summary.get('learning_rate_label') != VARIANT_LABEL):
        raise ValueError('training child learning-rate identity differs')
    if (canonical_json(summary.get('config')) != canonical_json(config.to_dict()) or
            canonical_json(summary.get('road_eot')) != canonical_json(protocol['road_eot'])):
        raise ValueError('completed-run configuration differs from frozen100 protocol')
    budget = protocol['budget_per_run']
    for field in ('planned', 'attempted', 'completed'):
        if canonical_json(summary.get('budget', {}).get(field)) != canonical_json(budget):
            raise ValueError('actual callback budget differs: ' + field)
    if canonical_json(summary.get('query_accounting')) != canonical_json(budget):
        raise ValueError('actual completed query accounting differs')
    bound_file(run, 'runtime_provenance.json', summary.get('runtime_provenance_sha256'))
    paths = {'training_manifest_sha256': training_manifest, 'model_manifest_sha256': model_manifest,
             'protocol_sha256': Path(root) / 'protocol/training_protocol.json',
             'schedule_sha256': Path(root) / 'protocol/train_schedule.json'}
    for key, path in paths.items():
        if summary.get(key) != sha(path):
            raise ValueError('training input identity differs: ' + key)
    schedule = read_json(bound_file(run, 'schedule.json', summary['schedule_sha256']))
    expected = [{'sample_id': r.sample_id, 'dataset': r.dataset, 'split': r.split}
                for r in engine.seeded_round_robin(records, 100, 20260919)]
    if schedule != expected or schedule != read_json(paths['schedule_sha256']):
        raise ValueError('frozen train schedule differs')
    trace_path = bound_file(run, 'attack_trace.jsonl', summary.get('attack_trace_sha256'))
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(trace) != 100:
        raise ValueError('exactly 100 completed trace entries are required')
    plan = build_execution_plan(config, engine.seeded_round_robin(records, 100, 20260919))
    if __package__:
        from .state_evidence import fingerprint
    else:
        from state_evidence import fingerprint
    for step, (row, expected_row, planned) in enumerate(zip(trace, schedule, plan), 1):
        require_current_objective(row,objective)
        if (type(row.get('global_step')) is not int or row['global_step'] != step or
                any(row.get(k) != v for k, v in expected_row.items()) or expected_row['split'] != 'train'):
            raise ValueError('trace is not the original complete train-only schedule')
        expected_fields = dict(asdict(planned), update_variant=objective, task_variant='det-only',
                               candidate_id=summary['candidate_id'],
                               normalizer_fingerprint=fingerprint(summary['normalizers']),
                               training_models=['fasterrcnn','ssd300'], model_count=2,
                               eot_samples_per_update=2, victim_forwards_minimum_this_update=8,
                               latent_backward_queries_this_update=2, latent_lr=config.latent_lr,
                               learning_rate_label=VARIANT_LABEL,
                               calibration_sha256=summary.get('calibration_sha256'))
        if any(canonical_json(row.get(k)) != canonical_json(v) for k, v in expected_fields.items()):
            raise ValueError('trace objective/configuration/calibration differs')
        for field in ('losses_pre','losses_post'):
            validate_loss_observations(row.get(field),objective)
    calibration = validate_calibration(run, summary, protocol, records)
    state_review = validate_states(run, summary, trace)
    bindings, checkpoints = summary.get('patch_bindings'), summary.get('checkpoint_bindings')
    engine.checked_bindings(run, bindings, ('zero_step_patch.pt', 'final_patch.pt'))
    engine.checked_bindings(run, checkpoints, engine.CHECKPOINTS, exact=True)
    for name, step in engine.CHECKPOINTS.items():
        bound_file(run, name, checkpoints[name])
        payload = torch.load(run / name, map_location='cpu', weights_only=True)
        if type(payload.get('metadata', {}).get('updates')) is not int or payload['metadata']['updates'] != step:
            raise ValueError('checkpoint update identity differs')
        engine.load_patch(run / name)
    zero, final = engine.load_patch(run / 'zero_step_patch.pt'), engine.load_patch(run / 'final_patch.pt')
    if (calibration.get('initial_crop_shape') != list(zero.shape) or
            calibration.get('initial_crop_dtype') != str(zero.dtype) or
            calibration.get('initial_crop_tensor_sha256') != hashlib.sha256(zero.contiguous().numpy().tobytes()).hexdigest()):
        raise ValueError('calibration did not bind the common unoptimized zero crop')
    for name, digest in bindings.items():
        bound_file(run, name, digest)
    if not torch.equal(final, engine.load_patch(run / 'checkpoint_0100.pt')):
        raise ValueError('final differs from fixed checkpoint100')
    random = engine.common_random_patch()
    if 'random_control_patch.pt' in bindings and not torch.equal(random, engine.load_patch(run / 'random_control_patch.pt')):
        raise ValueError('random control differs')
    return {'method': 'ours_det-only_' + objective, 'summary': summary, 'run': run,
            'patches': {'zero': zero, 'random': random, 'final': final},
            'summary_sha256': sha(run / 'summary.json'), 'trace_sha256': sha(trace_path),
            'patch_bindings': bindings, 'checkpoint_bindings': checkpoints,
            'calibration': calibration, 'state_review': state_review}


def _box_prediction(adapter, image):
    # The full original detector forward runs unchanged; discard Mask R-CNN's
    # unused mask tensor before retaining/serializing any output.
    prediction = adapter.predict(image)[0]
    return {key: prediction[key] for key in ('boxes', 'labels', 'scores')}


def evaluate(args, records, checked, device):
    """Four frozen detectors; training CORE is never changed or used for dispatch."""
    import torch
    engine = parent_evaluator()
    if tuple(engine.CORE) != CORE_MODELS or tuple(EVALUATION_MODELS) != (*CORE_MODELS, 'maskrcnn', 'fcos'):
        raise ValueError('source and evaluation detector roles differ')
    selected = engine.validation_rows(records)
    car, mask = engine.load_car(args.car, args.car_mask, device)
    initial = engine.build_context(engine.calibration_records(records)[0], car, mask, device)
    expected_zero = engine.canonicalize_rgb(engine.extract_road_patch(initial.background, initial.road_quad))
    if not torch.equal(checked['patches']['zero'].to(device), expected_zero):
        raise ValueError('zero patch differs from common fixed train initialization')
    del initial, expected_zero
    random_path = args.output / 'random_control_patch.pt'
    engine.save_patch(random_path, checked['patches']['random'], {'state':'unoptimized_random_control','seed':20260919})
    patches = {key: value.to(device) for key, value in checked['patches'].items()}
    rows, geometry, evidence, summaries = [], [], [], []
    prediction_counts = dict.fromkeys(EVALUATION_MODELS, 0)
    first = {domain: next(r.sample_id for r in selected if r.dataset == domain) for domain in ('kitti','nuscenes')}
    for model_index, model in enumerate(EVALUATION_MODELS):
        adapter = engine.build_offline_detector(model, device, args.model_manifest, args.weights_root)
        by_state = {state: [] for state in engine.STATES}
        try:
            with torch.no_grad():
                for record in selected:
                    context = engine.build_context(record, car, mask, device)
                    clean = _box_prediction(adapter, context.benign)
                    prediction_counts[model] += 1
                    visual = record.sample_id == first[record.dataset]
                    prefix = model + '_' + record.dataset
                    raw_predictions = {'sample_id':record.sample_id,'split':'validation','model':model,
                                       'target_box':context.object_box,'display_threshold_only':.3,
                                       'clean':engine.cpu_predictions(clean)} if visual else None
                    raw_inputs = {'sample_id':record.sample_id,'split':'validation','target_box':context.object_box,
                                  'clean':context.benign.detach().cpu(),'object_mask':context.object_mask.detach().cpu(),
                                  'road_quad':torch.as_tensor(context.road_quad).cpu()} if visual and model_index == 0 else None
                    if visual:
                        engine.save_preview(args.output/(prefix+'_clean.png'),context.benign,clean,context.object_box,
                                            model+' | '+record.sample_id+' | clean')
                    for state in engine.STATES:
                        render = engine.RoadRGBView(patches[state]).render(context)
                        invariant = engine.invariant_record(context,render,record,model,state)
                        geometry.append(invariant)
                        evidence.append(dict({k:invariant[k] for k in ('sample_id','dataset','split','model','state')},
                                             clean_input_sha256=typed_tensor_sha(context.benign),
                                             rendered_input_sha256=typed_tensor_sha(render.image),
                                             support_tensor_sha256=typed_tensor_sha(render.support)))
                        prediction = _box_prediction(adapter,render.image)
                        prediction_counts[model] += 1
                        row = {'method':checked['method'],'model':model,'sample_id':record.sample_id,
                               'dataset':record.dataset,'split':'validation','state':state,
                               'clean_candidates':engine.detection_candidates(clean,context.object_box,engine.VEHICLE_CLASS_IDS),
                               'adversarial_candidates':engine.detection_candidates(prediction,context.object_box,engine.VEHICLE_CLASS_IDS),
                               'clean_car_candidates':engine.detection_candidates(clean,context.object_box,engine.CAR_CLASS_IDS),
                               'adversarial_car_candidates':engine.detection_candidates(prediction,context.object_box,engine.CAR_CLASS_IDS)}
                        rows.append(row); by_state[state].append(row)
                        if visual:
                            raw_predictions[state]=engine.cpu_predictions(prediction)
                            engine.save_preview(args.output/(prefix+'_'+state+'.png'),render.image,prediction,context.object_box,
                                                model+' | '+record.sample_id+' | '+state)
                        if raw_inputs is not None:
                            raw_inputs[state]=render.image.detach().cpu()
                            raw_inputs[state+'_support']=render.support.detach().cpu()
                    if visual:
                        torch.save(raw_predictions,args.output/(prefix+'_predictions.pt'))
                    if raw_inputs is not None:
                        torch.save(raw_inputs,args.output/(record.dataset+'_inputs.pt'))
                for state in engine.STATES:
                    for domain in ('kitti','nuscenes','combined'):
                        subset=[r for r in by_state[state] if domain=='combined' or r['dataset']==domain]
                        value=engine.detection_summary(subset,1000,20260919,allow_zero_reference=True)
                        summaries.append(dict(value,model=model,state=state,dataset=domain))
        finally:
            del adapter
            if device.type=='cuda': torch.cuda.empty_cache()
    if len(rows)!=480 or len(geometry)!=480 or len(evidence)!=480 or prediction_counts!=dict.fromkeys(EVALUATION_MODELS,160):
        raise RuntimeError('four-detector evaluation coverage or budget differs')
    write_json(args.output/'per_sample.json',rows)
    write_json(args.output/'geometry_audit.json',{'status':'pass','unique_scenes':40,'checked_model_scene_states':480,'checks':geometry})
    write_json(args.output/'rendered_input_hashes.json',{
        'schema':'v29-passive-validation-input-sha-v1','rows':evidence,
        'hash_rule':'SHA256(canonical JSON dtype/shape followed by contiguous original dtype bytes)',
        'full_raw_input_scope':'first selected validation scene per dataset only; 2 input tensor files',
        'all_480_rendered_inputs_hashed':True,'additional_model_forwards':0})
    if __package__:
        from .compare_results import zero_to_final
    else:
        from compare_results import zero_to_final
    zero_results=zero_to_final(rows)
    result={'status':'development_validation_complete','candidate_id':engine.CANDIDATE_ID,
            'method':checked['method'],'updates':100,'records':len(rows),'summaries':summaries,
            'zero_to_final':zero_results,'summary_rows':len(summaries)+len(zero_results),
            'evaluation_models':list(EVALUATION_MODELS),'transfer_models_used':True,
            'evaluation_model_roles':{m:('source' if m in CORE_MODELS else 'transfer') for m in EVALUATION_MODELS},
            'evaluation_model_scene_forwards':prediction_counts,'evaluation_forward_total':sum(prediction_counts.values()),
            'test_data_opened':False,'formal_test_allowed':False,
            'metric':'tnr_det=(target_vehicle_AP50_clean-target_vehicle_AP50_adv)/target_vehicle_AP50_clean',
            'control_tensor_sha256':{k:engine._tensor_digest(v) for k,v in checked['patches'].items()},
            'random_control_patch_sha256':sha(random_path),'run_summary_sha256':checked['summary_sha256'],
            'run_trace_sha256':checked['trace_sha256'],'run_patch_bindings':checked['patch_bindings'],
            'run_checkpoint_bindings':checked['checkpoint_bindings'],
            'geometry_audit_sha256':sha(args.output/'geometry_audit.json'),
            'raw_prediction_fields':['boxes','labels','scores'],
            'note':'Fixed final100; no selection/filtering/training. Source and transfer models evaluated in this child.'}
    parent_id = result['candidate_id']
    summary = checked['summary']
    result.update(identity=IDENTITY, latent_lr=LEARNING_RATE, learning_rate_label=VARIANT_LABEL,
                  candidate_id=summary['candidate_id'], update_variant=summary['update_variant'],
                  training_objective_description=OBJECTIVE_DESCRIPTIONS[summary['update_variant']],
                  training_objective_models=list(CORE_MODELS),
                  training_target_classes=[3,6,8],
                  mathematical_objective=MATHEMATICAL_OBJECTIVE,
                  evaluation_target_classes=[3,6,8], evaluation_car_auxiliary_classes=[3],
                  task_variant='det-only', parent_metric_engine_candidate_id=parent_id,
                  parent_metric_engine_source_sha256=sha(Path(engine.__file__)),
                  protocol_sha256=summary['protocol_sha256'], seed=20260919,
                  calibration_sha256=summary['calibration_sha256'], normalizers=summary['normalizers'],
                  state_evidence_manifest_sha256=summary['state_evidence_manifest_sha256'],
                  training_runtime_provenance_sha256=summary['runtime_provenance_sha256'],
                  full_training_state_review=checked['state_review'],
                  training_manifest_sha256=summary['training_manifest_sha256'],
                  model_manifest_sha256=summary['model_manifest_sha256'],
                  run_output=str(checked['run'].relative_to(ROOT.resolve())),
                  source_images_full_raw_saved=2, all_rendered_input_hash_rows=480,
                  automatic_promotion=False, checkpoint_selection='fixed_final100')
    return result


def artifact_bindings(output):
    return {str(p.relative_to(output)): sha(p) for p in sorted(Path(output).rglob('*'))
            if p.is_file() and p.name not in ('summary.json', 'failure.json')}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--run-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-hostname')
    parser.add_argument('--package-manifest-sha256')
    parser.add_argument('--weights-root', type=Path)
    parser.add_argument('--source-prefix', type=Path)
    parser.add_argument('--source-root', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    verified, protocol = verify_inputs(require_sealed=True), load_protocol()
    if not args.run:
        print(json.dumps({'status': 'sealed_preflight_pass', 'inputs': verified,
                         'models_loaded': False, 'source_images_opened': False,
                         'evaluation': protocol['evaluation']}, ensure_ascii=False, indent=2))
        return
    if not args.expected_hostname or socket.gethostname() != args.expected_hostname:
        raise ValueError('explicit live hostname must match')
    if args.package_manifest_sha256 != verified['manifest_sha256']:
        raise ValueError('explicit sealed package SHA must match')
    if args.output is None or args.run_output is None:
        raise ValueError('run and new evaluation output required')
    if (args.source_prefix is None) != (args.source_root is None):
        raise ValueError('source mapping requires prefix and root')
    args.output, args.run_output = candidate_run_path(args.output), candidate_run_path(args.run_output)
    if args.output == args.run_output or args.run_output in args.output.parents or args.output in args.run_output.parents:
        raise ValueError('evaluation output must be separate from the completed run')
    if args.output.exists():
        raise FileExistsError('no overwrite/resume of evaluation output')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        import torch
        import torchvision
        from importlib.metadata import version
        if (torch.__version__.split('+')[0] != protocol['required_torch_base_version'] or
                torchvision.__version__.split('+')[0] != protocol['required_torchvision_base_version'] or
                version('diffusers') != protocol['required_diffusers_base_version']):
            raise ValueError('runtime library version differs')
        engine = parent_evaluator()
        args.training_manifest = PARENT / '01_公共数据集/runtime_v1/train_validation_manifest.csv'
        args.model_manifest = PARENT / '02_模型/model_manifest.json'
        args.car, args.car_mask = PARENT / '01_公共数据集/car_00.jpg', PARENT / '01_公共数据集/car_00_mask.jpg'
        records = engine.training_records(args.training_manifest, args.source_prefix, args.source_root)
        checked = validate_completed_run(args.run_output, records, args.training_manifest, args.model_manifest)
        if checked['summary'].get('package_manifest_sha256') != verified['manifest_sha256']:
            raise ValueError('completed training does not bind this sealed package')
        checks = engine.preflight(records, EVALUATION_MODELS, args.model_manifest, args.car, args.car_mask, args.weights_root)
        write_json(args.output / 'input_preflight.json', checks)
        write_json(args.output / 'invocation.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        if not checks['inputs_ready'] or not torch.cuda.is_available():
            raise RuntimeError('original offline assets and real CUDA are required; no fallback')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        write_json(args.output / 'runtime.json', {
            'candidate_id': checked['summary']['candidate_id'], 'hostname': socket.gethostname(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'diffusers': version('diffusers'),
            'cuda_version': torch.version.cuda, 'device_name': torch.cuda.get_device_name(device),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'torch_float32_matmul_precision': torch.get_float32_matmul_precision(),
            'training_manifest_sha256': sha(args.training_manifest),
            'model_manifest_sha256': sha(args.model_manifest),
            'package_manifest_sha256': verified['manifest_sha256'],
            'parent_metric_engine_source_sha256': sha(Path(engine.__file__))})
        result = evaluate(args, records, checked, device)
        torch.cuda.synchronize(device)
        result.update(seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                      package_manifest_sha256=verified['manifest_sha256'], hostname=socket.gethostname(),
                      artifact_bindings=artifact_bindings(args.output))
        write_json(args.output / 'summary.json', result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except BaseException as error:
        write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__,
                   'error': str(error), 'test_data_opened': False, 'formal_test_allowed': False})
        raise


if __name__ == '__main__':
    main()
