#!/usr/bin/env python3
"""CPU-only validated U1 export for one frozen v29 learning-rate child."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

if __package__:
    from .inputs import ROOT, IDENTITY, candidate_run_path, load_protocol, parent_evaluator, sha, verify_inputs, LEARNING_RATE, VARIANT_LABEL, EVALUATION_MODELS
    from .evaluate_validation import (artifact_bindings, bound_file, read_json, safe_file, write_json,
                                      require_current_objective, OBJECTIVE_DESCRIPTIONS, CORE_MODELS, validate_completed_run)
else:
    from inputs import ROOT, IDENTITY, candidate_run_path, load_protocol, parent_evaluator, sha, verify_inputs, LEARNING_RATE, VARIANT_LABEL, EVALUATION_MODELS
    from evaluate_validation import (artifact_bindings, bound_file, read_json, safe_file, write_json,
                                     require_current_objective, OBJECTIVE_DESCRIPTIONS, CORE_MODELS, validate_completed_run)

MODELS = EVALUATION_MODELS
DOMAINS = ('kitti', 'nuscenes', 'combined')
STATES = ('zero', 'random', 'final')


def row_key(row):
    return row['model'], row['dataset'], row['sample_id'], row['state']


def validate_rows(rows):
    if len(rows) != 480 or len({row_key(row) for row in rows}) != 480:
        raise ValueError('exactly 480 unique validation/model/state rows required')
    ids = {}
    for model in MODELS:
        for state in STATES:
            subset = [r for r in rows if r['model'] == model and r['state'] == state]
            if Counter(r['dataset'] for r in subset) != {'kitti': 20, 'nuscenes': 20}:
                raise ValueError('each model/state requires the same 20+20 validation scenes')
            identities = {(r['dataset'], r['sample_id']) for r in subset}
            if len(identities) != 40 or (ids and identities != ids):
                raise ValueError('validation identities differ between model/states')
            ids = identities
    for row in rows:
        if row.get('split') != 'validation' or row['model'] not in MODELS or row['state'] not in STATES:
            raise ValueError('test/train or unknown evaluation row')
        for field in ('clean_candidates', 'adversarial_candidates', 'clean_car_candidates', 'adversarial_car_candidates'):
            candidates = row.get(field)
            if not isinstance(candidates, list):
                raise ValueError('missing full metric candidates')
            for item in candidates:
                if (set(item) != {'score', 'iou', 'label'} or type(item['label']) is not int or
                        item['label'] not in ((3,) if '_car_' in field else (3, 6, 8)) or
                        any(type(item[key]) not in (int, float) or not math.isfinite(item[key]) for key in ('score', 'iou')) or
                        not 0 <= item['score'] <= 1 or not 0 < item['iou'] <= 1):
                    raise ValueError('invalid original target-vehicle candidate')
    return {row_key(row): row for row in rows}


def summarize_rows(rows, *, bootstrap_samples=1000):
    engine, output = parent_evaluator(), []
    for model in MODELS:
        for state in STATES:
            for domain in DOMAINS:
                subset = [r for r in rows if r['model'] == model and r['state'] == state
                          and (domain == 'combined' or r['dataset'] == domain)]
                value = engine.detection_summary(subset, bootstrap_samples, 20260919, allow_zero_reference=True)
                value.update(model=model, state=state, dataset=domain)
                output.append(value)
    return output


def zero_to_final(rows, *, bootstrap_samples=1000):
    indexed = {row_key(row): row for row in rows}
    derived = []
    for row in rows:
        if row['state'] != 'final':
            continue
        zero = indexed[(row['model'], row['dataset'], row['sample_id'], 'zero')]
        derived.append(dict(row, clean_candidates=zero['adversarial_candidates'],
                            clean_car_candidates=zero['adversarial_car_candidates']))
    output, engine = [], parent_evaluator()
    for model in MODELS:
        for domain in DOMAINS:
            subset = [r for r in derived if r['model'] == model and (domain == 'combined' or r['dataset'] == domain)]
            value = engine.detection_summary(subset, bootstrap_samples, 20260919, allow_zero_reference=True)
            value.update(model=model, dataset=domain, reference_state='zero', evaluated_state='final',
                         tnr_undefined_reason=None if value['tnr_defined'] else 'reference_AP50_is_zero')
            output.append(value)
    return output


def load_evaluation(path, objective='U1', *, root=ROOT, bootstrap_samples=1000):
    if objective != 'U1':
        raise ValueError('only current U1 evaluations are eligible')
    path = candidate_run_path(path, root)
    summary = read_json(safe_file(path, 'summary.json'))
    require_current_objective(summary,objective)
    if (summary.get('candidate_id') != IDENTITY + '-ours-det-only-' + objective or
            summary.get('update_variant') != objective or summary.get('task_variant') != 'det-only' or
            summary.get('status') != 'development_validation_complete' or
            summary.get('test_data_opened') is not False or summary.get('formal_test_allowed') is not False or
            summary.get('checkpoint_selection') != 'fixed_final100' or
            summary.get('evaluation_models') != list(MODELS) or
            type(summary.get('evaluation_forward_total')) is not int or summary.get('evaluation_forward_total') != 640 or
            type(summary.get('records')) is not int or summary.get('records') != 480 or
            type(summary.get('updates')) is not int or summary.get('updates') != 100 or
            type(summary.get('seed')) is not int or summary.get('seed') != 20260919 or
            summary.get('evaluation_model_scene_forwards') != dict.fromkeys(MODELS, 160)):
        raise ValueError('comparison requires completed v29 final100 validation with original budget')
    if (summary.get('training_objective_models') != list(CORE_MODELS) or
            summary.get('training_objective_description') != OBJECTIVE_DESCRIPTIONS[objective] or
            summary.get('training_target_classes') != [3,6,8] or
            summary.get('evaluation_target_classes') != [3,6,8] or
            summary.get('evaluation_car_auxiliary_classes') != [3]):
        raise ValueError('both-detector training and original evaluation objective descriptions differ')
    if (summary.get('identity') != IDENTITY or summary.get('latent_lr') != LEARNING_RATE or
            summary.get('learning_rate_label') != VARIANT_LABEL or summary.get('summary_rows') != 48 or
            summary.get('transfer_models_used') is not True or
            summary.get('evaluation_model_roles') != {m:('source' if m in CORE_MODELS else 'transfer') for m in MODELS}):
        raise ValueError('child learning-rate or model-role identity differs')
    if summary.get('protocol_sha256') != sha(Path(root) / 'protocol/training_protocol.json'):
        raise ValueError('evaluation protocol differs')
    engine = parent_evaluator()
    if (summary.get('parent_metric_engine_candidate_id') != engine.CANDIDATE_ID or
            summary.get('parent_metric_engine_source_sha256') != sha(Path(engine.__file__))):
        raise ValueError('evaluation metric engine identity differs')
    run_name = summary.get('run_output')
    if not isinstance(run_name, str) or Path(run_name).is_absolute() or '..' in Path(run_name).parts:
        raise ValueError('invalid source training run binding')
    source_run = candidate_run_path(Path(root) / run_name, root)
    training = read_json(bound_file(source_run, 'summary.json', summary.get('run_summary_sha256')))
    require_current_objective(training,objective)
    for field in ('identity', 'latent_lr', 'learning_rate_label', 'candidate_id', 'update_variant', 'task_variant', 'protocol_sha256', 'calibration_sha256',
                  'normalizers', 'state_evidence_manifest_sha256', 'training_manifest_sha256', 'model_manifest_sha256'):
        if summary.get(field) != training.get(field):
            raise ValueError('evaluation/training provenance differs: ' + field)
    for name, field in (('attack_trace.jsonl', 'run_trace_sha256'), ('calibration.json', 'calibration_sha256'),
                        ('state_evidence/MANIFEST.json', 'state_evidence_manifest_sha256'),
                        ('runtime_provenance.json', 'training_runtime_provenance_sha256')):
        bound_file(source_run, name, summary.get(field))
    if summary.get('training_runtime_provenance_sha256') != training.get('runtime_provenance_sha256'):
        raise ValueError('training runtime provenance binding differs')
    bindings = summary.get('artifact_bindings', {})
    required = {'per_sample.json', 'geometry_audit.json', 'rendered_input_hashes.json', 'random_control_patch.pt'}
    if not isinstance(bindings, dict) or not required.issubset(bindings):
        raise ValueError('missing evaluation artifact bindings')
    for name, digest in bindings.items():
        bound_file(path, name, digest)
    if set(bindings) != {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'summary.json'}:
        raise ValueError('evaluation artifact file set differs')
    rows = read_json(path / 'per_sample.json')
    indexed = validate_rows(rows)
    # Metadata only: original manifest, never source image bytes or model weights.
    original_records = engine.training_records(Path(engine.ROOT) / '01_公共数据集/runtime_v1/train_validation_manifest.csv')
    fixed_ids = {(r.dataset, r.sample_id) for r in engine.validation_rows(original_records)}
    if {(r['dataset'], r['sample_id']) for r in rows} != fixed_ids:
        raise ValueError('validation sample identities differ from frozen manifest')
    training_checked = validate_completed_run(source_run, original_records,
        Path(engine.ROOT)/'01_公共数据集/runtime_v1/train_validation_manifest.csv',
        Path(engine.ROOT)/'02_模型/model_manifest.json', root=root, protocol=load_protocol())
    if (summary.get('full_training_state_review') != training_checked['state_review'] or
            summary.get('package_manifest_sha256') != training.get('package_manifest_sha256')):
        raise ValueError('full training-state review or package provenance differs')
    recomputed = summarize_rows(rows, bootstrap_samples=bootstrap_samples)
    if recomputed != summary.get('summaries'):
        raise ValueError('saved summary differs from original metric recomputation')
    zero_summary = zero_to_final(rows, bootstrap_samples=bootstrap_samples)
    if zero_summary != summary.get('zero_to_final'):
        raise ValueError('saved zero-to-final differs from original metric recomputation')
    geometry = read_json(path / 'geometry_audit.json')
    required_checks = {'support_nonempty', 'protected_support_disjoint',
                       'protected_vehicle_pixels_exact', 'outside_support_pixels_exact', 'finite_input'}
    if (geometry.get('status') != 'pass' or geometry.get('unique_scenes') != 40 or
            geometry.get('checked_model_scene_states') != 480 or len(geometry.get('checks', [])) != 480 or
            {row_key(r) for r in geometry['checks']} != set(indexed) or
            any(set(r.get('checks', {})) != required_checks or
                any(value is not True for value in r['checks'].values()) or
                r.get('support_pixels', 0) <= 0 or r.get('protected_pixels', 0) <= 0
                for r in geometry['checks'])):
        raise ValueError('geometry evidence is incomplete or failed')
    hashes = read_json(path / 'rendered_input_hashes.json')
    if (hashes.get('schema') != 'v29-passive-validation-input-sha-v1' or
            hashes.get('all_480_rendered_inputs_hashed') is not True or
            hashes.get('additional_model_forwards') != 0 or len(hashes.get('rows', [])) != 480 or
            {row_key(r) for r in hashes['rows']} != set(indexed)):
        raise ValueError('full passive input hash coverage differs')
    for row in hashes['rows']:
        for key in ('clean_input_sha256', 'rendered_input_sha256', 'support_tensor_sha256'):
            if not isinstance(row.get(key), str) or len(row[key]) != 64 or any(c not in '0123456789abcdef' for c in row[key]):
                raise ValueError('invalid typed input SHA')
    paired = {}
    clean = {}
    for row in hashes['rows']:
        key = (row['dataset'],row['sample_id'],row['state'])
        values = tuple(row[k] for k in ('clean_input_sha256','rendered_input_sha256','support_tensor_sha256'))
        if key in paired and paired[key] != values:
            raise ValueError('four detectors did not receive identical scene/state inputs')
        paired[key] = values
        scene_key = key[:2]
        stable = (values[0],values[2])
        if scene_key in clean and clean[scene_key] != stable:
            raise ValueError('clean scene or road support changed across states')
        clean[scene_key] = stable
    return {'path': path, 'summary': summary, 'rows': rows, 'index': indexed,
            'hashes': {row_key(r): r for r in hashes['rows']}, 'summaries': recomputed,
            'zero_to_final': zero_summary, 'training_state_review':training_checked['state_review']}



def export_evaluation(evaluation, verified):
    summary = evaluation['summary']
    if summary.get('package_manifest_sha256') != verified['manifest_sha256']:
        raise ValueError('evaluation does not bind this sealed child package')
    path = evaluation['path']
    return {'identity':IDENTITY,'status':'validated_evaluation','update_variant':'U1',
            'mathematical_objective':'C0','candidate_id':summary['candidate_id'],
            'latent_lr':LEARNING_RATE,'learning_rate_label':VARIANT_LABEL,
            'evaluation_path':str(path),'source_summary_sha256':sha(path/'summary.json'),
            'package_manifest_sha256':verified['manifest_sha256'],
            'protocol_sha256':summary['protocol_sha256'],'summary_rows':48,
            'evaluation_models':list(MODELS),'training_models':list(CORE_MODELS),
            'training_manifest_sha256':summary['training_manifest_sha256'],
            'model_manifest_sha256':summary['model_manifest_sha256'],
            'parent_metric_engine_source_sha256':summary['parent_metric_engine_source_sha256'],
            'control_tensor_sha256':summary['control_tensor_sha256'],'seed':summary['seed'],
            'summaries':evaluation['summaries'],'zero_to_final':evaluation['zero_to_final'],
            'per_sample_path':str(path/'per_sample.json'),'per_sample_sha256':sha(path/'per_sample.json'),
            'rendered_input_hashes_path':str(path/'rendered_input_hashes.json'),
            'rendered_input_hashes_sha256':sha(path/'rendered_input_hashes.json'),
            'training_state_review':evaluation['training_state_review'],
            'source_training_run':summary['run_output'],'training_summary_sha256':summary['run_summary_sha256'],
            'evaluation_forward_total':640,'records':480,'new_model_forwards':0,
            'models_loaded':False,'source_images_opened':False,'test_data_opened':False,
            'formal_test_allowed':False,'automatic_promotion':False,
            'checkpoint_selection':'fixed_final100','bootstrap_samples':1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    verified = verify_inputs(require_sealed=True)
    load_protocol()
    if not args.run:
        print(json.dumps({'status':'sealed_preflight_pass','inputs':verified,
                          'models_loaded':False,'source_images_opened':False,'mode':'single_U1_cpu_export'},ensure_ascii=False))
        return
    if args.evaluation is None or args.output is None:
        raise ValueError('--run requires an evaluation and a separate new output directory')
    source, output = candidate_run_path(args.evaluation), candidate_run_path(args.output)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('export output must be separate from evaluation evidence')
    if output.exists():
        raise FileExistsError('no export overwrite/resume')
    output.mkdir(parents=True)
    try:
        evaluation = load_evaluation(source)
        result = export_evaluation(evaluation, verified)
        write_json(output/'export.json',result)
        receipt = {k:v for k,v in result.items() if k not in ('summaries','zero_to_final')}
        receipt.update(export_sha256=sha(output/'export.json'),artifact_bindings=artifact_bindings(output))
        write_json(output/'summary.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
    except BaseException as error:
        write_json(output/'failure.json',{'status':'failed_retained','type':type(error).__name__,
                                          'error':str(error),'test_data_opened':False})
        raise


if __name__ == '__main__':
    main()
