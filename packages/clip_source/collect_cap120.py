#!/usr/bin/env python3
"""Collect one post-evaluation point after binding completed raw-audit receipts.

This reader rehashes saved evidence. It does not load models or tensor payloads,
recompute metrics, select a threshold, or start inference. Historical-baseline
comparisons are descriptive because methods have different budgets and constraints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import clip_evaluate as evaluation
import compare_sweep as comparison

INPUT_SCHEMA = 'r2-joint-clip120-collection-inputs-v1'
OUTPUT_SCHEMA = 'r2-joint-clip120-descriptive-comparison-v1'
DETECTORS = ('fasterrcnn', 'ssd300', 'maskrcnn', 'fcos')
DEPTH_MODELS = evaluation.DEPTH_MODELS
DEPTH_METRICS = ('abs_rel_clean_ref', 'rmse_clean_ref', 'rmse_log_clean_ref')
DETECTION_METHODS = ('thys', 'nap', 'dpatch', 'cap', 'tsea')
DEPTH_METHODS = ('saam_monodepth2', 'saam_dpt_large')
METRIC_SHA = '50bcf1aad80ae902d6e660017b1614d83c9a8a085b6a6cbe02a5402e31752c81'
TRAINING_SHA = '38ae14801d2e330236bd7b408f08ef29c5f93b4e802c6a8f2414a5a352ceb23f'
MODEL_SHA = 'd994315183884c678c35e1b4cdb03c0b459b74ba27fd16bc95e3c17717c0b5dc'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def plain_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink evidence path forbidden: ' + str(path))
    return path


def read(path):
    path = plain_path(path)
    return comparison._load_json(path.read_bytes())


def bound(root, name, expected_sha):
    evaluation.require_sha(expected_sha)
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe artifact path')
    path = plain_path(Path(root) / relative)
    if not path.is_file() or sha(path) != expected_sha:
        raise ValueError('bound artifact SHA mismatch: ' + str(path))
    return path


def require(value, expected, context):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(context + ' differs')


def check_baselines(document):
    baselines = comparison._baselines(document)
    expected = {('detection', m, 'tnr_det', method) for method in DETECTION_METHODS for m in DETECTORS}
    expected |= {('depth', m, metric, method) for method in DEPTH_METHODS for m in DEPTH_MODELS for metric in DEPTH_METRICS}
    source_rows = document.get('sources', [])
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError('baseline source evidence is required')
    sources = {row['id']: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError('duplicate baseline source id')
    for source in sources.values():
        evaluation.require_sha(source.get('sha256'))
        if not source.get('path') or type(source.get('bytes')) is not int or source['bytes'] <= 0:
            raise ValueError('invalid baseline source binding')
    observed = set()
    for item in baselines.values():
        key = (item['task'], item['model'], item['metric'], item['baseline_method'])
        if key in observed:
            raise ValueError('duplicate baseline scientific comparison')
        observed.add(key)
        percent = item['metric'] in ('tnr_det', 'abs_rel_clean_ref')
        require(item['unit'], 'percent' if percent else 'native', 'baseline unit')
        direction = 'larger_is_more_AP50_drop' if item['task'] == 'detection' else 'larger_is_more_prediction_deviation'
        require(item['direction'], direction, 'baseline direction')
        if item.get('source_id') not in sources or not item.get('source_field'):
            raise ValueError('baseline source id/field is required')
        evaluation.require_sha(item.get('remote_evaluation_summary_sha256'))
        item['source_evidence'] = dict(sources[item['source_id']], field=item['source_field'],
                                      remote_evaluation_summary_sha256=item['remote_evaluation_summary_sha256'])
    if observed != expected or len(baselines) != 44:
        raise ValueError('all 20 detection and 24 depth baseline comparisons are required')
    return baselines


def _audit(receipt, task, eval_sha, summary, training_sha, package_sha, source):
    require(receipt.get('status'), task + '_raw_audit_passed', task + ' audit status')
    require(receipt.get('exploratory_posthoc'), True, task + ' posthoc disclosure')
    require(receipt.get('selection_context'), evaluation.SELECTION_CONTEXT, task + ' selection context')
    require(receipt.get('evaluation_summary_sha256'), eval_sha, task + ' evaluation binding')
    require(receipt.get('candidate_id'), summary['candidate_id'], task + ' candidate')
    require(receipt.get('updates'), 500, task + ' audit updates')
    require(receipt.get('checkpoint_selection'), 'fixed_final500', task + ' checkpoint')
    require(receipt.get('test_data_opened'), False, task + ' test isolation')
    require(receipt.get('audit_source_sha256'), sha(source / ('clip_audit_' + task + '.py')), task + ' auditor source')
    counts = receipt.get('counts', {})
    require(counts.get('rows_checked'), 640 if task == 'depth' else 480, task + ' audited rows')
    require(counts.get('summaries_checked'), 48, task + ' audited summaries')
    require(counts.get('raw_tensor_checks'), 800 if task == 'depth' else 1920, task + ' raw tensor count')
    for key in ('artifact_sha_checks', 'scalar_checks'):
        if type(counts.get(key)) is not int or counts[key] <= 0:
            raise ValueError('missing positive audit count: ' + key)
    identity = receipt.get('identity_bindings', {})
    for key, value in {'run_summary_sha256': training_sha, 'model_manifest_sha256': MODEL_SHA,
                       'wrapper_sha256': sha(source / 'clip_evaluate.py'), 'method': summary['method'], 'updates': 500}.items():
        require(identity.get(key), value, task + ' audit identity.' + key)
    require(identity.get('clip_training_package', {}).get('manifest_sha256'), package_sha, task + ' audit package')
    if task == 'depth':
        require(receipt.get('tolerance'), 0, 'depth raw tolerance')
        require(receipt.get('depth_metric_source_sha256'), METRIC_SHA, 'depth metric source')
        require(receipt.get('no_model_loading_or_inference'), True, 'depth audit inference')
    else:
        require(receipt.get('training_summary_sha256'), training_sha, 'detector training binding')
        require(receipt.get('evaluation_wrapper_sha256'), sha(source / 'clip_evaluate.py'), 'detector wrapper')
        require(receipt.get('storage_helpers_sha256'), sha(source / 'clip_audit_depth.py'), 'detector storage helpers')
        require(receipt.get('metric_source_sha256'), METRIC_SHA, 'detector metric source')
        require(receipt.get('tolerated_iou_count'), 0, 'detector actual IoU tolerance')
        require(receipt.get('tolerated_iou'), [], 'detector actual IoU tolerance detail')
        require(receipt.get('no_models_loaded_or_called'), True, 'detector audit inference')


def _controls(root, summary):
    """Compare typed hashes from SHA-bound raw manifests, without loading tensors."""
    controls = {}
    scenes = {(r['dataset'], r['sample_id']) for r in summary['validation_scenes']}
    for task, filename, models in (('detection', 'raw_predictions/MANIFEST.json', DETECTORS),
                                   ('depth', 'depth_raw/MANIFEST.json', DEPTH_MODELS)):
        manifest = read(root / filename)
        require(manifest.get('model_forward_count'), 640, task + ' raw forwards')
        entries = manifest.get('entries', [])
        expected = {(m, d, sample) for m in models for d, sample in scenes}
        found = {}
        for row in entries:
            key = (row['model'], row['dataset'], row['sample_id'])
            if key in found:
                raise ValueError('duplicate raw manifest model-scene')
            require(summary['artifact_bindings'].get(row.get('path')), row.get('sha256'), task + ' raw artifact binding')
            evaluation.require_sha(row.get('sha256'))
            hashes = row.get('tensor_sha256', {})
            selected = {}
            for state in ('clean', 'zero', 'random'):
                value = hashes.get(state)
                if task == 'depth':
                    evaluation.require_sha(value)
                else:
                    if not isinstance(value, dict) or set(value) != {'boxes', 'labels', 'scores'}:
                        raise ValueError('detector control typed hashes missing')
                    for digest in value.values():
                        evaluation.require_sha(digest)
                selected[state] = value
            if task == 'depth':
                mask = row.get('object_mask', {})
                require(summary['artifact_bindings'].get(mask.get('path')), mask.get('sha256'), 'depth mask artifact binding')
                evaluation.require_sha(mask.get('sha256'))
                selected['object_mask'] = evaluation.require_sha(mask.get('tensor_sha256'))
                require(row.get('semantics'), evaluation.SEMANTICS[row['model']], 'native depth semantics')
            found[key] = selected
        if set(found) != expected or len(entries) != 160:
            raise ValueError(task + ' raw manifest coverage differs')
        controls[task] = found
    return controls


def _metrics(root, summary):
    detector = read(bound(root, 'detection_summary.json', summary['detection_summary_sha256']))
    depth = read(bound(root, 'depth_summary.json', summary['depth_summary_sha256']))
    require(detector.get('updates'), 500, 'detector summary update count')
    require(detector.get('method'), summary['method'], 'detector summary method')
    require(detector.get('run_summary_sha256'), summary['run_summary_sha256'], 'detector summary training binding')
    require(detector.get('summary_rows'), 48, 'detector summary count')
    require(depth.get('summary_rows'), 48, 'depth summary count')
    require(depth.get('records'), 640, 'depth metric rows')
    require(depth.get('raw_manifest_sha256'), sha(root / 'depth_raw/MANIFEST.json'), 'depth raw manifest binding')
    result, diagnostics = {}, []
    det_rows = detector.get('summaries', [])
    require(len(det_rows), 36, 'detector clean-reference summary count')
    for row in det_rows:
        if row.get('dataset') == 'combined' and row.get('state') == 'final':
            require(row.get('scene_count'), 40, 'detector combined scene count')
            require(row.get('tnr_defined'), True, 'detector defined TNR')
            key = ('detection', row['model'], 'tnr_det')
            if key in result:
                raise ValueError('duplicate detector final metric')
            result[key] = comparison._number(row['tnr_det'], 'TNR') * 100.
    dep_rows = depth.get('summaries', [])
    require(len(dep_rows), 48, 'depth summary row count')
    for row in dep_rows:
        if row.get('dataset') == 'combined' and row.get('reference_state') == 'clean' and row.get('evaluated_state') == 'final':
            require(row.get('scene_count'), 40, 'depth combined scene count')
            require(row.get('depth_representation'), evaluation.SEMANTICS[row['model']], 'metric native depth semantics')
            delta = comparison._number(row['macro']['delta_sigma']['mean'], 'delta_sigma')
            diagnostics.append({'model': row['model'], 'metric': 'delta_sigma', 'value': delta,
                'unit': 'native_log_ratio', 'depth_representation': row['depth_representation'],
                'positive_shift_interpretation': 'nearer' if evaluation.SEMANTICS[row['model']] == 'relative_inverse_depth' else 'farther',
                'dataset': 'combined', 'scene_count': 40, 'reference_state': 'clean', 'evaluated_state': 'final',
                'included_in_main_comparisons': False})
            for metric in DEPTH_METRICS:
                key = ('depth', row['model'], metric)
                if key in result:
                    raise ValueError('duplicate depth final metric')
                value = comparison._number(row['macro'][metric]['mean'], metric)
                result[key] = value * 100. if metric == 'abs_rel_clean_ref' else value
    expected = {('detection', m, 'tnr_det') for m in DETECTORS}
    expected |= {('depth', m, metric) for m in DEPTH_MODELS for metric in DEPTH_METRICS}
    if set(result) != expected:
        raise ValueError('missing/foreign model or metric in fixed final combined table')
    return result, diagnostics


def clipping_diagnostics(run, training):
    rows = [comparison._load_json(line) for line in (run / 'attack_trace.jsonl').read_bytes().splitlines()]
    if len(rows) != 500:
        raise ValueError('500 clipping diagnostic rows required')
    values = []
    threshold = training['gradient_clipping']['max_norm']
    for index, row in enumerate(rows, 1):
        require(row.get('global_step'), index, 'clip trace step')
        require(row.get('split'), 'train', 'clip trace split')
        value = row.get('clipping_diagnostics', {})
        if value.get('max_norm') != threshold or isinstance(value.get('max_norm'), bool):
            raise ValueError('trace clipping threshold differs')
        for field in ('norm_before', 'norm_after', 'realized_update_rms'):
            if comparison._number(value.get(field), field) < 0:
                raise ValueError('negative clipping diagnostic')
        if type(value.get('clipped')) is not bool:
            raise ValueError('clipped diagnostic must be boolean')
        require(value['clipped'], threshold is not None and value['norm_before'] > threshold, 'actual clipping trigger')
        values.append(value)
    result = {'steps': 500, 'triggered_steps': sum(v['clipped'] for v in values),
        'triggered_fraction': sum(v['clipped'] for v in values) / 500,
        'max_norm_before': max(v['norm_before'] for v in values),
        'max_norm_after': max(v['norm_after'] for v in values),
        'mean_realized_update_rms': sum(v['realized_update_rms'] for v in values) / 500}
    declared = training.get('clipping_summary', {})
    for key, value in result.items():
        require(declared.get(key), value, 'actual trace/summary clipping.' + key)
    return result


def check_state_layout(manifest):
    entries = manifest.get('entries', [])
    expected = [('initialization', 0, 0)]
    for outer in range(20):
        expected.append(('outer_start', 0, outer * 25))
        for step in range(outer * 25 + 1, outer * 25 + 26):
            expected.extend((phase, step, step if phase == 'post_update' else step - 1)
                            for phase in ('before_denoise', 'pre_backward', 'pre_update', 'post_update'))
        expected.append(('outer_end', 0, (outer + 1) * 25))
    observed = [(row.get('phase'), row.get('global_step'), row.get('updates_applied')) for row in entries]
    require(observed, expected, 'complete ordered 2041-state layout')
    require([row.get('index') for row in entries], list(range(1, 2042)), 'state sequence indices')
    recoveries = manifest.get('recovery_checkpoints', [])
    require(len(recoveries), 20, 'recovery checkpoint count')
    initial = manifest.get('initialization_state')
    if not isinstance(initial, dict):
        raise ValueError('initialization state storage missing')
    for outer, recovery in enumerate(recoveries, 1):
        require(recovery.get('completed_outer_round'), outer, 'recovery round')
        require(recovery.get('executed_updates'), 25 * outer, 'recovery update count')
    for row in entries:
        expected_stored = initial if row['phase'] == 'initialization' else None
        if row['phase'] == 'outer_end':
            recovery = recoveries[row['updates_applied'] // 25 - 1]
            expected_stored = {key: recovery[key] for key in ('path', 'sha256', 'bytes')}
        require(row.get('stored_state'), expected_stored, 'state storage pointer')
    return entries


def collect_group(entry, plan_sha, package_sha, source):
    paths = {key: plain_path(entry[key]) for key in ('run', 'evaluation', 'audit_depth', 'audit_detection')}
    if len(set(paths.values())) != 4:
        raise ValueError('training/evaluation/audits must use four separate directories')
    root, run = paths['evaluation'], paths['run']
    summary = read(root / 'summary.json')
    eval_sha = sha(root / 'summary.json')
    for key, value in {'schema': 'r2-joint-clip120-eight-model-development-evaluation-v1',
                       'status': 'development_validation_complete', 'scene_count': 40, 'updates': 500,
                       'training_budget_updates': 500, 'evaluation_forward_total': 1280,
                       'checkpoint_selection': 'fixed_final500', 'test_data_opened': False}.items():
        require(summary.get(key), value, 'evaluation.' + key)
    require(summary.get('exploratory_posthoc'), True, 'evaluation posthoc disclosure')
    require(summary.get('selection_context'), evaluation.SELECTION_CONTEXT, 'evaluation selection context')
    training_path = bound(run, 'summary.json', summary['run_summary_sha256'])
    training = read(training_path)
    method, models = evaluation.run_role(training, package_sha)
    require(summary.get('method'), method, 'evaluation training method')
    require(summary.get('training_models'), list(models), 'evaluation training models')
    require(summary.get('candidate_id'), training['candidate_id'], 'evaluation candidate')
    require(summary.get('gradient_clipping'), training['gradient_clipping'], 'evaluation clipping policy')
    require(training['gradient_clipping']['group_id'], entry['id'], 'input canonical group')
    require(training['gradient_clipping']['grid_sha256'], plan_sha, 'group frozen plan SHA')
    require(summary.get('clipping_comparison_gate_sha256'), training['clipping_comparison_gate_sha256'], 'comparison gate binding')
    state_manifest = read(bound(run, 'state_evidence/MANIFEST.json', training['state_evidence_manifest_sha256']))
    evaluation.validate_comparison_gate(run, training, state_manifest)
    require(state_manifest.get('schema'), 'r2-joint-clip500-lean-state-manifest-v1', 'state schema')
    require(state_manifest.get('status'), 'complete', 'state completion')
    require(state_manifest.get('executed_updates'), 500, 'state executed updates')
    require(state_manifest.get('candidate_id'), training['candidate_id'], 'state candidate')
    entries = check_state_layout(state_manifest)
    ledger_path = bound(run, 'state_evidence/ledger.jsonl', state_manifest['ledger_sha256'])
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    require(ledger, entries, 'state ledger')
    for row in entries:
        require(row.get('recording_preserved_rng'), True, 'state recorder RNG')
        evaluation.require_sha(row.get('state_fingerprint'))
        stored = row.get('stored_state')
        if stored is not None:
            path = bound(run, stored['path'], stored['sha256'])
            require(path.stat().st_size, stored['bytes'], 'stored state size')
    for field, filename in (('attack_trace_sha256', 'attack_trace.jsonl'), ('schedule_sha256', 'schedule.json'),
                             ('runtime_provenance_sha256', 'runtime_provenance.json')):
        bound(run, filename, training[field])
    for mapping in ('patch_bindings', 'checkpoint_bindings'):
        for filename, digest in training[mapping].items():
            bound(run, filename, digest)
    bindings = summary.get('source_bindings', {})
    for key, value in {'wrapper_sha256': sha(source / 'clip_evaluate.py'),
                       'clip_training_package_manifest_sha256': package_sha,
                       'model_manifest_sha256': MODEL_SHA, 'training_manifest_sha256': TRAINING_SHA,
                       'depth_evaluator_sha256': METRIC_SHA, 'r2_manifest_sha256': evaluation.R2_MANIFEST}.items():
        require(bindings.get(key), value, 'evaluation source.' + key)
    for key, value in (('model_manifest_sha256', MODEL_SHA), ('training_manifest_sha256', TRAINING_SHA)):
        require(training.get(key), value, 'training source.' + key)
    scenes = summary.get('validation_scenes', [])
    require(len(scenes), 40, 'validation scene count')
    if (Counter(row.get('dataset') for row in scenes) != {'kitti': 20, 'nuscenes': 20}
            or any(row.get('split') != 'validation' for row in scenes)
            or len({(r['dataset'], r['sample_id']) for r in scenes}) != 40):
        raise ValueError('validation identities/coverage differ')
    artifacts = summary.get('artifact_bindings', {})
    required = {'CLIP_PLAN.json', 'detection_summary.json', 'depth_summary.json',
                'raw_predictions/MANIFEST.json', 'depth_raw/MANIFEST.json', 'rendered_input_hashes.json'}
    if not required <= set(artifacts):
        raise ValueError('evaluation artifact bindings incomplete')
    for filename, digest in artifacts.items():
        bound(root, filename, digest)
    require(artifacts['CLIP_PLAN.json'], plan_sha, 'evaluation frozen plan SHA')
    for task in ('depth', 'detection'):
        receipt = read(paths['audit_' + task] / 'summary.json')
        _audit(receipt, task, eval_sha, summary, summary['run_summary_sha256'], package_sha, source)
    controls = _controls(root, summary)
    metrics, depth_diagnostics = _metrics(root, summary)
    actual_clipping = clipping_diagnostics(run, training)
    receipt = {'group_id': entry['id'], 'evaluation_summary_sha256': eval_sha,
               'training_summary_sha256': summary['run_summary_sha256'],
               'depth_audit_sha256': sha(paths['audit_depth'] / 'summary.json'),
               'detection_audit_sha256': sha(paths['audit_detection'] / 'summary.json'),
               'rehash_evaluation_artifact_count': len(artifacts), 'audit_verified': True,
               'actual_clipping': actual_clipping, 'depth_direction_diagnostics': depth_diagnostics,
               'paths': {key: str(path) for key, path in paths.items()}}
    common = {'scenes': scenes, 'initialization_state_fingerprint': entries[0]['state_fingerprint'],
              'schedule_sha256': training['schedule_sha256'], 'controls': controls}
    return metrics, receipt, common



def collect_files(plan_path, baselines_path, inputs_path):
    """One user-requested posthoc point; no grid interpolation or threshold claim."""
    from clip_evidence import validate_followup_plan
    plan, baselines_document, inputs = read(plan_path), read(baselines_path), read(inputs_path)
    validate_followup_plan(plan)
    plan_sha, baseline_sha = sha(plan_path), sha(baselines_path)
    baselines = check_baselines(baselines_document)
    require(inputs.get('schema'), INPUT_SCHEMA, 'collection input schema')
    require(inputs.get('grid_sha256'), plan_sha, 'collection followup plan SHA')
    require(inputs.get('baselines_sha256'), baseline_sha, 'collection baseline SHA')
    source = plain_path(inputs['training_source'])
    package_sha = inputs['training_package_manifest_sha256']
    evaluation.bind_long_package(source, package_sha)
    if sha(source / 'clip_evaluate.py') != sha(Path(__file__).with_name('clip_evaluate.py')):
        raise ValueError('collector evaluator implementation differs from sealed source')
    members = {row['path']: row for row in read(source / 'PACKAGE_MANIFEST.json')['files']}
    for name, digest in (('collect_cap120.py', sha(Path(__file__))), ('BASELINE_REFERENCES.json', baseline_sha)):
        if name not in members or members[name].get('sha256') != digest:
            raise ValueError('collector and baseline references must be sealed package members')
        bound(source, name, digest)
    entries = inputs.get('groups')
    if not isinstance(entries, list) or len(entries) != 1 or entries[0].get('id') != 'cap120':
        raise ValueError('exactly one explicit cap120 input group required')
    metrics, receipt, controls = collect_group(entries[0], plan_sha, package_sha, source)
    rows = []
    for comparison_id, baseline in baselines.items():
        value = metrics[(baseline['task'], baseline['model'], baseline['metric'])]
        delta = value - baseline['value']
        rows.append({'group_id': 'cap120', 'max_norm': 120.0, 'comparison_id': comparison_id,
                     'task': baseline['task'], 'model': baseline['model'], 'metric': baseline['metric'],
                     'unit': baseline['unit'], 'baseline_method': baseline['baseline_method'],
                     'direction': baseline['direction'], 'baseline_value': baseline['value'],
                     'ours_value': value, 'delta': delta, 'audit_verified': True,
                     'observed_relation': 'higher' if delta > 0 else ('lower' if delta < 0 else 'exact_sampled_tie')})
    if len(rows) != 44:
        raise ValueError('all 44 baseline comparisons required')
    return {'schema': OUTPUT_SCHEMA, 'status': 'single_posthoc_point_raw_audits_verified',
        'group_id': 'cap120', 'max_norm': 120.0, 'canonical_group_count': 1,
        'comparison_count': 44, 'verified_point_count': 44, 'rows': rows,
        'group_receipts': [receipt], 'metric_values': [
            {'task': key[0], 'model': key[1], 'metric': key[2], 'value': value,
             'unit': 'percent' if key[2] in ('tnr_det', 'abs_rel_clean_ref') else 'native'}
            for key, value in sorted(metrics.items())],
        'scene_count': 40, 'evaluation_forward_total': 1280,
        'updates_per_group': 500, 'checkpoint_selection': 'fixed_final500',
        'grid_sha256': plan_sha, 'baselines_sha256': baseline_sha,
        'training_package_manifest_sha256': package_sha,
        'input_manifest_sha256': sha(inputs_path), 'collector_source_sha256': sha(Path(__file__)),
        'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
        'preregistered_grid_point': False, 'critical_threshold_established': False,
        'automatic_promotion': False, 'test_data_opened': False,
        'no_model_loading_or_inference': True, 'raw_tensor_metrics_recomputed_in_collector': False,
        'within_run_controls_and_coverage_verified': True,
        'cross_run_controls_reverified_in_this_collector': False,
        'claims': 'single user-requested point after viewing six-group development outcomes; descriptive historical-baseline comparison, not an independent validation claim, equal-budget causal gain, or GT depth error',
        'metric_selection': 'combined40 clean-to-final; detection TNR and depth AbsRel multiplied by100; other depth metrics native',
        'baseline_source_evidence': {key: item['source_evidence'] for key, item in baselines.items()},
        'input_bindings': {key: {'path': str(plain_path(path)), 'sha256': sha(path)}
            for key, path in (('plan', plan_path), ('baselines', baselines_path), ('inputs', inputs_path))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'baselines', 'inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect_files(args.plan, args.baselines, args.inputs)
    digest = comparison.write_result(result, args.output)
    print(json.dumps({'status': result['status'], 'sha256': digest, 'groups': 1, 'points': len(result['rows'])}))


if __name__ == '__main__':
    main()
