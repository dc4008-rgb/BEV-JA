"""Synthetic receipt/artifact contract tests; no model or raw tensor inference."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from test_cap120_fixtures import environment_evidence

import collect_cap120 as collect
import clip_evaluate as evaluation
import compare_sweep as comparison

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return collect.sha(path)


def binary(path, content=b'synthetic fixture, not a tensor'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return collect.sha(path)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        gate = mock.patch('clip_evidence.validate_followup_plan', side_effect=lambda plan: plan)
        gate.start(); self.addCleanup(gate.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        names = ('clip_training.py', 'clip_evaluate.py', 'clip_audit_depth.py', 'clip_audit_detection.py',
                 'collect_cap120.py', 'BASELINE_REFERENCES.json')
        for name in names:
            shutil.copyfile(ROOT / name, self.source / name)
        self.package_sha = save(self.source / 'PACKAGE_MANIFEST.json', {'files': [
            {'path': name, 'bytes': (self.source / name).stat().st_size, 'sha256': collect.sha(self.source / name)}
            for name in names]})
        self.baselines = self.source / 'BASELINE_REFERENCES.json'
        self.plan = self.root / 'PLAN.json'
        groups = [{'id': 'cap120', 'max_norm': 120.0, 'executed_updates': 500}]
        self.runtime, self.execution, self.off, self.envgate = environment_evidence(self.package_sha)
        self.plan_sha = save(self.plan, {'schema': 'r2-joint-clip120-followup-plan-v1',
            'status': 'fixed_user_requested_post_evaluation_followup', 'groups': groups,
            'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
            'execution_off': self.execution, 'reference': {'summary_sha256': '2' * 64,
                'trace_sha256': '6' * 64, 'state_evidence_manifest_sha256': '3' * 64,
                'training_manifest_sha256': collect.TRAINING_SHA}})
        self.entries = [self.make_group(row) for row in groups]
        self.inputs = self.root / 'INPUTS.json'
        save(self.inputs, {'schema': collect.INPUT_SCHEMA, 'grid_sha256': self.plan_sha,
             'baselines_sha256': collect.sha(self.baselines), 'training_source': str(self.source),
             'training_package_manifest_sha256': self.package_sha, 'groups': self.entries})

    def tearDown(self):
        self.temp.cleanup()

    def make_group(self, group):
        base = self.root / group['id']
        paths = {key: base / key for key in ('run', 'evaluation', 'audit_depth', 'audit_detection')}
        for path in paths.values():
            path.mkdir(parents=True)
        run, output = paths['run'], paths['evaluation']
        shutil.copyfile(self.plan, run / 'CLIP_PLAN.json')
        shutil.copyfile(self.plan, output / 'CLIP_PLAN.json')
        policy = {'group_id': group['id'], 'max_norm': group['max_norm'], 'grid_sha256': self.plan_sha,
                  'scope': 'global_total_latent_gradient', 'position': 'after_100000_before_lr', 'norm_dtype': 'float32'}
        summary = {'identity': evaluation.LONG_IDENTITY, 'candidate_id': evaluation.LONG_CANDIDATE_ID,
            'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
            'off_control_binding': self.off, 'execution_off_environment_gate': self.envgate,
            'status': 'train_optimization_complete', 'run_role': 'fresh_joint500_clip120_posthoc_followup_training',
            'executed_updates': 500, 'checkpoint_selection': 'fixed final500 only',
            'test_data_opened': False, 'validation_data_opened': False, 'fresh_start': True, 'resume_enabled': False,
            'package_manifest_sha256': self.package_sha, 'gradient_clipping': policy,
            'accepted_r2_source_manifest_sha256': evaluation.R2_MANIFEST,
            'accepted_joint_v2_manifest_sha256': evaluation.JOINT_V2_MANIFEST,
            'training_models': ['fasterrcnn', 'ssd300'], 'training_depth_models': ['monodepth2', 'dpt_large'],
            'depth_weight': 1., 'depth_beta': .1, 'config': {'outer_rounds': 20},
            'budget': {key: dict(evaluation.LONG_BUDGET) for key in ('planned', 'attempted', 'completed')},
            'model_manifest_sha256': collect.MODEL_SHA, 'training_manifest_sha256': collect.TRAINING_SHA}
        threshold = group['max_norm']
        clipped = threshold is not None and 20. > threshold
        diagnostics = {'max_norm': threshold, 'norm_before': 20., 'norm_after': 20. if threshold is None else threshold,
                       'clipped': clipped, 'realized_update_rms': .01}
        trace = [{'global_step': i, 'split': 'train', 'clipping_diagnostics': diagnostics} for i in range(1, 501)]
        (run / 'attack_trace.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in trace))
        summary['attack_trace_sha256'] = collect.sha(run / 'attack_trace.jsonl')
        summary['clipping_summary'] = {'steps': 500, 'triggered_steps': 500 if clipped else 0,
            'triggered_fraction': 1. if clipped else 0., 'max_norm_before': 20., 'max_norm_after': diagnostics['norm_after'],
            'mean_realized_update_rms': sum(.01 for _ in range(500)) / 500}
        summary['schedule_sha256'] = save(run / 'schedule.json', {'synthetic': True})
        summary['runtime_provenance_sha256'] = save(run / 'runtime_provenance.json', self.runtime)
        summary['patch_bindings'] = {name: binary(run / name) for name in ('zero_step_patch.pt', 'final_patch.pt')}
        summary['checkpoint_bindings'] = {('checkpoint_%04d.pt' % step): binary(run / ('checkpoint_%04d.pt' % step)) for step in (100, 250, 500)}
        def stored(name):
            path = run / 'state_evidence' / name
            digest = binary(path)
            return {'path': str(path.relative_to(run)), 'sha256': digest, 'bytes': path.stat().st_size}
        initial = stored('initialization.pt')
        recoveries = [dict(stored('recovery_%04d.pt' % (25 * i)), executed_updates=25 * i,
                          completed_outer_round=i) for i in range(1, 21)]
        state_entries = []
        def entry(phase, step, updates, storage=None):
            state_entries.append({'index': len(state_entries) + 1, 'phase': phase, 'global_step': step,
                 'updates_applied': updates, 'state_fingerprint': '1' * 64,
                 'recording_preserved_rng': True, 'stored_state': storage})
        entry('initialization', 0, 0, initial)
        for outer in range(20):
            entry('outer_start', 0, outer * 25)
            for step in range(outer * 25 + 1, outer * 25 + 26):
                for phase in ('before_denoise', 'pre_backward', 'pre_update', 'post_update'):
                    entry(phase, step, step if phase == 'post_update' else step - 1)
            entry('outer_end', 0, 25 * (outer + 1), {key: recoveries[outer][key] for key in ('path', 'sha256', 'bytes')})
        ledger = run / 'state_evidence/ledger.jsonl'
        ledger.parent.mkdir(exist_ok=True)
        ledger.write_text(''.join(json.dumps(row) + '\n' for row in state_entries))
        reference = {'candidate_id': 'r2_joint_depth_long500_20261001-two-detector-foa-plus-fixed-depth',
                     'summary_sha256': '2' * 64, 'state_manifest_sha256': '3' * 64}
        count = 2041 if threshold is None else 4
        state = {'schema': 'r2-joint-clip500-lean-state-manifest-v1', 'status': 'complete',
                 'candidate_id': summary['candidate_id'], 'executed_updates': 500, 'entries': state_entries,
                 'reference_binding': reference, 'comparison_counts': {'states_checked': count},
                 'gradient_clipping': policy, 'ledger_sha256': collect.sha(ledger),
                 'initialization_state': initial, 'recovery_checkpoints': recoveries}
        summary['state_evidence_manifest_sha256'] = save(run / 'state_evidence/MANIFEST.json', state)
        summary['clipping_comparison_gate_sha256'] = save(run / 'CLIPPING_COMPARISON_GATE.json', {
            'status': 'exact_full_trajectory' if threshold is None else 'exact_initialization_and_first_forward',
            'tolerance': 0, 'compared_states': count, 'reference': reference,
            'max_norm': threshold, 'group_id': group['id'], 'grid_sha256': self.plan_sha})
        training_sha = save(run / 'summary.json', summary)
        method, models = evaluation.run_role(summary, self.package_sha)
        scenes = [{'dataset': d, 'sample_id': d + str(i), 'split': 'validation'} for d in ('kitti', 'nuscenes') for i in range(20)]
        for task, victims, name in [('detection', collect.DETECTORS, 'raw_predictions/MANIFEST.json'),
                                   ('depth', collect.DEPTH_MODELS, 'depth_raw/MANIFEST.json')]:
            entries = []
            for model in victims:
                for scene in scenes:
                    path = task + '_raw/' + model + '_' + scene['sample_id'] + '.pt'
                    hashes = {s: ('9' * 64 if task == 'depth' else {k: '9' * 64 for k in ('boxes', 'scores', 'labels')}) for s in ('clean', 'zero', 'random', 'final')}
                    row = dict(scene, model=model, path=path, sha256=binary(output / path), tensor_sha256=hashes)
                    if task == 'depth':
                        mask = 'depth_raw/mask_' + scene['sample_id'] + '.pt'
                        row.update(semantics=evaluation.SEMANTICS[model], object_mask={
                            'path': mask, 'sha256': binary(output / mask), 'tensor_sha256': '8' * 64})
                    entries.append(row)
            save(output / name, {'model_forward_count': 640, 'entries': entries})
        det_rows = [dict(model=m, dataset=d, state=s, scene_count=40 if d == 'combined' else 20,
                    tnr_defined=True, tnr_det=.4) for m in collect.DETECTORS for s in ('zero', 'random', 'final') for d in ('kitti', 'nuscenes', 'combined')]
        dep_rows = [dict(model=m, dataset=d, reference_state=ref, evaluated_state=ev, scene_count=40 if d == 'combined' else 20,
                    depth_representation=evaluation.SEMANTICS[m], macro={metric: {'mean': .1} for metric in (*collect.DEPTH_METRICS, 'delta_sigma')})
                    for m in collect.DEPTH_MODELS for ref, ev in evaluation.COMPARISONS for d in ('kitti', 'nuscenes', 'combined')]
        det_sha = save(output / 'detection_summary.json', {'updates': 500, 'method': method, 'run_summary_sha256': training_sha,
                       'summary_rows': 48, 'summaries': det_rows})
        dep_sha = save(output / 'depth_summary.json', {'summary_rows': 48, 'records': 640, 'summaries': dep_rows,
                       'raw_manifest_sha256': collect.sha(output / 'depth_raw/MANIFEST.json')})
        save(output / 'rendered_input_hashes.json', {'synthetic': True})
        source_bindings = {'wrapper_sha256': collect.sha(self.source / 'clip_evaluate.py'),
            'clip_training_package_manifest_sha256': self.package_sha, 'model_manifest_sha256': collect.MODEL_SHA,
            'training_manifest_sha256': collect.TRAINING_SHA, 'depth_evaluator_sha256': collect.METRIC_SHA,
            'r2_manifest_sha256': evaluation.R2_MANIFEST}
        evaluated = {'schema': 'r2-joint-clip120-eight-model-development-evaluation-v1', 'status': 'development_validation_complete',
             'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
             'scene_count': 40, 'updates': 500, 'training_budget_updates': 500, 'evaluation_forward_total': 1280,
             'checkpoint_selection': 'fixed_final500', 'test_data_opened': False, 'run': str(run),
             'run_summary_sha256': training_sha, 'candidate_id': summary['candidate_id'], 'method': method,
             'training_models': list(models), 'gradient_clipping': policy, 'validation_scenes': scenes,
             'clipping_comparison_gate_sha256': summary['clipping_comparison_gate_sha256'], 'source_bindings': source_bindings,
             'detection_summary_sha256': det_sha, 'depth_summary_sha256': dep_sha,
             'artifact_bindings': {str(p.relative_to(output)): collect.sha(p) for p in output.rglob('*') if p.is_file()}}
        eval_sha = save(output / 'summary.json', evaluated)
        identity = {'run_summary_sha256': training_sha, 'model_manifest_sha256': collect.MODEL_SHA,
                    'wrapper_sha256': source_bindings['wrapper_sha256'], 'method': method, 'updates': 500,
                    'clip_training_package': {'manifest_sha256': self.package_sha}}
        for task in ('depth', 'detection'):
            audit = {'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
                 'status': task + '_raw_audit_passed', 'evaluation_summary_sha256': eval_sha,
                 'candidate_id': summary['candidate_id'], 'updates': 500, 'checkpoint_selection': 'fixed_final500',
                 'test_data_opened': False, 'audit_source_sha256': collect.sha(self.source / ('clip_audit_' + task + '.py')),
                 'counts': {'rows_checked': 640 if task == 'depth' else 480, 'summaries_checked': 48,
                            'raw_tensor_checks': 800 if task == 'depth' else 1920, 'artifact_sha_checks': 1, 'scalar_checks': 1},
                 'identity_bindings': identity}
            if task == 'depth':
                audit.update(tolerance=0, depth_metric_source_sha256=collect.METRIC_SHA, no_model_loading_or_inference=True)
            else:
                audit.update(training_summary_sha256=training_sha, evaluation_wrapper_sha256=source_bindings['wrapper_sha256'],
                    storage_helpers_sha256=collect.sha(self.source / 'clip_audit_depth.py'), metric_source_sha256=collect.METRIC_SHA,
                    tolerated_iou_count=0, tolerated_iou=[], no_models_loaded_or_called=True)
            save(paths['audit_' + task] / 'summary.json', audit)
        return dict(id=group['id'], **{key: str(path) for key, path in paths.items()})

    def test_complete_actual_receipt_chain_and_units_and_diagnostics(self):
        result = collect.collect_files(self.plan, self.baselines, self.inputs)
        self.assertEqual(len(result['rows']), 44)
        self.assertTrue(result['within_run_controls_and_coverage_verified'])
        self.assertFalse(result['preregistered_grid_point'])
        self.assertFalse(result['critical_threshold_established'])
        self.assertTrue(result['exploratory_posthoc'])
        self.assertEqual(result['group_receipts'][0]['actual_clipping']['triggered_fraction'], 0.)
        self.assertEqual(len(result['group_receipts'][0]['depth_direction_diagnostics']), 4)
        for row in result['rows']:
            self.assertEqual(row['ours_value'], 40. if row['metric'] == 'tnr_det' else 10. if row['metric'] == 'abs_rel_clean_ref' else .1)
        self.assertEqual(len(result['metric_values']), 16)

    def test_modified_metric_or_raw_bytes_are_rejected(self):
        path = Path(self.entries[0]['evaluation']) / 'depth_summary.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            collect.collect_files(self.plan, self.baselines, self.inputs)

    def test_raw_tensor_file_tampering_is_rejected_without_tensor_loading(self):
        output = Path(self.entries[0]['evaluation'])
        manifest = collect.read(output / 'depth_raw/MANIFEST.json')
        path = output / manifest['entries'][0]['path']
        path.write_bytes(b'changed raw bytes after original audit')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            collect.collect_files(self.plan, self.baselines, self.inputs)

    def test_missing_group_and_mismatched_plan_fail_closed(self):
        inputs = collect.read(self.inputs)
        save(self.inputs, dict(inputs, groups=inputs['groups'][:-1]))
        with self.assertRaisesRegex(ValueError, 'cap120 input group'):
            collect.collect_files(self.plan, self.baselines, self.inputs)
        save(self.inputs, dict(inputs, grid_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'plan SHA'):
            collect.collect_files(self.plan, self.baselines, self.inputs)

    def test_audit_verified_flag_cannot_replace_actual_audit_receipt(self):
        receipt = Path(self.entries[0]['audit_depth']) / 'summary.json'
        audit = collect.read(receipt)
        save(receipt, dict(audit, status='failed_retained', audit_verified=True))
        with self.assertRaisesRegex(ValueError, 'audit status'):
            collect.collect_files(self.plan, self.baselines, self.inputs)

    def test_foreign_audit_evaluation_binding_is_rejected(self):
        receipt = Path(self.entries[0]['audit_detection']) / 'summary.json'
        audit = collect.read(receipt)
        save(receipt, dict(audit, evaluation_summary_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'evaluation binding'):
            collect.collect_files(self.plan, self.baselines, self.inputs)


if __name__ == '__main__':
    unittest.main()
