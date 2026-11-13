"""CPU synthetic evidence/metric tests. No pretrained weights or source image reads."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import inputs
from training import evaluate_validation as evaluation
from training import compare_results as comparison


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = inputs.parent_evaluator()
        cls.manifest = inputs.PARENT / '01_公共数据集/runtime_v1/train_validation_manifest.csv'
        cls.models = inputs.PARENT / '02_模型/model_manifest.json'
        cls.records = cls.engine.training_records(cls.manifest)
        cls.protocol = inputs.load_protocol()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'protocol').mkdir()
        for name in ('training_protocol.json', 'train_schedule.json'):
            shutil.copyfile(ROOT / 'protocol' / name, self.root / 'protocol' / name)
        self.run = self.make_run('C0')

    def write(self, path, data):
        evaluation.write_json(path, data)

    @staticmethod
    def contextual_stats(model):
        return {'objective':'contextual_car_class_probability','direction':'minimize',
                'model_id':model,'class_index':3,
                'matching_rule':('rpn_proposals_plus_gt_iou_ge_native_roi_fg_threshold' if model=='fasterrcnn'
                                 else 'native_ssd_matcher_including_best_anchor_for_gt'),
                'matching_iou_threshold':.5,'matched_prediction_count':1,'candidate_count':2,
                'model_forward_equivalents':1,'backbone_forward_calls':1,'classifier_forward_calls':1,
                'extra_clean_forward_calls':0,'adaptation':'road_patch_gt_matched_pre_nms_class_suppression'}

    def make_run(self, objective):
        run = self.root / 'runs' / ('train_' + objective)
        run.mkdir(parents=True)
        identity = inputs.IDENTITY + '-ours-det-only-' + objective
        schedule = evaluation.read_json(self.root / 'protocol/train_schedule.json')
        shutil.copyfile(self.root / 'protocol/train_schedule.json', run / 'schedule.json')
        zero, final = torch.full((1, 3, 64, 128), .2), torch.full((1, 3, 64, 128), .6)
        for name, rgb in (('zero_step_patch.pt', zero), ('final_patch.pt', final)):
            self.engine.save_patch(run / name, rgb, {'candidate_id': identity})
        checkpoints = {}
        for name, step in self.engine.CHECKPOINTS.items():
            self.engine.save_patch(run / name, final, {'updates': step, 'candidate_id': identity})
            checkpoints[name] = inputs.sha(run / name)
        ids = self.protocol['calibration']['sample_ids']
        originals = {r.sample_id: r for r in self.records}
        normalizers = {'fasterrcnn': .25, 'ssd300': .5}
        samples = []
        for name in ids:
            record = originals[name]
            samples.append({'sample_id': name, 'dataset': record.dataset, 'split': 'train',
                            'source_sha256': record.source_sha256, 'road_quad_norm': record.road_quad_norm,
                            'eot': None, 'models': {k: {'raw_loss': v, 'absolute_loss': v,
                                'diagnostics':self.contextual_stats(k) if objective=='C3' else {}}
                                for k, v in normalizers.items()}})
        calibration = {'schema': 'v23_fixed_train_loss_calibration_v1', 'candidate_id': identity,
                       'attack_objective': objective, 'protocol_sha256': inputs.sha(self.root / 'protocol/training_protocol.json'),
                       'source': 'fixed_first_train_road_crop_before_vae_noise_or_optimization',
                       'eot': None, 'victim_forward_calls': 8, 'autograd_calls': 0,
                       'model_loss_attempts': 8, 'model_loss_completed': 8,
                       'initial_crop_shape': list(zero.shape), 'initial_crop_dtype': str(zero.dtype),
                       'initial_crop_tensor_sha256': hashlib.sha256(zero.numpy().tobytes()).hexdigest(),
                       'optimized_v20_states_used': False, 'test_data_opened': False,
                       'validation_data_opened': False, 'updated_parameters': 0, 'sample_ids': ids, 'samples': samples,
                       'normalizers': normalizers, 'models': {k: {'raw_sample_absolute_losses': [v]*4,
                           'raw_mean_absolute': v, 'normalizer': v, 'floor': 1e-6, 'floor_applied': False}
                           for k, v in normalizers.items()}}
        self.write(run / 'calibration.json', calibration)
        calibration_sha = inputs.sha(run / 'calibration.json')
        from ours_stateful import StatefulOursConfig, build_execution_plan
        from training.state_evidence import fingerprint
        config = StatefulOursConfig(**self.protocol['stateful_sd35'])
        plan = build_execution_plan(config, self.engine.seeded_round_robin(self.records, 100, 20260919))
        state_dir = run / 'state_evidence'; state_dir.mkdir()
        entries, trace, previous = [], [], None
        def save_state(phase, step, state):
            nonlocal previous
            index = len(entries) + 1
            name = f'state_evidence/{index:04d}_{phase}_step{step:04d}.pt'
            torch.save({'schema': 'v23-training-state-v1', 'metadata': {'candidate_id': identity,
                'attack_objective': objective, 'global_step': step, 'phase': phase, 'resume_enabled': False},
                'state': state}, run / name)
            entry = {'index': index, 'phase': phase, 'global_step': step, 'path': name,
                     'sha256': inputs.sha(run / name), 'bytes': (run / name).stat().st_size,
                     'previous_entry_sha256': previous, 'recording_preserved_rng': True,
                     'recording_rng_before_sha256': 'a'*64, 'recording_rng_after_sha256': 'a'*64}
            entry['entry_sha256'] = hashlib.sha256(evaluation.canonical_json(entry)).hexdigest()
            previous = entry['entry_sha256']; entries.append(entry)
            return {'path': name, 'sha256': entry['sha256']}
        save_state('initialization', 0, {'config': config.to_dict(), 'normalizers': normalizers, 'calibration_sha256': calibration_sha})
        for step, scheduled in enumerate(schedule, 1):
            outer = (step-1)//25 + 1
            if (step-1)%25 == 0:
                save_state('outer_start', 0, {'outer_round': outer})
            state_bindings = {}
            losses={k:{'raw':v,'normalized':1.,'objective_diagnostics_by_eot_draw':[
                self.contextual_stats(k) if objective=='C3' else {} for _ in range(2)]}
                for k,v in normalizers.items()}
            for phase in evaluation.PHASES:
                state = {'latents': torch.tensor([step], dtype=torch.bfloat16)}
                if phase == 'before_denoise':
                    state.update(planned=asdict(plan[step-1]), normalizers=normalizers, calibration_sha256=calibration_sha)
                if phase == 'gradients':state['losses_pre']=losses
                if phase == 'post_update':state['losses_post']=losses
                state_bindings[phase] = save_state(phase, step, state)
            trace.append(dict(scheduled, **asdict(plan[step-1]) | {'split':'train'}, state_bindings=state_bindings,
                              attack_objective=objective, task_variant='det-only', training_models=['fasterrcnn','ssd300'],
                              candidate_id=identity, normalizer_fingerprint=fingerprint(normalizers),
                              model_count=2, eot_samples_per_update=2, victim_forwards_minimum_this_update=8,
                              latent_backward_queries_this_update=2, latent_lr=.05, calibration_sha256=calibration_sha))
            trace[-1].update(losses_pre=losses,losses_post=losses)
            if step%25 == 0:
                save_state('outer_end', 0, {'outer_round': outer})
        (state_dir / 'ledger.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in entries))
        self.write(state_dir / 'MANIFEST.json', {'schema': 'v23-training-state-manifest-v1',
                   'candidate_id': identity, 'attack_objective': objective, 'status': 'complete', 'executed_updates': 100,
                   'resume_enabled': False, 'entries': entries, 'ledger_sha256': inputs.sha(state_dir / 'ledger.jsonl')})
        (run / 'attack_trace.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in trace))
        self.write(run / 'runtime_provenance.json', {'scope': 'synthetic CPU test; no real GPU execution'})
        summary = {'candidate_id': identity, 'attack_objective': objective, 'task_variant': 'det-only',
                   'status': 'train_optimization_complete', 'executed_updates': 100, 'seed': 20260919,
                   'training_models': ['fasterrcnn', 'ssd300'], 'test_data_opened': False,
                   'validation_data_opened': False, 'formal_test_result': False, 'resume_enabled': False,
                   'config': config.to_dict(), 'road_eot': self.protocol['road_eot'],
                   'budget': {k: self.protocol['budget_per_run'] for k in ('planned','attempted','completed')},
                   'query_accounting': self.protocol['budget_per_run'],
                   'training_manifest_sha256': inputs.sha(self.manifest), 'model_manifest_sha256': inputs.sha(self.models),
                   'protocol_sha256': inputs.sha(self.root / 'protocol/training_protocol.json'),
                   'schedule_sha256': inputs.sha(run / 'schedule.json'), 'attack_trace_sha256': inputs.sha(run / 'attack_trace.jsonl'),
                   'calibration_sha256': inputs.sha(run / 'calibration.json'), 'normalizers': normalizers,
                   'calibration_sample_ids': ids, 'state_evidence_manifest_sha256': inputs.sha(state_dir / 'MANIFEST.json'),
                   'runtime_provenance_sha256': inputs.sha(run / 'runtime_provenance.json'),
                   'patch_bindings': {x: inputs.sha(run / x) for x in ('zero_step_patch.pt', 'final_patch.pt')},
                   'checkpoint_bindings': checkpoints}
        self.write(run / 'summary.json', summary)
        return run

    def validate(self, run=None, records=None):
        return evaluation.validate_completed_run(run or self.run, self.records if records is None else records,
                    self.manifest, self.models, root=self.root, protocol=self.protocol)

    def edit_summary(self, **changes):
        summary = evaluation.read_json(self.run / 'summary.json'); summary.update(changes)
        self.write(self.run / 'summary.json', summary)

    def test_complete_new_identity_and_all_400_states(self):
        checked = self.validate()
        self.assertEqual(checked['state_review']['complete_update_phases'], 400)
        self.assertEqual(checked['summary']['attack_objective'], 'C0')
        self.assertEqual(self.validate(self.make_run('C3'))['summary']['attack_objective'], 'C3')

    def test_old_identity_and_incomplete_updates_rejected(self):
        for updates, identity in ((100, 'normalized-v17-roadpatch-20260920-ours-det-only'),
                                   (100,'normalized-v21-ssd-training-20260920-ours-det-only-C0'),
                                   (100,'normalized-v22-ssd-fallback-20260920-ours-det-only-C0'),
                                   (99, inputs.IDENTITY + '-ours-det-only-C0')):
            self.edit_summary(executed_updates=updates, candidate_id=identity)
            with self.assertRaises(ValueError): self.validate()

    def test_legacy_ssd_only_field_and_old_objectives_are_rejected(self):
        summary=evaluation.read_json(self.run/'summary.json')
        for objective in ('C1','C2','c3'):
            wrong=deepcopy(summary);wrong.update(attack_objective=objective,candidate_id=inputs.IDENTITY+'-ours-det-only-'+objective)
            with self.assertRaisesRegex(ValueError,'objective identity'):
                evaluation.require_current_objective(wrong)
        wrong=deepcopy(summary);wrong['ssd_objective']='C0'
        self.write(self.run/'summary.json',wrong)
        with self.assertRaisesRegex(ValueError,'legacy SSD-only'):
            self.validate()

    def test_C3_requires_contextual_diagnostics_for_both_detectors(self):
        run=self.make_run('C3');summary=evaluation.read_json(run/'summary.json')
        value=evaluation.read_json(run/'calibration.json')
        value['samples'][0]['models']['fasterrcnn']['diagnostics']={}
        self.write(run/'calibration.json',value);summary['calibration_sha256']=inputs.sha(run/'calibration.json')
        with self.assertRaisesRegex(ValueError,'Contextual-road.*fasterrcnn'):
            evaluation.validate_calibration(run,summary,self.protocol,self.records)
        loss={m:{'raw':.5,'normalized':1.,'objective_diagnostics_by_eot_draw':[self.contextual_stats(m) for _ in range(2)]}
              for m in ('fasterrcnn','ssd300')}
        evaluation.validate_loss_observations(loss,'C3')
        loss['ssd300']['objective_diagnostics_by_eot_draw'][1]['class_index']=8
        with self.assertRaisesRegex(ValueError,'Contextual-road.*ssd300'):
            evaluation.validate_loss_observations(loss,'C3')

    def test_test_split_and_foreign_output_rejected(self):
        records = [replace(self.records[0], split='test'), *self.records[1:]]
        with self.assertRaises(ValueError): self.validate(records=records)
        with self.assertRaises(ValueError):
            evaluation.candidate_run_path(self.root / 'parent_v17/runs/fake', self.root)

    def test_calibration_scale_cannot_reuse_other_objective_or_tamper(self):
        value = evaluation.read_json(self.run / 'calibration.json'); value['attack_objective'] = 'C3'
        self.write(self.run / 'calibration.json', value)
        self.edit_summary(calibration_sha256=inputs.sha(self.run / 'calibration.json'))
        with self.assertRaisesRegex(ValueError, 'objective'):
            evaluation.validate_calibration(self.run, evaluation.read_json(self.run / 'summary.json'), self.protocol, self.records)
        value['attack_objective'] = 'C0'; value['normalizers']['ssd300'] = .7
        self.write(self.run / 'calibration.json', value)
        self.edit_summary(calibration_sha256=inputs.sha(self.run / 'calibration.json'), normalizers=value['normalizers'])
        with self.assertRaisesRegex(ValueError, 'arithmetic'):
            evaluation.validate_calibration(self.run, evaluation.read_json(self.run / 'summary.json'), self.protocol, self.records)

    def test_frozen_config_and_actual_callback_budget_are_required(self):
        original = evaluation.read_json(self.run / 'summary.json')
        wrong = deepcopy(original); wrong['config']['outer_rounds'] = 16
        self.write(self.run / 'summary.json', wrong)
        with self.assertRaisesRegex(ValueError, 'configuration'): self.validate()
        wrong = deepcopy(original); wrong['budget']['completed']['autograd_calls'] = 199
        self.write(self.run / 'summary.json', wrong)
        with self.assertRaisesRegex(ValueError, 'budget'): self.validate()
        wrong = deepcopy(original); wrong['budget']['completed']['autograd_calls'] = 200.0
        self.write(self.run / 'summary.json', wrong)
        with self.assertRaisesRegex(ValueError, 'budget'): self.validate()

    def test_missing_outer_state_cannot_be_a_completed_400_phase_run(self):
        path = self.run / 'state_evidence/MANIFEST.json'
        manifest = evaluation.read_json(path); manifest['entries'].pop()
        ledger = self.run / 'state_evidence/ledger.jsonl'
        ledger.write_text(''.join(json.dumps(x)+'\n' for x in manifest['entries']))
        manifest['ledger_sha256'] = inputs.sha(ledger)
        self.write(path, manifest); self.edit_summary(state_evidence_manifest_sha256=inputs.sha(path))
        with self.assertRaisesRegex(ValueError, '409 states'): self.validate()

    def test_default_preflight_does_not_construct_model_or_read_source_images(self):
        args = SimpleNamespace(run=False)
        with patch.object(evaluation, 'parse_args', return_value=args), \
             patch.object(evaluation, 'verify_inputs', return_value={'sealed':True}), \
             patch.object(evaluation, 'parent_evaluator', side_effect=AssertionError('model path used')), \
             patch('builtins.print'):
            evaluation.main()

    def test_state_bytes_and_trace_binding_cannot_be_changed(self):
        path = next((self.run / 'state_evidence').glob('*.pt'))
        with path.open('ab') as stream: stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA'): self.validate()

    def test_ledger_chain_is_not_only_manifest_hash(self):
        path = self.run / 'state_evidence/MANIFEST.json'
        value = evaluation.read_json(path)
        value['entries'][17]['previous_entry_sha256'] = '0'*64
        ledger = self.run / 'state_evidence/ledger.jsonl'
        ledger.write_text(''.join(json.dumps(x)+'\n' for x in value['entries']))
        value['ledger_sha256'] = inputs.sha(ledger)
        self.write(path, value); self.edit_summary(state_evidence_manifest_sha256=inputs.sha(path))
        with self.assertRaisesRegex(ValueError, 'chain'): self.validate()

    def test_checkpoint100_is_fixed_not_best(self):
        self.engine.save_patch(self.run / 'final_patch.pt', torch.full((1,3,64,128), .1), {})
        summary = evaluation.read_json(self.run / 'summary.json')
        summary['patch_bindings']['final_patch.pt'] = inputs.sha(self.run / 'final_patch.pt')
        self.write(self.run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'checkpoint100'): self.validate()

    def synthetic_evaluation(self, objective):
        run = self.run if objective == 'C0' else self.make_run(objective)
        checked = self.validate(run)
        output = self.root / 'runs' / ('validation_' + objective); output.mkdir()
        context = SimpleNamespace(background=torch.full((1,3,12,20), .2), benign=torch.full((1,3,12,20), .2),
                                  object_mask=torch.zeros((1,1,12,20)), object_box=(5,2,9,5),
                                  road_quad=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]]))
        context.object_mask[...,2:5,5:9] = 1
        class Render:
            def __init__(self, rgb): self.rgb = rgb
            def render(self, ctx):
                support = torch.zeros((1,1,12,20)); support[...,8:11,6:14] = 1
                return SimpleNamespace(image=torch.where(support.bool(), self.rgb.mean(), ctx.benign), support=support)
        class Adapter:
            def predict(self, image):
                return [{'boxes': torch.tensor([[5.,2.,9.,5.]]), 'labels': torch.tensor([3]),
                         'scores': image.mean().reshape(1)}]
        args = SimpleNamespace(output=output, car=None, car_mask=None, model_manifest=self.models, weights_root=None)
        original_summary = self.engine.detection_summary
        with patch.object(evaluation, 'ROOT', self.root), patch.object(self.engine, 'load_car', return_value=(None,None)), \
             patch.object(self.engine, 'build_context', return_value=context), \
             patch.object(self.engine, 'extract_road_patch', return_value=checked['patches']['zero']), \
             patch.object(self.engine, 'RoadRGBView', Render), \
             patch.object(self.engine, 'build_offline_detector', return_value=Adapter()), \
             patch.object(self.engine, 'detection_summary', side_effect=lambda rows, n, seed, **kw: original_summary(rows, 2, seed, **kw)):
            result = evaluation.evaluate(args, self.records, checked, torch.device('cpu'))
        result.update(package_manifest_sha256='d'*64, artifact_bindings=evaluation.artifact_bindings(output))
        self.write(output / 'summary.json', result)
        return comparison.load_evaluation(output, objective, root=self.root, bootstrap_samples=2)

    def test_full_parent_metric_cpu_integration_and_common_controls(self):
        a, b = self.synthetic_evaluation('C0'), self.synthetic_evaluation('C3')
        result = comparison.compare(a, b)
        self.assertEqual(result['status'], 'complete_observations')
        self.assertTrue(result['shared_control_predictions_exact'])
        self.assertEqual(len(result['tables']), 24)
        self.assertEqual(a['summary']['evaluation_forward_total'], 320)
        self.assertEqual(len(list(a['path'].glob('*inputs.pt'))), 2)
        self.assertEqual(len(list(a['path'].glob('*predictions.pt'))), 4)
        self.assertEqual(len(list(a['path'].glob('*.png'))), 16)
        self.assertEqual(len(a['hashes']), 240)
        self.assertIn('zero→final', comparison.markdown(result))

    def test_prediction_drift_is_reported_not_cached_or_failed(self):
        a = self.synthetic_evaluation('C0'); b = deepcopy(a)
        b['summary'].update(attack_objective='C3',candidate_id=inputs.IDENTITY+'-ours-det-only-C3')
        key = next(iter(b['index']))
        b['index'][key]['clean_candidates'][0]['score'] += .01
        result = comparison.compare(a, b)
        self.assertFalse(result['shared_control_predictions_exact'])
        self.assertTrue(result['warnings'])
        self.assertEqual(result['status'], 'complete_observations')

    def test_actual_control_input_mismatch_fails(self):
        a = self.synthetic_evaluation('C0'); b = deepcopy(a)
        b['summary'].update(attack_objective='C3',candidate_id=inputs.IDENTITY+'-ours-det-only-C3')
        key = next(k for k in b['hashes'] if k[-1] == 'zero')
        b['hashes'][key]['rendered_input_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'input differs'): comparison.compare(a, b)

    def test_comparison_requires_new_C0_and_C3_and_explicit_scope(self):
        a=self.synthetic_evaluation('C0')
        with self.assertRaisesRegex(ValueError,'objective identity'):
            comparison.compare(a,deepcopy(a))
        b=self.synthetic_evaluation('C3');report=comparison.compare(a,b)
        self.assertEqual(b['summary']['training_target_classes'],[3])
        self.assertEqual(b['summary']['evaluation_target_classes'],[3,6,8])
        self.assertEqual(b['summary']['training_objective_models'],['fasterrcnn','ssd300'])
        self.assertIn('两个模型',comparison.markdown(report))
        self.assertIn('C3_minus_C0',report['differences'][0])

    def test_zero_reference_tnr_is_null_no_nan_or_selection(self):
        a = self.synthetic_evaluation('C0')
        rows = deepcopy(a['rows'])
        for row in rows:
            if row['state'] == 'zero':
                row['adversarial_candidates'] = []; row['adversarial_car_candidates'] = []
        result = comparison.zero_to_final(rows, bootstrap_samples=2)
        self.assertEqual(len(result), 6)
        self.assertTrue(all(x['tnr_det'] is None and not x['tnr_defined'] for x in result))
        json.dumps(result, allow_nan=False)

    def test_negative_tnr_is_retained_without_changing_population(self):
        a=self.synthetic_evaluation('C0');rows=deepcopy(a['rows'])
        for row in rows:
            for field in ('clean_candidates','clean_car_candidates'):
                row[field]=[{'score':.9,'iou':.2,'label':3},{'score':.8,'iou':.8,'label':3}]
            for field in ('adversarial_candidates','adversarial_car_candidates'):
                row[field]=[{'score':.95,'iou':.8,'label':3}]
        values=comparison.summarize_rows(rows,bootstrap_samples=2)
        self.assertEqual(len(values),18)
        self.assertTrue(all(r['tnr_det']==-1. for r in values))


if __name__ == '__main__':
    unittest.main()
