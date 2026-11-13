"""CPU-only contract checks for final500 evaluation; no models, images or CUDA.

Run: python -B test_clip_evaluation.py --r2-source /frozen/r2 [-v]
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from test_cap120_fixtures import environment_evidence

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--r2-source', type=Path, default=ROOT.parent / 'meeting_full_repeat_20260924_r2')
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(ROOT))
import clip_evaluate as evaluation
import clip_audit_depth as storage
import clip_audit_detection as detection


def load(path):
    spec = importlib.util.spec_from_file_location('_long_eval_test_compare', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def policy(group='cap120', threshold=120.0, grid_sha='7' * 64):
    return {'group_id': group, 'max_norm': threshold, 'grid_sha256': grid_sha,
            'scope': 'global_total_latent_gradient', 'position': 'after_100000_before_lr', 'norm_dtype': 'float32'}


def training_summary(package):
    return {'identity': evaluation.LONG_IDENTITY, 'candidate_id': evaluation.LONG_CANDIDATE_ID,
            'gradient_clipping': policy(), 'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
            'status': 'train_optimization_complete', 'run_role': 'fresh_joint500_clip120_posthoc_followup_training',
            'executed_updates': 500, 'checkpoint_selection': 'fixed final500 only',
            'test_data_opened': False, 'validation_data_opened': False, 'fresh_start': True,
            'resume_enabled': False, 'package_manifest_sha256': package,
            'accepted_r2_source_manifest_sha256': evaluation.R2_MANIFEST,
            'accepted_joint_v2_manifest_sha256': evaluation.JOINT_V2_MANIFEST,
            'training_models': ['fasterrcnn', 'ssd300'],
            'training_depth_models': ['monodepth2', 'dpt_large'],
            'depth_weight': 1.0, 'depth_beta': .1, 'config': {'outer_rounds': 20},
            'budget': {key: dict(evaluation.LONG_BUDGET) for key in ('planned', 'attempted', 'completed')}}


class ClipEvaluationTests(unittest.TestCase):
    def setUp(self):
        # Historical literal source SHA validation has independent contract tests.
        # This fixture tests evaluator ledger/patch/receipt gates using synthetic files.
        gate = mock.patch('clip_evidence.validate_followup_plan', side_effect=lambda plan: plan)
        gate.start(); self.addCleanup(gate.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        self.package.mkdir()
        names = ('clip_training.py', 'clip_evaluate.py', 'clip_audit_depth.py', 'clip_audit_detection.py')
        for name in names:
            if (ROOT / name).is_file():
                shutil.copyfile(ROOT / name, self.package / name)
            else:
                (self.package / name).write_text('# synthetic source member\n')
        save(self.package / 'PACKAGE_MANIFEST.json', {'files': [
            {'path': name, 'sha256': digest(self.package / name), 'bytes': (self.package / name).stat().st_size}
            for name in names]})
        self.package_sha = digest(self.package / 'PACKAGE_MANIFEST.json')

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_package_binding_and_mutation_rejection(self):
        self.assertEqual(evaluation.bind_long_package(self.package, self.package_sha)['verified_members'], 4)
        for invalid in (None, '', 'PENDING', '0' * 64, self.package_sha.upper()):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation.bind_long_package(self.package, invalid)
        with (self.package / 'clip_training.py').open('a') as stream:
            stream.write('\n# synthetic mismatch\n')
        with self.assertRaisesRegex(ValueError, 'member mismatch'):
            evaluation.bind_long_package(self.package, self.package_sha)

    def test_role_accepts_only_fresh_fixed500_and_exact_budget(self):
        valid = training_summary(self.package_sha)
        method, models = evaluation.run_role(valid, self.package_sha)
        self.assertEqual(method, 'joint_fixed_depth_weight_one_final500_posthoc_cap120')
        self.assertEqual(models, ('fasterrcnn', 'ssd300', 'monodepth2', 'dpt_large'))
        for field, value in [('identity', 'r2_joint_depth_20261001'), ('executed_updates', 100),
                             ('depth_weight', 0.), ('depth_beta', .2), ('fresh_start', False),
                             ('resume_enabled', True), ('exploratory_posthoc', False), ('selection_context', 'preregistered'), ('validation_data_opened', True),
                             ('package_manifest_sha256', '0' * 64), ('training_depth_models', ['monodepth2'])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluation.run_role(dict(valid, **{field: value}), self.package_sha)
        wrong = copy.deepcopy(valid)
        wrong['budget']['completed']['autograd_calls'] = 499
        with self.assertRaisesRegex(ValueError, 'call budget'):
            evaluation.run_role(wrong, self.package_sha)

    def make_completed_run(self, group='cap120', threshold=120.0):
        import torch
        run, parent = self.root / 'run', self.root / 'parent'
        run.mkdir()
        (run / 'state_evidence').mkdir()
        summary = training_summary(self.package_sha)
        for field, name in [('training_manifest_sha256', '01_公共数据集/runtime_v1/train_validation_manifest.csv'),
                            ('model_manifest_sha256', '02_模型/model_manifest.json')]:
            path = parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic CPU evidence', encoding='utf-8')
            summary[field] = digest(path)
        def stored(name):
            path = run / 'state_evidence' / name
            torch.save({'synthetic': torch.tensor([1.0])}, path)
            return {'path': str(path.relative_to(run)), 'sha256': digest(path), 'bytes': path.stat().st_size}
        initial = stored('initialization.pt')
        recovery = [dict(stored('recovery_%04d.pt' % (25 * i)), completed_outer_round=i, executed_updates=25 * i)
                    for i in range(1, 21)]
        entries = []
        def entry(phase, step, updates, state=None):
            entries.append({'index': len(entries) + 1, 'phase': phase, 'global_step': step,
                            'updates_applied': updates, 'state_fingerprint': '1' * 64,
                            'stored_state': state, 'recording_preserved_rng': True})
        entry('initialization', 0, 0, initial)
        for outer in range(20):
            entry('outer_start', 0, outer * 25)
            for step in range(outer * 25 + 1, outer * 25 + 26):
                for phase in ('before_denoise', 'pre_backward', 'pre_update', 'post_update'):
                    entry(phase, step, step if phase == 'post_update' else step - 1)
            entry('outer_end', 0, (outer + 1) * 25,
                  {key: recovery[outer][key] for key in ('path', 'sha256', 'bytes')})
        ledger = run / 'state_evidence/ledger.jsonl'
        ledger.write_text(''.join(json.dumps(row) + '\n' for row in entries), encoding='utf-8')
        reference = {'candidate_id': 'r2_joint_depth_long500_20261001-two-detector-foa-plus-fixed-depth',
                     'summary_sha256': '2' * 64, 'state_manifest_sha256': '3' * 64, 'reference_run': '/synthetic'}
        plan = {'schema': 'r2-joint-clip120-followup-plan-v1', 'status': 'fixed_user_requested_post_evaluation_followup',
                'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
                'groups': [{'id': 'cap120', 'max_norm': 120.0, 'executed_updates': 500}],
                'reference': {'summary_sha256': reference['summary_sha256'],
                              'trace_sha256': '6' * 64, 'state_evidence_manifest_sha256': reference['state_manifest_sha256'],
                              'training_manifest_sha256': summary['training_manifest_sha256']}}
        runtime, execution, off, envgate = environment_evidence(self.package_sha)
        plan['execution_off'] = execution
        summary.update(off_control_binding=off, execution_off_environment_gate=envgate)
        save(run / 'runtime_provenance.json', runtime)
        summary['runtime_provenance_sha256'] = digest(run / 'runtime_provenance.json')
        save(run / 'CLIP_PLAN.json', plan)
        summary['gradient_clipping'] = policy(group, threshold, digest(run / 'CLIP_PLAN.json'))
        counts = {'states_checked': 2041 if threshold is None else 4}
        gate = {'status': 'exact_full_trajectory' if threshold is None else 'exact_initialization_and_first_forward',
                'tolerance': 0, 'reference': reference, 'compared_states': counts['states_checked'],
                'max_norm': threshold, 'group_id': group, 'grid_sha256': summary['gradient_clipping']['grid_sha256']}
        save(run / 'CLIPPING_COMPARISON_GATE.json', gate)
        summary['clipping_comparison_gate_sha256'] = digest(run / 'CLIPPING_COMPARISON_GATE.json')
        manifest = {'schema': 'r2-joint-clip500-lean-state-manifest-v1', 'status': 'complete',
                    'candidate_id': evaluation.LONG_CANDIDATE_ID, 'executed_updates': 500,
                    'gradient_clipping': summary['gradient_clipping'], 'entries': entries, 'initialization_state': initial, 'recovery_checkpoints': recovery,
                    'ledger_sha256': digest(ledger), 'reference_binding': reference, 'comparison_counts': counts}
        save(run / 'state_evidence/MANIFEST.json', manifest)
        summary['state_evidence_manifest_sha256'] = digest(run / 'state_evidence/MANIFEST.json')
        summary['checkpoint_bindings'] = {}
        for step in (100, 250, 500):
            path = run / ('checkpoint_%04d.pt' % step)
            torch.save({'rgb': torch.ones(1, 3, 2, 2) * .25, 'metadata': {'updates': step}}, path)
            summary['checkpoint_bindings'][path.name] = digest(path)
        summary['patch_bindings'] = {}
        for name in ('zero_step_patch.pt', 'final_patch.pt', 'initial_vae_reconstruction.pt'):
            path = run / name
            torch.save({'rgb': torch.ones(1, 3, 2, 2) * .25}, path)
            summary['patch_bindings'][name] = digest(path)
        save(run / 'summary.json', summary)
        engine = SimpleNamespace(load_patch=lambda p: torch.load(p, weights_only=True, map_location='cpu')['rgb'],
                                 common_random_patch=lambda: torch.ones(1, 3, 2, 2) * .5)
        return run, parent, summary, manifest, engine

    def test_complete_lean_schema_and_final500_patch_gate(self):
        import torch
        run, parent, summary, manifest, engine = self.make_completed_run()
        compare = load(args.r2_source / 'compare.py')
        result, patches = evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)
        self.assertEqual(result['executed_updates'], 500)
        self.assertEqual(set(patches), {'zero', 'random', 'final'})
        self.assertEqual(len(manifest['entries']), 2041)
        torch.save({'rgb': torch.ones(1, 3, 2, 2) * .75}, run / 'final_patch.pt')
        summary['patch_bindings']['final_patch.pt'] = digest(run / 'final_patch.pt')
        save(run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'fixed checkpoint500'):
            evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)
        manifest['entries'][2]['updates_applied'] = 8
        save(run / 'state_evidence/MANIFEST.json', manifest)
        summary['state_evidence_manifest_sha256'] = digest(run / 'state_evidence/MANIFEST.json')
        save(run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, '2041-record'):
            evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)

    def test_both_auditors_share_actual_run_binding(self):
        run, parent, training, manifest, engine = self.make_completed_run()
        child = self.root / 'child'
        name = 'parent_v17/02_模型/model_manifest.json'
        path = child / name
        path.parent.mkdir(parents=True)
        shutil.copyfile(parent / '02_模型/model_manifest.json', path)
        summary = {'run': str(run), 'run_summary_sha256': digest(run / 'summary.json'),
                   'candidate_id': training['candidate_id'], 'method': evaluation.run_role(training, self.package_sha)[0],
                   'training_models': list(evaluation.run_role(training, self.package_sha)[1]),
                   'gradient_clipping': training['gradient_clipping'],
                   'selection_context': evaluation.SELECTION_CONTEXT, 'exploratory_posthoc': True,
                   'clipping_comparison_gate_sha256': training['clipping_comparison_gate_sha256'],
                   'source_bindings': {'wrapper_sha256': digest(ROOT / 'clip_evaluate.py'),
                                       'clip_training_package_manifest_sha256': self.package_sha,
                                       'model_manifest_sha256': training['model_manifest_sha256'],
                                       'training_manifest_sha256': training['training_manifest_sha256']}}
        entries = {name: {'sha256': digest(path)}}
        self.assertIs(detection.storage.bind_run_model_wrapper, storage.bind_run_model_wrapper)
        binding = storage.bind_run_model_wrapper(summary, child, entries, self.package_sha, self.package)
        self.assertEqual(binding['updates'], 500)
        summary['method'] = 'original_r2_detection_only'
        with self.assertRaisesRegex(ValueError, 'method/training models'):
            storage.bind_run_model_wrapper(summary, child, entries, self.package_sha, self.package)

    def test_active_clipping_accepts_four_states_but_never_fake_full_trajectory(self):
        run, parent, summary, manifest, engine = self.make_completed_run()
        compare = load(args.r2_source / 'compare.py')
        evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)
        gate = json.loads((run / 'CLIPPING_COMPARISON_GATE.json').read_text())
        gate.update(status='exact_full_trajectory', compared_states=2041)
        save(run / 'CLIPPING_COMPARISON_GATE.json', gate)
        summary['clipping_comparison_gate_sha256'] = digest(run / 'CLIPPING_COMPARISON_GATE.json')
        save(run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'mode/count/tolerance'):
            evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)

    def test_policy_rejects_nonfinite_bool_unknown_and_wrong_semantics(self):
        for field, value in [('group_id', 'best_validation'), ('max_norm', True), ('max_norm', float('nan')),
                             ('max_norm', float('inf')), ('max_norm', -1.), ('max_norm', None),
                             ('norm_dtype', 'bfloat16'), ('scope', 'per_detector'), ('position', 'before_multiplier'),
                             ('grid_sha256', 'PENDING')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                evaluation.validate_clip_policy(dict(policy(), **{field: value}))
        for value in (0., 1.):
            with self.assertRaises(ValueError):
                evaluation.validate_clip_policy(policy('off', value))
        with self.assertRaises(ValueError):
            evaluation.validate_clip_policy(policy('zero', 0.))
        evaluation.validate_clip_policy(policy())

    def test_group_threshold_and_reference_must_match_frozen_plan(self):
        run, parent, summary, manifest, engine = self.make_completed_run()
        compare = load(args.r2_source / 'compare.py')
        for field, value in [('max_norm', 13.), ('group_id', 'p90'), ('grid_sha256', '0' * 64)]:
            wrong = copy.deepcopy(summary)
            wrong['gradient_clipping'][field] = value
            save(run / 'summary.json', wrong)
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)
        save(run / 'summary.json', summary)
        manifest['reference_binding']['summary_sha256'] = '0' * 64
        save(run / 'state_evidence/MANIFEST.json', manifest)
        summary['state_evidence_manifest_sha256'] = digest(run / 'state_evidence/MANIFEST.json')
        save(run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)

    def test_driver_or_execution_off_binding_mutation_is_rejected(self):
        run, parent, summary, manifest, engine = self.make_completed_run()
        compare = load(args.r2_source / 'compare.py')
        summary['off_control_binding']['compared_states'] = 4
        save(run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'execution OFF binding'):
            evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)
        summary['off_control_binding']['compared_states'] = 2041
        runtime = json.loads((run / 'runtime_provenance.json').read_text())
        runtime['gpu_identity'] = 'SYNTHETIC DEVICE, GPU-SYNTHETIC, CHANGED-DRIVER'
        save(run / 'runtime_provenance.json', runtime)
        summary['runtime_provenance_sha256'] = digest(run / 'runtime_provenance.json')
        save(run / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'gpu_driver'):
            evaluation.checked_long_run(run, engine, parent, compare, self.package_sha)

    def test_scientific_evaluation_and_metric_functions_are_unchanged(self):
        import ast
        old_root = ROOT.parent / 'r2_joint_depth_long500_20261001'
        functions = {'clip_evaluate.py': ('load_metric_functions', 'evaluate_depth', 'summarize_depth', 'depth_preflight'),
                     'clip_audit_depth.py': ('metric_functions', 'exact', 'tensor_sha'),
                     'clip_audit_detection.py': ('functions', 'prediction_valid', 'compare_rows')}
        for filename, names in functions.items():
            before = ast.parse((old_root / filename.replace('clip_', 'long_')).read_text())
            after = ast.parse((ROOT / filename).read_text())
            for name in names:
                old = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
                new = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
                self.assertEqual(ast.dump(old), ast.dump(new), filename + ':' + name)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
