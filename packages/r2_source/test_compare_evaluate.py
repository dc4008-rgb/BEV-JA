import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from compare import (batch_summary, compare_training, compare_evaluations, expected_order, load_training,
                     read_json, sha, tensor_sha, tree_difference, write_json)
from evaluate import zero_to_final
from evidence import fingerprint


class ExactComparisonTest(unittest.TestCase):
    def test_dtype_and_sub_display_precision_are_significant(self):
        self.assertIsNotNone(tree_difference(torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64)))
        a = torch.tensor([1.], dtype=torch.float32)
        b = torch.nextafter(a, torch.tensor([2.]))
        value = tree_difference(a, b)
        self.assertEqual(value['reason'], 'tensor_bytes')
        self.assertEqual(value['first_different_flat_element'], 0)
        self.assertLess(value['max_abs_difference'], 1e-6)

    def test_bf16_original_bits_and_signed_zero(self):
        a = torch.tensor([0., -1., 2.], dtype=torch.bfloat16)
        self.assertIsNone(tree_difference(a, a.clone()))
        b = a.clone()
        b[0] = -0.0
        self.assertIsNotNone(tree_difference(a, b))
        self.assertNotEqual(tensor_sha(a), tensor_sha(b))

    def test_nan_is_never_exact_acceptance(self):
        a = torch.tensor([float('nan')])
        self.assertEqual(tree_difference(a, a)['reason'], 'nonfinite_tensor')

    def test_nested_first_difference_localizes(self):
        left = {'first': [torch.ones(2)], 'gradient': {'raw': torch.zeros(3)}}
        right = copy.deepcopy(left)
        right['gradient']['raw'][2] = 1e-9
        self.assertEqual(tree_difference(left, right)['path'], '$.gradient.raw')

    def make_run(self, root, changed_index=None):
        root.mkdir()
        states = root / 'state_evidence'
        states.mkdir()
        entries = []
        for index, (phase, step) in enumerate(expected_order(), 1):
            payload = {'schema': 'meeting-full-state-v1',
                       'metadata': {'candidate_id': 'new', 'phase': phase, 'global_step': step, 'resume_enabled': False},
                       'state': {'latent': torch.tensor([index], dtype=torch.bfloat16), 'raw_gradient': torch.tensor([1e-5]),
                                 'rng': torch.tensor([1, 2], dtype=torch.uint8)}}
            if index == changed_index:
                payload['state']['raw_gradient'][0] += 1e-8
            # Different container basename hashes are deliberately permitted.
            path = 'state_evidence/%04d_%s.pt' % (index, root.name)
            torch.save(payload, root / path)
            entries.append({'index': index, 'phase': phase, 'global_step': step, 'path': path,
                            'sha256': sha(root / path), 'bytes': (root / path).stat().st_size,
                            'scientific_fingerprint': fingerprint(payload)})
        write_json(states / 'MANIFEST.json', {'schema': 'meeting-full-state-manifest-v1', 'candidate_id': 'new',
                                            'status': 'complete', 'executed_updates': 100, 'entries': entries})
        write_json(root / 'runtime_provenance.json', {'pid': 100 if root.name == 'one' else 200,
                    'repeatability_identity': {'hostname': 'same', 'gpu_uuid': 'same', 'torch': '2.8.0'}})
        for name in ('zero_step_patch.pt', 'checkpoint_0100.pt', 'final_patch.pt'):
            torch.save({'schema': 'road_patch_rgb_v1', 'renderer_contract': 'same', 'canonical_rgb': torch.ones((1, 3, 2, 2)),
                        'metadata': {'run_path': str(root)}}, root / name)
        write_json(root / 'summary.json', {'status': 'train_optimization_complete', 'executed_updates': 100,
                    'test_data_opened': False, 'candidate_id': 'new', 'seed': 20260919,
                    'protocol_sha256': 'a' * 64, 'package_manifest_sha256': 'b' * 64,
                    'training_manifest_sha256': 'c' * 64, 'model_manifest_sha256': 'd' * 64,
                    'runtime_provenance_sha256': sha(root / 'runtime_provenance.json'),
                    'state_evidence_manifest_sha256': sha(states / 'MANIFEST.json')})

    def test_full409_exact_despite_different_archive_names_and_pids(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / 'one', Path(temporary) / 'two'
            self.make_run(a)
            self.make_run(b)
            self.assertNotEqual(sha(a / 'state_evidence/0001_one.pt'), sha(b / 'state_evidence/0001_two.pt'))
            value = compare_training(a, b)
            self.assertEqual(value['status'], 'exact')
            self.assertEqual(value['counts']['states_checked'], 409)
            self.assertEqual(batch_summary([a, b])['status'], 'partial_exact_pending_five_complete_runs')

    def test_first_backward_difference_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / 'one', Path(temporary) / 'two'
            self.make_run(a)
            self.make_run(b, changed_index=5)
            value = compare_training(a, b)
            self.assertEqual(value['status'], 'different')
            self.assertEqual(value['first_difference']['global_step'], 1)
            self.assertEqual(value['first_difference']['phase'], 'pre_update')
            self.assertEqual(value['first_difference']['path'], '$.state_payload.state.raw_gradient')

    def test_incomplete100_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'one'
            self.make_run(run)
            path = run / 'state_evidence/MANIFEST.json'
            data = read_json(path)
            data['entries'].pop()
            write_json(path, data)
            summary = read_json(run / 'summary.json')
            summary['state_evidence_manifest_sha256'] = sha(path)
            write_json(run / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, '409'):
                load_training(run)

    def test_evidence_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'one'
            self.make_run(run)
            (run / 'state_evidence/0001_one.pt').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                load_training(run)


class ZeroReferenceMetricTest(unittest.TestCase):
    def test_zero_reference_preserves_candidate_sets_negative_and_null(self):
        from evaluate import MODELS
        calls = []
        class Engine:
            @staticmethod
            def detection_summary(rows, count, seed, allow_zero_reference):
                calls.append((rows, count, seed, allow_zero_reference))
                return {'tnr_defined': bool(rows and rows[0]['dataset'] != 'nuscenes'),
                        'tnr_det': -0.25 if rows and rows[0]['dataset'] != 'nuscenes' else None}
        rows = []
        for model in MODELS:
            for dataset in ('kitti', 'nuscenes'):
                for state in ('zero', 'random', 'final'):
                    rows.append({'model': model, 'dataset': dataset, 'sample_id': dataset + '1', 'state': state,
                                 'clean_candidates': ['original_clean'], 'clean_car_candidates': ['original_clean_car'],
                                 'adversarial_candidates': [state], 'adversarial_car_candidates': [state + '_car']})
        result = zero_to_final(Engine, rows)
        self.assertEqual(len(result), 12)
        for subset, count, seed, allow in calls:
            self.assertEqual((count, seed, allow), (1000, 20260919, True))
            self.assertTrue(all(r['clean_candidates'] == ['zero'] and r['adversarial_candidates'] == ['final'] for r in subset))
        self.assertEqual(result[0]['tnr_det'], -0.25)
        self.assertIsNone(result[1]['tnr_det'])
        self.assertEqual(result[1]['tnr_undefined_reason'], 'reference_AP50_is_zero')


class RawEvaluationComparisonTest(unittest.TestCase):
    def make_evaluation(self, root, change=False):
        from evaluate import MODELS
        root.mkdir()
        (root / 'raw_predictions').mkdir()
        entries = []
        for model in MODELS:
            for index in range(40):
                dataset = 'kitti' if index < 20 else 'nuscenes'
                states = {state: {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([3]),
                                  'scores': torch.tensor([.8])} for state in ('clean', 'zero', 'random', 'final')}
                if change and model == MODELS[0] and index == 7:
                    states['final']['scores'][0] = torch.nextafter(states['final']['scores'][0], torch.tensor(1.))
                relative = 'raw_predictions/%s_%02d.pt' % (model, index)
                torch.save(states, root / relative)
                entries.append({'model': model, 'dataset': dataset, 'sample_id': str(index), 'path': relative,
                                'sha256': sha(root / relative),
                                'tensor_sha256': {state: {k: tensor_sha(v) for k, v in value.items()} for state, value in states.items()}})
        write_json(root / 'raw_predictions/MANIFEST.json', {'entries': entries, 'model_forward_count': 640})
        for name in ('per_sample.json', 'rendered_input_hashes.json', 'geometry_audit.json'):
            write_json(root / name, {'same': True})
        write_json(root / 'summary.json', {'status': 'development_validation_complete', 'updates': 100,
                    'evaluation_forward_total': 640, 'test_data_opened': False, 'protocol_sha256': 'a',
                    'package_manifest_sha256': 'b', 'evaluation_models': list(MODELS), 'control_tensor_sha256': {},
                    'parent_metric_engine_source_sha256': 'c', 'summaries': [], 'zero_to_final': [],
                    'seconds': 1 if root.name == 'one' else 2,
                    'artifact_bindings': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}})

    def test_all640_raw_outputs_exact_ignoring_elapsed_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / 'one', Path(temporary) / 'two'
            self.make_evaluation(a)
            self.make_evaluation(b)
            value = compare_evaluations(a, b)
            self.assertEqual(value['status'], 'exact')
            self.assertEqual(value['counts']['prediction_files_checked'], 160)

    def test_raw_output_difference_detected_even_if_summary_metric_same(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / 'one', Path(temporary) / 'two'
            self.make_evaluation(a)
            self.make_evaluation(b, change=True)
            value = compare_evaluations(a, b)
            self.assertEqual(value['status'], 'different')
            self.assertEqual(value['first_difference']['path'], '$.raw_predictions.fasterrcnn.7.final.scores')


if __name__ == '__main__':
    unittest.main()
