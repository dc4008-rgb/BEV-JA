"""Metadata-only provenance, fixed learning-rate/scope gates; outer batch owns execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import inputs


class InputContractTests(unittest.TestCase):
    def test_real_local_provenance_and_train_schedule(self):
        result = inputs.verify_inputs(require_sealed=False)
        self.assertEqual(result['parent_files_verified'], 64)
        self.assertFalse(result['source_images_opened'])
        self.assertFalse(result['models_loaded'])
        self.assertEqual(result['latent_lr'], inputs.LEARNING_RATE)

    def test_only_u1_package_bound_lr_and_four_eval_models(self):
        protocol = inputs.load_protocol()
        self.assertEqual(protocol['update_variants'], ['U1'])
        self.assertEqual(protocol['objectives'], ['U1'])
        self.assertEqual(protocol['stateful_sd35']['latent_lr'], inputs.LEARNING_RATE)
        self.assertEqual(protocol['latent_lr'], inputs.LEARNING_RATE)
        self.assertEqual(protocol['learning_rate_label'], inputs.VARIANT_LABEL)
        self.assertEqual(protocol['training_models'], ['fasterrcnn', 'ssd300'])
        self.assertEqual(protocol['evaluation']['models'], ['fasterrcnn', 'ssd300', 'maskrcnn', 'fcos'])
        self.assertEqual(protocol['evaluation']['model_forwards_per_run'], 640)
        self.assertEqual(protocol['evaluation']['rows_per_run'], 480)
        self.assertEqual(protocol['evaluation']['summary_rows_per_run'], 48)
        self.assertEqual(protocol['stage_order'], ['train_U1', 'validation_U1'])
        self.assertEqual(protocol['stateful_sd35']['guidance_scale'], 4.5)
        self.assertTrue(protocol['batch_training_before_any_validation'])
        self.assertEqual(protocol['same_batch_new_control'], inputs.LEARNING_RATE == .05)

    def test_wrong_variant_lr_budget_split_resume_rejected(self):
        original = inputs.protocol_contract()
        cases = []
        for key, value in [('test_access_allowed', True), ('resume_enabled', True), ('dual_allowed', True),
                           ('update_variants', ['U0', 'U1']), ('latent_lr', .3), ('learning_rate_label', 'other'),
                           ('mathematical_objective', 'C3'), ('training_target_classes', [3]),
                           ('clip_threshold', .1), ('threshold_reused_without_refit', False)]:
            changed = copy.deepcopy(original)
            changed[key] = value
            cases.append(changed)
        for value in [True, 100.0, 400]:
            changed = copy.deepcopy(original)
            changed['train_schedule']['updates'] = value
            cases.append(changed)
        for value in [True, .3, *[x for x in (.05, .1, .2) if x != inputs.LEARNING_RATE]]:
            changed = copy.deepcopy(original)
            changed['stateful_sd35']['latent_lr'] = value
            cases.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            for changed in cases:
                path.write_text(json.dumps(changed))
                with patch.object(inputs, 'PROTOCOL_PATH', path), self.assertRaises(ValueError):
                    inputs.load_protocol()

    def test_paths_refuse_escape_and_runs_root(self):
        for path in [inputs.PARENT / 'runs/x', inputs.ROOT / 'runs', inputs.ROOT / 'runs/../../escape']:
            with self.assertRaises(ValueError):
                inputs.candidate_run_path(path)

    def test_entry_hash_duplicate_symlink_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').write_text('fixed')
            entry = {'path': 'a', 'sha256': inputs.sha(root / 'a'), 'bytes': 5}
            self.assertEqual(inputs._verify_entries(root, [entry]), {'a'})
            with self.assertRaises(ValueError):
                inputs._verify_entries(root, [entry, entry])
            (root / 'link').symlink_to(root / 'a')
            for name in ('link', '../a'):
                with self.assertRaises(ValueError):
                    inputs._verify_entries(root, [dict(entry, path=name)])
            (root / 'a').write_text('other')
            with self.assertRaises(ValueError):
                inputs._verify_entries(root, [entry])

    def test_child_matrix_rejects_standalone_execution(self):
        result = subprocess.run([sys.executable, '-B', str(inputs.ROOT / 'training/run_matrix.py'), '--run'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('outer run_batch.py', result.stderr)
        self.assertIn('U0 execution is forbidden', result.stderr)


if __name__ == '__main__':
    unittest.main()
