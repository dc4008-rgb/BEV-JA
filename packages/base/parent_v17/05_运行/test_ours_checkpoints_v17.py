"""Verify real canonical checkpoint serialization and immutable observation steps."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

spec = importlib.util.spec_from_file_location("ours_checkpoint_runner", Path(__file__).with_name("run_ours_stateful_validation.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from road_patch_contract import load_patch, canonicalize_rgb


class CheckpointTests(unittest.TestCase):
    def test_canonical_round_trip_and_step_identity(self):
        source = torch.linspace(0, 1, 3 * 128 * 256).reshape(1, 3, 128, 256)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (25, 50, 75, 100):
                name, digest = runner.save_training_checkpoint(root, source, step)
                self.assertEqual(digest, runner.sha256(root / name))
                self.assertTrue(torch.equal(load_patch(root / name), canonicalize_rgb(source)))
                payload = torch.load(root / name, map_location="cpu", weights_only=True)
                self.assertEqual(payload["metadata"]["updates"], step)

    def test_overwrite_and_unregistered_step_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = torch.zeros(1, 3, 64, 128)
            runner.save_training_checkpoint(root, source, 25)
            with self.assertRaises(FileExistsError):
                runner.save_training_checkpoint(root, source + 1, 25)
            with self.assertRaises(ValueError):
                runner.save_training_checkpoint(root, source, 26)


if __name__ == "__main__":
    unittest.main()
