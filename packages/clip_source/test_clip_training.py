"""Follow-up training CLI refuses accidental reuse of a six-group mode."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import clip_training as training


class TrainingEntryTests(unittest.TestCase):
    def arguments(self, output):
        return ["--child", "/child", "--r2-source", "/r2", "--v2-source", "/v2",
                "--reference-run", "/original", "--clip-plan", "/followup.json",
                "--clip-plan-sha256", "a" * 64, "--clip-group", "cap120",
                "--off-run", "/verified_off", "--output", str(output),
                "--expected-hostname", "synthetic", "--package-manifest-sha256", "b" * 64]

    def test_only_cap120_and_explicit_off_are_accepted(self):
        argv = self.arguments("/new_output")
        args = training.parse_args(argv)
        self.assertEqual(args.clip_group, "cap120")
        self.assertEqual(args.depth_weight, 1.0)
        for bad in ("off", "p1", "120"):
            changed = list(argv)
            changed[changed.index("cap120")] = bad
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(changed)
        missing = argv[:]
        index = missing.index("--off-run")
        del missing[index:index + 2]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            training.parse_args(missing)

    def test_off_mismatch_stops_entry_before_asset_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve() / "not_created"
            plan, policy = {"fixture": True}, {"group_id": "cap120", "max_norm": 120.0}
            with patch.object(training.socket, "gethostname", return_value="synthetic"), \
                 patch.object(training, "load_plan", return_value=(plan, policy)), \
                 patch.object(training, "require_completed_off", side_effect=ValueError("OFF mismatch")) as gate, \
                 patch.object(training, "verify_tree") as assets:
                with self.assertRaisesRegex(ValueError, "OFF mismatch"):
                    training.main(self.arguments(output))
                gate.assert_called_once_with(Path("/verified_off"), policy, "b" * 64,
                                             plan=plan, reference_run=Path("/original"))
                assets.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
