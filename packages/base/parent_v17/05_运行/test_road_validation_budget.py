"""Synthetic v17 budget and checkpoint tests; no model weights or scene files."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

import torch
import run_road_patch_candidate as runner
from road_patch_contract import load_patch


class ValidationBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.records = runner.training_records(runner.ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setup_loop(self, stack, zero_models=(), allow_validation=False):
        def context(record, car, mask, device):
            if record.split not in (("train", "validation") if allow_validation else ("train",)):
                raise AssertionError("optimization accessed a non-train record")
            image = torch.full((1, 3, 64, 128), .4, device=device)
            return SimpleNamespace(background=image, benign=image, road_quad=None, object_box=(1, 1, 2, 2))

        def objective(method, adapter, image, box, targets):
            coefficient = 0. if adapter.model_id in zero_models else 1.
            direction = 1. if method == "contextual" else -1.
            # Interior optimum avoids the deliberate clipping-boundary zero
            # gradient gate during a budget/checkpoint test.
            return coefficient * (direction * (image - .43).square().mean() + 1), {"synthetic": True}

        def adapter(model_id, device, manifest, weights_root):
            return SimpleNamespace(model_id=model_id, predict=lambda image: [{
                "boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}])

        stack.enter_context(mock_patch.object(runner, "build_context", context))
        stack.enter_context(mock_patch.object(runner, "build_offline_detector", adapter))
        stack.enter_context(mock_patch.object(runner, "load_car", return_value=(None, None)))
        stack.enter_context(mock_patch.object(runner, "extract_road_patch", side_effect=lambda image, quad: image))
        stack.enter_context(mock_patch.object(runner, "RoadRGBView", side_effect=lambda value:
                            SimpleNamespace(render=lambda context: SimpleNamespace(image=value))))
        stack.enter_context(mock_patch.object(runner, "objective", objective))
        stack.enter_context(mock_patch.object(runner, "print", create=True))
        stack.enter_context(mock_patch.object(torch.cuda, "reset_peak_memory_stats"))
        stack.enter_context(mock_patch.object(torch.cuda, "synchronize"))
        stack.enter_context(mock_patch.object(torch.cuda, "max_memory_allocated", return_value=0))

    def args(self, output, mode="validation", method="contextual"):
        return SimpleNamespace(method=method, mode=mode, output=output,
            model_manifest=runner.ROOT / "02_模型/model_manifest.json", weights_root=None,
            training_manifest=runner.ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv",
            source_prefix=None, source_root=None, device="cuda:0")

    def test_validation_100_checkpoints_queries_and_unchanged_pilot(self):
        for method in ("contextual", "robust_dpatch"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                self.setup_loop(stack, allow_validation=True)
                output = Path(temp) / "validation"
                output.mkdir()
                original_update = runner.patch_update
                snapshots = {}
                calls = [0]

                def update(patch, gradient, *args):
                    original_update(patch, gradient, *args)
                    calls[0] += 1
                    if calls[0] in (20, 25, 50, 75, 100):
                        snapshots[calls[0]] = patch.detach().clone()

                with mock_patch.object(runner, "patch_update", side_effect=update):
                    result = runner.optimize(self.args(output, method=method), self.records, torch.device("cpu"))
                self.assertEqual(result["updates"], 100)
                self.assertEqual(result["domain_exposures"], {"kitti": 50, "nuscenes": 50})
                self.assertEqual(result["budget_counts"]["patch_updates"], 100)
                self.assertEqual(result["budget_counts"]["input_backward"], dict.fromkeys(runner.CORE, 200))
                self.assertEqual(result["budget_counts"]["loss_forward"], dict.fromkeys(runner.CORE, 204))
                self.assertEqual(result["schedule_sha256"], runner.sha256(output / "schedule.json"))
                self.assertEqual(result["checkpoint_steps"], [25, 50, 75, 100])
                self.assertFalse(result["test_data_opened"])
                self.assertEqual(result["seed"], runner.SEED)
                schedule = json.loads((output / "schedule.json").read_text())
                self.assertEqual({row["split"] for row in schedule}, {"train"})
                expected = runner.seeded_round_robin(self.records, 100, runner.SEED)
                self.assertEqual([row["sample_id"] for row in schedule], [row.sample_id for row in expected])
                for step in (25, 50, 75, 100):
                    name = "checkpoint_%04d.pt" % step
                    self.assertEqual(result["checkpoint_bindings"][name], runner.sha256(output / name))
                    payload = torch.load(output / name, weights_only=True)
                    self.assertEqual(payload["metadata"]["updates"], step)
                    self.assertEqual(payload["metadata"]["candidate_id"], runner.CANDIDATE_ID)
                    self.assertEqual(payload["metadata"]["train_schedule_sha256"], result["schedule_sha256"])
                    torch.testing.assert_close(load_patch(output / name), snapshots[step], rtol=0, atol=0)
                    self.assertEqual(tuple(load_patch(output / name).shape), (1, 3, 64, 128))
                torch.testing.assert_close(load_patch(output / "checkpoint_0100.pt"), load_patch(output / "final_patch.pt"), rtol=0, atol=0)
                pilot_dir = Path(temp) / "pilot"
                pilot_dir.mkdir()
                pilot = runner.optimize(self.args(pilot_dir, mode="pilot", method=method), self.records, torch.device("cpu"))
                self.assertEqual(pilot["updates"], 20)
                self.assertEqual(pilot["domain_exposures"], {"kitti": 10, "nuscenes": 10})
                self.assertEqual(pilot["checkpoint_bindings"], {})
                self.assertEqual(json.loads((pilot_dir / "schedule.json").read_text()), schedule[:20])
                torch.testing.assert_close(load_patch(pilot_dir / "final_patch.pt"), snapshots[20], rtol=0, atol=0)
                runner.write_json(output / "summary.json", result)
                eval_dir = Path(temp) / "evaluated"
                eval_dir.mkdir()
                eval_args = self.args(eval_dir, method=method)
                eval_args.run_output = output
                evaluated = runner.evaluate(eval_args, self.records, torch.device("cpu"))
                self.assertEqual(evaluated["records"], 240)
                rows = json.loads((eval_dir / "per_sample.json").read_text())
                self.assertEqual({row["model"] for row in rows}, set(runner.CORE))
                self.assertEqual({row["split"] for row in rows}, {"validation"})
                self.assertEqual(len({row["sample_id"] for row in rows}), 40)

    def test_zero_model_draws_are_counted_and_smoke_still_rejects(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            self.setup_loop(stack, zero_models=("ssd300",))
            output = Path(temp) / "pilot"
            output.mkdir()
            result = runner.optimize(self.args(output, mode="pilot"), self.records, torch.device("cpu"))
            diagnostics = result["gradient_diagnostics"]
            self.assertEqual(diagnostics["zero_model_eot_gradients"]["ssd300"], {"0": 20, "1": 20})
            self.assertEqual(diagnostics["zero_model_eot_gradients"]["fasterrcnn"], {"0": 0, "1": 0})
            self.assertEqual(diagnostics["zero_model_update_counts"]["ssd300"], 20)
            self.assertEqual(diagnostics["zero_total_gradient_steps"], [])
            smoke_dir = Path(temp) / "smoke"
            smoke_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "smoke found no road-patch gradient"):
                runner.optimize(self.args(smoke_dir, mode="smoke"), self.records, torch.device("cpu"))
            self.assertTrue((smoke_dir / "zero_gradient_diagnostic.json").is_file())
            self.assertFalse((smoke_dir / "final_patch.pt").exists())

    def test_all_zero_failure_is_retained_without_overwrite(self):
        runs = runner.ROOT / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_failure_", dir=str(runs)) as temp, ExitStack() as stack:
            self.setup_loop(stack, zero_models=runner.CORE)
            output = Path(temp)
            args = self.args(output)
            original_optimize = runner.optimize
            stack.enter_context(mock_patch.object(runner, "arguments", return_value=args))
            stack.enter_context(mock_patch.object(runner, "preflight", return_value={"inputs_ready": True, "checks": []}))
            stack.enter_context(mock_patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock_patch.object(runner, "optimize", side_effect=lambda args, records, device:
                                original_optimize(args, records, torch.device("cpu"))))
            with self.assertRaisesRegex(RuntimeError, "empty or zero patch gradient"):
                runner.main()
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed_retained")
            self.assertEqual(failure["candidate_id"], runner.CANDIDATE_ID)
            self.assertFalse(failure["test_data_opened"])
            self.assertEqual(json.loads((output / "gradient_diagnostics.json").read_text())["zero_total_gradient_steps"], [1])
            self.assertTrue((output / "zero_step_patch.pt").exists())
            self.assertFalse((output / "final_patch.pt").exists())
            with self.assertRaises(FileExistsError):
                runner.main()

    def test_test_records_rejected_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temp, mock_patch.object(runner, "build_offline_detector") as detector:
            records = [replace(self.records[0], split="test")] + self.records[1:]
            with self.assertRaisesRegex(ValueError, "only train/validation"):
                runner.optimize(self.args(Path(temp)), records, torch.device("cpu"))
            detector.assert_not_called()

    def test_cuda_initialization_precedes_peak_reset(self):
        events = []
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            for name in ("set_device", "init", "reset_peak_memory_stats"):
                stack.enter_context(mock_patch.object(torch.cuda, name, side_effect=lambda *args, name=name: events.append(name)))
            stack.enter_context(mock_patch.object(torch.cuda, "manual_seed_all"))
            stack.enter_context(mock_patch.object(runner, "build_offline_detector", side_effect=RuntimeError("stop before actual model")))
            with self.assertRaisesRegex(RuntimeError, "stop before actual model"):
                runner.optimize(self.args(Path(temp)), self.records, torch.device("cuda:0"))
            self.assertEqual(events, ["set_device", "init", "reset_peak_memory_stats"])


if __name__ == "__main__":
    unittest.main()
