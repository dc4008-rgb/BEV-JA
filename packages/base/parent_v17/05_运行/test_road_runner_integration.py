"""Synthetic CPU runner integration, never real-weight/GPU efficacy evidence.

Only the approved train/validation manifest metadata and local frozen vehicle
assets are read. Scene backgrounds are generated, tiny detector weights random.
CUDA bookkeeping is mocked to exercise the real baseline loop on CPU.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

import torch

import run_road_patch_candidate as runner
from advrm_insertion import insert_car, parse_road_quad
from carriers import SceneContext
from road_patch_contract import load_patch
from test_road_detection_objectives import tiny_adapter


def synthetic_context(record, car, mask, device):
    if record.split not in ("train", "validation"):
        raise AssertionError("integration fixture cannot open test data")
    generator = torch.Generator().manual_seed(20260919)
    background = (0.2 + 0.6 * torch.rand(1, 3, 320, 1024, generator=generator)).to(device)
    quad = parse_road_quad(record.road_quad_norm)
    inserted = insert_car(background, car, mask, quad)
    return SceneContext(background, inserted.image, car, mask, inserted.mask,
                        inserted.geometry.box_xyxy, quad)


def synthetic_detector(model_id, device, manifest, weights_root):
    adapter = tiny_adapter(model_id)
    adapter.model.to(device).eval()
    backbone = adapter.model.backbone
    original = backbone.forward

    def with_context(value):
        features = original(value)
        # Explicit synthetic global context gives an off-object road patch a
        # differentiable route to target ROI logits. Not a pretrained network.
        context = value.mean(dim=(1, 2, 3), keepdim=True)
        return type(features)((key, feature + context) for key, feature in features.items())

    backbone.forward = with_context
    adapter.predict = lambda image: adapter.model([sample for sample in image])
    return adapter


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.records = runner.training_records(runner.ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_contextual_full_synthetic_loop(self):
        self.check_method("contextual")

    def test_robust_full_synthetic_loop(self):
        self.check_method("robust_dpatch")

    def check_method(self, method):
        with tempfile.TemporaryDirectory(prefix="road_v17_integration_") as temporary, ExitStack() as stack:
            output = Path(temporary)
            args = SimpleNamespace(method=method, mode="smoke", output=output,
                                   model_manifest=runner.ROOT / "02_模型/model_manifest.json", weights_root=None,
                                   training_manifest=runner.ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv")
            stack.enter_context(mock_patch.object(runner, "build_offline_detector", synthetic_detector))
            stack.enter_context(mock_patch.object(runner, "build_context", synthetic_context))
            stack.enter_context(mock_patch.object(torch.cuda, "reset_peak_memory_stats"))
            stack.enter_context(mock_patch.object(torch.cuda, "synchronize"))
            stack.enter_context(mock_patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            result = runner.optimize(args, self.records, torch.device("cpu"))
            self.assertEqual(result["updates"], 2)
            self.assertEqual(result["candidate_id"], runner.CANDIDATE_ID)
            self.assertEqual(result["seed"], 20260919)
            self.assertEqual(result["schedule_sha256"], runner.sha256(output / "schedule.json"))
            self.assertEqual(result["checkpoint_bindings"], {})
            self.assertEqual(result["domain_exposures"], {"kitti": 1, "nuscenes": 1})
            self.assertFalse(result["test_data_opened"])
            self.assertFalse(result["formal_test_allowed"])
            self.assertEqual(result["budget_counts"]["input_backward"], {"fasterrcnn": 4, "ssd300": 4})
            self.assertEqual(result["gradient_diagnostics"]["zero_model_update_counts"],
                             {"fasterrcnn": 0, "ssd300": 0})
            trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(len(trace), 2)
            for row in trace:
                self.assertGreater(row["gradient_l2"], 0)
                self.assertGreater(row["max_rgb_delta"], 0)
                self.assertEqual(len(row["model_draws"]), 4)
            final = load_patch(output / "final_patch.pt")
            initial = load_patch(output / "zero_step_patch.pt")
            self.assertEqual(tuple(final.shape), (1, 3, 64, 128))
            self.assertFalse(torch.equal(final, initial))
            for dataset in ("kitti", "nuscenes"):
                self.assertTrue((output / (dataset + "_clean_zero_random_final.png")).is_file())
                preview = torch.load(output / (dataset + "_preview.pt"), weights_only=True)
                left, top, right, bottom = map(int, preview["target_box"])
                torch.testing.assert_close(preview["clean"][..., top:bottom, left:right],
                                           preview["final"][..., top:bottom, left:right], rtol=0, atol=0)
            # Exercise actual evaluator I/O and state matching after saving the
            # development summary; AP may be zero with random-weight detectors.
            runner.write_json(output / "summary.json", result)
            eval_dir = output / "validation"
            eval_dir.mkdir()
            args.run_output = output
            args.output = eval_dir
            summary = runner.evaluate(args, self.records, torch.device("cpu"))
            self.assertEqual(summary["records"], 24)  # 4 scenes x 2 models x 3 patch states.
            self.assertFalse(summary["formal_test_allowed"])
            rows = json.loads((eval_dir / "per_sample.json").read_text())
            self.assertEqual({row["split"] for row in rows}, {"validation"})
            self.assertEqual(len(list(eval_dir.glob("*.png"))), 16)
            self.assertEqual(len(list(eval_dir.glob("*_predictions.pt"))), 4)
            prediction = torch.load(eval_dir / "fasterrcnn_kitti_predictions.pt", weights_only=True)
            self.assertTrue({"clean", "zero", "random", "final"}.issubset(prediction))
            self.assertEqual(prediction["split"], "validation")
            with (output / "random_control_patch.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "patch/control changed"):
                runner.evaluate(args, self.records, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
