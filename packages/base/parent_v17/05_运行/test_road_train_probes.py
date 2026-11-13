"""Train-probe contract tests use synthetic images/adapters, never real efficacy."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import evaluate_road_train_probes as probes
from carriers import SceneContext
from road_patch_contract import save_patch


class TinyDetector:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(.7))
        self.calls = 0

    def predict(self, image):
        if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            raise AssertionError("probe escaped inference-only execution")
        self.calls += 1
        score = (self.parameter - .1 * image.mean()).clamp(0, 1)
        return [{"boxes": torch.tensor([[400., 150., 500., 190.], [0., 0., 20., 20.],
                                        [400., 150., 480., 190.]]),
                 "scores": torch.stack((score, score * .9, score * .8)),
                 "labels": torch.tensor([3, 1, 8])}]


def synthetic_context(record):
    if record.split != "train":
        raise AssertionError("validation/test image factory was invoked")
    image = torch.full((1, 3, 320, 1024), .5)
    mask = torch.zeros((1, 1, 320, 1024))
    mask[..., 150:190, 400:500] = 1
    image[..., 150:190, 400:500] = .9
    return SceneContext(image.clone(), image, torch.full((1, 3, 40, 100), .9),
                        torch.ones((1, 1, 40, 100)), mask, (400, 150, 500, 190),
                        tuple(tuple(point) for point in json.loads(record.road_quad_norm)))


class TrainProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.manifest = probes.ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv"
        cls.models = probes.ROOT / "02_模型/model_manifest.json"
        # Only CSV metadata; no train, validation or test images opened here.
        cls.records = probes.training_records(cls.manifest)
        cls.calibration = probes.calibration_records(cls.records)
        cls.ids = [r.sample_id for r in cls.calibration]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="road_train_probe_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs" / "completed"
        self.run.mkdir(parents=True)
        save_patch(self.run / "zero_step_patch.pt", torch.zeros((1, 3, 64, 128)))
        for step in probes.CHECKPOINT_STEPS:
            save_patch(self.run / ("checkpoint_%04d.pt" % step),
                       torch.full((1, 3, 64, 128), step / 200.), {"updates": step})
        save_patch(self.run / "final_patch.pt", torch.full((1, 3, 64, 128), .5))
        self.summary = {
            "candidate_id": probes.CANDIDATE_ID, "status": "development_optimization_complete",
            "method": "contextual", "mode": "validation", "updates": 100,
            "train_models": list(probes.CORE), "test_data_opened": False,
            "calibration_sample_ids": self.ids,
            "training_manifest_sha256": probes.sha256(self.manifest),
            "model_manifest_sha256": probes.sha256(self.models),
            "patch_sha256": {name: probes.sha256(self.run / name)
                             for name in ("zero_step_patch.pt", "final_patch.pt")},
            "checkpoint_bindings": {"checkpoint_%04d.pt" % step:
                                    probes.sha256(self.run / ("checkpoint_%04d.pt" % step))
                                    for step in probes.CHECKPOINT_STEPS},
        }
        self.write_summary()

    def write_summary(self):
        probes.write_json(self.run / "summary.json", self.summary)

    def load(self):
        return probes.load_bound_run(self.run, self.manifest, self.models)

    def test_complete_baseline_and_both_ours_variants_bind(self):
        summary, states, _ = self.load()
        self.assertEqual(len(states), 5)
        self.assertEqual(probes.fixed_train_probes(self.records, summary["calibration_sample_ids"]), self.calibration)
        self.summary.update(status="train_optimization_complete", executed_updates=100,
                            patch_bindings=self.summary["patch_sha256"])
        for variant, models in (("det-only", probes.CORE),
                                ("dual", (*probes.CORE, "monodepth2", "dpt_large"))):
            with self.subTest(variant=variant):
                self.summary.update(candidate_id=probes.CANDIDATE_ID + "-ours-" + variant,
                                    task_variant=variant, training_models=list(models))
                self.write_summary()
                self.assertEqual(self.load()[0]["task_variant"], variant)

    def test_rejects_validation_test_and_calibration_order_drift(self):
        for split in ("validation", "test"):
            altered = (replace(self.calibration[0], split=split), *self.calibration[1:])
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train rows"):
                probes.require_train_probes(altered, self.ids)
        with self.assertRaisesRegex(ValueError, "calibration_sample_ids"):
            probes.fixed_train_probes(self.records, list(reversed(self.ids)))
        validation_id = next(r.sample_id for r in self.records if r.split == "validation")
        with self.assertRaisesRegex(ValueError, "calibration_sample_ids"):
            probes.fixed_train_probes(self.records, [validation_id, *self.ids[1:]])
        with self.assertRaisesRegex(ValueError, "test/unknown"):
            probes.fixed_train_probes([replace(self.records[0], split="test"), *self.records[1:]], self.ids)

    def test_checkpoint_byte_tampering_rejected_before_torch_load(self):
        with (self.run / "checkpoint_0050.pt").open("ab") as stream:
            stream.write(b"tampered")
        with patch.object(probes, "load_patch", side_effect=AssertionError("must hash before loading")):
            with self.assertRaisesRegex(ValueError, "hash binding failed: checkpoint_0050"):
                self.load()

    def test_completion_updates_and_all_four_checkpoints_required(self):
        for changes in ({"updates": 99}, {"status": "failed_retained"},
                        {"test_data_opened": True}, {"candidate_id": "normalized-v16-roadpatch-20260919"},
                        {"checkpoint_bindings": {}}):
            with self.subTest(changes=changes):
                original = self.summary.copy()
                self.summary.update(changes)
                self.write_summary()
                with self.assertRaises(ValueError):
                    self.load()
                self.summary = original
                self.write_summary()

    def test_manifest_binding_and_final100_equality_required(self):
        self.summary["training_manifest_sha256"] = "0" * 64
        self.write_summary()
        with self.assertRaisesRegex(ValueError, "input binding"):
            self.load()
        self.summary["training_manifest_sha256"] = probes.sha256(self.manifest)
        save_patch(self.run / "final_patch.pt", torch.full((1, 3, 64, 128), .9))
        self.summary["patch_sha256"]["final_patch.pt"] = probes.sha256(self.run / "final_patch.pt")
        self.write_summary()
        with self.assertRaisesRegex(ValueError, "final_patch must equal"):
            self.load()

    def test_matched_car_confidence_requires_car_iou_half_no_match_zero(self):
        rows = [{"label": 3, "score": .99, "iou": .4999},
                {"label": 8, "score": .98, "iou": 1.0},
                {"label": 3, "score": .4, "iou": .5}]
        self.assertEqual(probes.matched_car_confidence(rows), .4)
        self.assertEqual(probes.matched_car_confidence(rows[:2]), 0.)
        self.assertEqual(probes.matched_car_confidence([]), 0.)

    def test_synthetic_full_query_budget_raw_records_and_no_optimization(self):
        _, states, _ = self.load()
        copies = {name: value.clone() for name, value in states.items()}
        adapters = {model: TinyDetector() for model in probes.CORE}
        emitted = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.optim, "Adam", side_effect=AssertionError("optimization forbidden")))
            stack.enter_context(patch.object(torch.autograd, "grad", side_effect=AssertionError("backward forbidden")))
            rows, counts = probes.evaluate_probes(self.calibration, states, self.ids, torch.device("cpu"),
                                                  adapters.__getitem__, synthetic_context, emitted.append)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len(emitted), 40)
        self.assertEqual(counts, {"clean_model_forwards": 8, "patch_model_forwards": 40,
                                  "optimization_updates": 0, "backward_calls": 0, "total_model_forwards": 48})
        self.assertEqual({r["split"] for r in rows}, {"train"})
        self.assertEqual({r["sample_id"] for r in rows}, set(self.ids))
        for adapter in adapters.values():
            self.assertEqual(adapter.calls, 24)
            self.assertIsNone(adapter.parameter.grad)
            self.assertEqual(float(adapter.parameter), float(torch.tensor(.7)))
        for state in states:
            torch.testing.assert_close(states[state], copies[state], rtol=0, atol=0)
        for row in rows:
            self.assertEqual(len(row["clean_raw_detections"]["boxes"]), 3)
            self.assertEqual(len(row["adversarial_raw_detections"]["boxes"]), 3)
            self.assertEqual(len(row["clean_candidates"]), 2)
            self.assertGreater(row["matched_car_confidence_clean"], 0)
        summaries = probes.aggregate_probes(rows)
        self.assertEqual(len(summaries), 30)
        self.assertTrue(all(summary["scene_count"] in (2, 4) for summary in summaries))

    def test_cli_cuda_required_failure_retained_and_existing_output_rejected(self):
        output = self.root / "runs" / "probe"
        argv = ["--run-output", str(self.run), "--output", str(output),
                "--training-manifest", str(self.manifest), "--model-manifest", str(self.models),
                "--device", "cpu"]
        with patch.object(probes, "ROOT", self.root), patch.object(probes, "preflight", return_value={"inputs_ready": True}):
            with self.assertRaisesRegex(RuntimeError, "require CUDA"):
                probes.main(argv)
            failure_sha = probes.sha256(output / "failure.json")
            with self.assertRaises(FileExistsError):
                probes.main(argv)
            self.assertEqual(probes.sha256(output / "failure.json"), failure_sha)
        self.assertFalse((output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
