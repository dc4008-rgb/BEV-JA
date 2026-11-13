"""CPU synthetic evidence gates; no pretrained assets or real test data accessed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("evaluate_road_comparison.py")
spec = importlib.util.spec_from_file_location("road_comparison_under_test", PATH)
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)
import torch


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / "train_validation.csv"
        self.models = self.root / "models.json"
        self.train.write_text("synthetic manifest\n")
        self.models.write_text("synthetic models\n")
        self.records = [SimpleNamespace(sample_id="%s_%s_%03d" % (domain, split, i),
                                       dataset=domain, split=split)
                        for domain in ("kitti", "nuscenes")
                        for split, count in (("train", 100), ("validation", 20))
                        for i in range(count)]
        self.zero = torch.full((1, 3, 64, 128), .4)
        self.final = torch.full_like(self.zero, .6)

    def fixture(self, method):
        run = self.root / "runs" / method
        run.mkdir(parents=True)
        comparison.save_patch(run / "zero_step_patch.pt", self.zero)
        comparison.save_patch(run / "final_patch.pt", self.final)
        cp = {}
        for name, step in comparison.CHECKPOINTS.items():
            comparison.save_patch(run / name, self.final, {"updates": step})
            cp[name] = comparison.sha256(run / name)
        summary = {"candidate_id": comparison.CANDIDATE_ID, "test_data_opened": False,
                   "checkpoint_bindings": cp,
                   "training_manifest_sha256": comparison.sha256(self.train),
                   "model_manifest_sha256": comparison.sha256(self.models)}
        if method in ("contextual", "robust_dpatch"):
            comparison.save_patch(run / "random_control_patch.pt", comparison.common_random_patch())
            bound = {name: comparison.sha256(run / name) for name in
                     ("zero_step_patch.pt", "final_patch.pt", "random_control_patch.pt")}
            summary.update(method=method, status="development_optimization_complete", mode="validation",
                           updates=100, train_models=list(comparison.CORE), patch_sha256=bound,
                           final_patch_sha256=bound["final_patch.pt"])
            trace_file, field = "trace.jsonl", "step"
        else:
            variant = method[len("ours_"):]
            summary.update(candidate_id=comparison.CANDIDATE_ID + "-ours-" + variant,
                           task_variant=variant, status="train_optimization_complete", executed_updates=100,
                           training_models=list(comparison.CORE) + ([] if variant == "det-only" else ["monodepth2", "dpt_large"]),
                           patch_bindings={name: comparison.sha256(run / name) for name in ("zero_step_patch.pt", "final_patch.pt")})
            trace_file, field = "attack_trace.jsonl", "global_step"
        trace = [{field: i, "sample_id": r.sample_id, "dataset": r.dataset, "split": "train"}
                 for i, r in enumerate(comparison.seeded_round_robin(self.records, 100, comparison.SEED), 1)]
        (run / trace_file).write_text("".join(json.dumps(row) + "\n" for row in trace))
        comparison.write_json(run / "summary.json", summary)
        return run, summary

    def check(self, run):
        return comparison.validate_completed_run(run, self.records, self.train, self.models, self.root)

    def test_all_four_completed_identities_and_common_controls(self):
        rng = torch.get_rng_state().clone()
        zeros, randoms = [], []
        for method in ("contextual", "robust_dpatch", "ours_det-only", "ours_dual"):
            run, _ = self.fixture(method)
            checked = self.check(run)
            self.assertEqual(checked["method"], method)
            zeros.append(checked["patches"]["zero"])
            randoms.append(checked["patches"]["random"])
        self.assertTrue(all(torch.equal(zeros[0], value) for value in zeros))
        self.assertTrue(all(torch.equal(randoms[0], value) for value in randoms))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_reject_incomplete_updates_and_actual_trace(self):
        run, summary = self.fixture("contextual")
        summary["updates"] = 99
        comparison.write_json(run / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "validation100"):
            self.check(run)
        summary["updates"] = 100
        comparison.write_json(run / "summary.json", summary)
        lines = (run / "trace.jsonl").read_text().splitlines()
        (run / "trace.jsonl").write_text("\n".join(lines[:-1]))
        with self.assertRaisesRegex(ValueError, "exactly100"):
            self.check(run)

    def test_reject_altered_artifact_and_missing_checkpoint(self):
        run, summary = self.fixture("contextual")
        with (run / "final_patch.pt").open("ab") as stream:
            stream.write(b"altered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.check(run)
        run2, summary2 = self.fixture("ours_det-only")
        del summary2["checkpoint_bindings"]["checkpoint_0050.pt"]
        comparison.write_json(run2 / "summary.json", summary2)
        with self.assertRaisesRegex(ValueError, "bindings"):
            self.check(run2)

    def test_reject_checkpoint_step_metadata_and_final_mismatch(self):
        run, summary = self.fixture("ours_det-only")
        comparison.save_patch(run / "checkpoint_0025.pt", self.final, {"updates": 50})
        summary["checkpoint_bindings"]["checkpoint_0025.pt"] = comparison.sha256(run / "checkpoint_0025.pt")
        comparison.write_json(run / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.check(run)
        comparison.save_patch(run / "checkpoint_0025.pt", self.final, {"updates": 25})
        comparison.save_patch(run / "checkpoint_0100.pt", self.zero, {"updates": 100})
        for name in ("checkpoint_0025.pt", "checkpoint_0100.pt"):
            summary["checkpoint_bindings"][name] = comparison.sha256(run / name)
        comparison.write_json(run / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "differs from fixed checkpoint100"):
            self.check(run)

    def test_reject_wrong_split_schedule_manifest_and_root(self):
        run, summary = self.fixture("ours_dual")
        self.records[0].split = "test"
        with self.assertRaisesRegex(ValueError, "train/validation"):
            self.check(run)
        self.records[0].split = "train"
        rows = (run / "attack_trace.jsonl").read_text().splitlines()
        row = json.loads(rows[0]); row["split"] = "validation"
        rows[0] = json.dumps(row)
        (run / "attack_trace.jsonl").write_text("\n".join(rows))
        with self.assertRaisesRegex(ValueError, "train-only schedule"):
            self.check(run)
        self.train.write_text("changed manifest")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.check(run)
        with self.assertRaisesRegex(ValueError, "inside this candidate"):
            comparison.candidate_run_path(self.root / "foreign", self.root)

    def test_reject_hash_bound_but_noncommon_random(self):
        run, summary = self.fixture("robust_dpatch")
        comparison.save_patch(run / "random_control_patch.pt", self.zero)
        summary["patch_sha256"]["random_control_patch.pt"] = comparison.sha256(run / "random_control_patch.pt")
        comparison.write_json(run / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "random control differs"):
            self.check(run)

    def test_mock_full240_records_four_methods_controls_match_and_geometry_covered(self):
        # Small synthetic render/adapter only. Production CLI rejects CPU, and no
        # preflight or pretrained-loading path is replaced in the actual runner.
        context = SimpleNamespace(background=torch.full((1, 3, 12, 20), .2),
                                  benign=torch.full((1, 3, 12, 20), .2),
                                  object_mask=torch.zeros((1, 1, 12, 20)),
                                  object_box=(5, 2, 9, 5), road_quad=((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
        context.object_mask[..., 2:5, 5:9] = 1
        class Render:
            def __init__(self, rgb):
                self.rgb = rgb
            def render(self, ctx):
                support = torch.zeros_like(ctx.object_mask)
                support[..., 8:11, 6:14] = 1
                image = torch.where(support.bool().expand_as(ctx.benign), self.rgb.mean(), ctx.benign)
                return SimpleNamespace(image=image, support=support)
        class Adapter:
            def predict(self, image):
                return [{"boxes": torch.tensor([[5., 2., 9., 5.]]), "labels": torch.tensor([3]),
                         "scores": torch.tensor([float(image.mean())])}]
        all_controls = []
        original_summary = comparison.detection_summary
        with patch.object(comparison, "load_car", return_value=(None, None)), \
             patch.object(comparison, "build_context", return_value=context), \
             patch.object(comparison, "extract_road_patch", return_value=self.zero), \
             patch.object(comparison, "RoadRGBView", Render), \
             patch.object(comparison, "build_offline_detector", return_value=Adapter()), \
             patch.object(comparison, "detection_summary", side_effect=lambda rows, samples, seed, **kw: original_summary(rows, 2, seed, **kw)):
            for method in ("contextual", "robust_dpatch", "ours_det-only", "ours_dual"):
                run, _ = self.fixture(method)
                checked = self.check(run)
                output = self.root / "runs" / (method + "_evaluation")
                output.mkdir()
                args = SimpleNamespace(car=Path("unused"), car_mask=Path("unused"), output=output,
                                       model_manifest=self.models, weights_root=None)
                result = comparison.evaluate_comparison(args, self.records, checked, torch.device("cpu"))
                self.assertEqual(result["records"], 240)
                self.assertEqual(len(result["summaries"]), 18)
                audit = json.loads((output / "geometry_audit.json").read_text())
                self.assertEqual(audit["unique_scenes"], 40)
                self.assertEqual(audit["checked_model_scene_states"], 240)
                self.assertTrue(all(all(row["checks"].values()) for row in audit["checks"]))
                self.assertEqual(len(list(output.glob("*.png"))), 16)
                self.assertEqual(len(list(output.glob("*_inputs.pt"))), 2)
                self.assertEqual(len(list(output.glob("*_predictions.pt"))), 4)
                records = json.loads((output / "per_sample.json").read_text())
                controls = [{k: v for k, v in row.items() if k != "method"}
                            for row in records if row["state"] in ("zero", "random")]
                all_controls.append(controls)
                # Evaluator must never create a random artifact in an ours run.
                if method.startswith("ours"):
                    self.assertFalse((run / "random_control_patch.pt").exists())
            self.assertTrue(all(all_controls[0] == values for values in all_controls))


if __name__ == "__main__":
    unittest.main()
