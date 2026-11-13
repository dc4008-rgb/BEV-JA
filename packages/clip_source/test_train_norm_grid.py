"""Synthetic CPU fixtures only: these are not experiment data or frozen plans."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import train_norm_grid as grid


def tensor_stats(norm):
    return {"dtype": "torch.bfloat16", "shape": [1, 16, 64, 64], "numel": 65536,
        "all_finite": True, "finite_count": 65536, "nan_count": 0,
        "positive_inf_count": 0, "negative_inf_count": 0,
        "finite_component_statistics": {"l2": norm, "rms": norm / 256,
            "mean": 0.0, "std_population": norm / 256}}


def synthetic_row(step, norm=None):
    norm = float(step) if norm is None else norm
    return {"global_step": step, "sample_id": "SYNTHETIC_TEST_ONLY",
        "split": "train", "depth_weight": 1.0,
        "training_models": ["fasterrcnn", "ssd300"],
        "training_depth_models": ["monodepth2", "dpt_large"],
        "training_tasks": ["detection", "depth"],
        "mathematical_objective": "R2_detector_FOA_plus_fixed_depth_semantic_target_far",
        "update_statistics": {"coefficient": 100000.0, "lr": .05,
            "normalization": "none", "clamp": "none", "inputs_modified": False,
            "numel": 65536, "shape": [1, 16, 64, 64]},
        "tensor_statistics": {key: tensor_stats(norm) for key in (
            "latent", "weighted_loss_gradient", "amplified_gradient", "proposed_update",
            "candidate_latent", "realized_delta_float32")}}


def synthetic_summary():
    return {"status": "train_optimization_complete", "identity": grid.IDENTITY,
        "candidate_id": grid.IDENTITY + "-two-detector-foa-plus-fixed-depth",
        "run_role": "fresh_joint500_training", "executed_updates": 500,
        "seed": grid.SEED, "depth_weight": 1.0, "latent_lr": .05,
        "gradient_multiplier": 100000.0, "normalizations": [], "update_clamp": None,
        "training_models": ["fasterrcnn", "ssd300"],
        "training_depth_models": ["monodepth2", "dpt_large"],
        "training_tasks": ["detection", "depth"], "depth_disabled_path": False,
        "training_manifest_sha256": grid.TRAINING_MANIFEST_SHA,
        "model_manifest_sha256": grid.MODEL_MANIFEST_SHA,
        "package_manifest_sha256": grid.PACKAGE_MANIFEST_SHA,
        "historical_source_manifest_sha256": grid.CHILD_MANIFEST_SHA,
        "fresh_start": True, "test_data_opened": False, "validation_data_opened": False,
        "checkpoint_selection": "fixed final500 only", "config": dict(grid.CONFIG),
        "schedule_sha256": "0" * 64, "state_evidence_manifest_sha256": "1" * 64,
        "attack_trace_sha256": "2" * 64}


class TrainGridTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="synthetic_clip_grid_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.summary_path = self.root / "summary.json"
        self.trace_path = self.root / "attack_trace.jsonl"
        self.rows = [synthetic_row(step) for step in range(1, 501)]
        self.summary = synthetic_summary()

    def write_inputs(self):
        trace = ("\n".join(json.dumps(row, allow_nan=False) for row in self.rows) + "\n").encode()
        self.trace_path.write_bytes(trace)
        self.summary["attack_trace_sha256"] = hashlib.sha256(trace).hexdigest()
        self.summary_path.write_text(json.dumps(self.summary, allow_nan=False))

    def build(self):
        self.write_inputs()
        return grid.build_plan(self.summary_path, self.trace_path)

    def test_complete_synthetic_trace_produces_fixed_grid(self):
        plan = self.build()
        self.assertEqual(plan["schema"], grid.SCHEMA)
        self.assertEqual(plan["status"], "thresholds_frozen_from_train")
        self.assertEqual([r["id"] for r in plan["groups"]], ["off", "p90", "p50", "p10", "p1", "zero"])
        self.assertIsNone(plan["groups"][0]["max_norm"])
        for row, expected in zip(plan["groups"][1:], [450.1, 250.5, 50.9, 5.99, 0.0]):
            self.assertAlmostEqual(row["max_norm"], expected, places=12)
        self.assertEqual([r["reference_counterfactual_exceed_count"] for r in plan["groups"]], [0, 50, 250, 450, 495, 500])
        self.assertEqual(plan["reference_gradient_scale"]["derived_l2_steps"], 0)
        self.assertEqual(plan["required_total_updates"], 3000)
        self.assertTrue(all(row["executed_updates"] == 500 for row in plan["groups"]))
        self.assertEqual(plan["reference"]["summary_sha256"], hashlib.sha256(self.summary_path.read_bytes()).hexdigest())
        self.assertFalse(plan["threshold_rule"]["validation_or_test_used"])

    def test_rejects_changed_trace_without_updating_summary_hash(self):
        self.write_inputs()
        with self.trace_path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            grid.build_plan(self.summary_path, self.trace_path)

    def test_rejects_nontrain_and_duplicate_steps(self):
        for key, value in (("split", "validation"), ("split", "test"), ("global_step", 1)):
            with self.subTest(key=key, value=value):
                before = self.rows[1][key]
                self.rows[1][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self.build()
                self.rows[1][key] = before

    def test_rejects_incomplete_or_extra_trace(self):
        full = copy.deepcopy(self.rows)
        for rows in (full[:499], full + [synthetic_row(501)]):
            self.rows = rows
            with self.assertRaisesRegex(ValueError, "exactly 500"):
                self.build()

    def test_rejects_wrong_reference_identity_or_training_binding(self):
        for key, value in (("identity", "old_det100"), ("executed_updates", 100),
            ("depth_weight", 0.0), ("seed", 9), ("normalizations", ["RMS"]),
            ("update_clamp", 1.0), ("gradient_multiplier", 20000.0),
            ("training_manifest_sha256", "f" * 64), ("validation_data_opened", True)):
            with self.subTest(key=key):
                before = self.summary[key]
                self.summary[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self.build()
                self.summary[key] = before

    def test_rejects_negative_norm(self):
        self.rows[0]["tensor_statistics"]["amplified_gradient"]["finite_component_statistics"]["l2"] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.build()

    def test_rejects_nonfinite_json_even_when_hash_bound(self):
        self.write_inputs()
        data = self.trace_path.read_bytes().replace(b'"l2": 1.0', b'"l2": NaN', 1)
        self.trace_path.write_bytes(data)
        self.summary["attack_trace_sha256"] = hashlib.sha256(data).hexdigest()
        self.summary_path.write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            grid.build_plan(self.summary_path, self.trace_path)

    def test_rejects_finite_component_subset(self):
        self.rows[0]["tensor_statistics"]["latent"]["finite_count"] = 65535
        with self.assertRaisesRegex(ValueError, "finite_count"):
            self.build()

    def test_rejects_shape_numel_mismatch(self):
        self.rows[0]["tensor_statistics"]["amplified_gradient"]["shape"] = [1, 16, 32, 32]
        with self.assertRaisesRegex(ValueError, "shape and numel"):
            self.build()

    def test_rejects_l2_rms_disagreement(self):
        self.rows[0]["tensor_statistics"]["amplified_gradient"]["finite_component_statistics"]["rms"] = 1.0
        with self.assertRaisesRegex(ValueError, "L2 and RMS"):
            self.build()

    def test_derives_missing_l2_and_marks_it(self):
        del self.rows[0]["tensor_statistics"]["amplified_gradient"]["finite_component_statistics"]["l2"]
        plan = self.build()
        self.assertEqual(plan["reference_gradient_scale"]["derived_l2_steps"], 1)
        self.assertEqual(plan["reference_gradient_scale"]["reported_l2_steps"], 499)
        self.assertEqual(plan["reference_gradient_scale"]["minimum_l2"], 1.0)

    def test_equal_quantiles_are_single_run_with_aliases(self):
        self.rows = [synthetic_row(step, 42.0) for step in range(1, 501)]
        plan = self.build()
        self.assertEqual(len(plan["groups"]), 3)
        self.assertEqual(plan["groups"][1]["aliases"], ["p90", "p50", "p10", "p1"])
        self.assertEqual(plan["groups"][1]["reference_counterfactual_exceed_fraction"], 0)
        self.assertEqual(plan["required_total_updates"], 1500)

    def test_zero_quantiles_merge_zero_alias_but_not_off(self):
        self.rows = [synthetic_row(step, 0.0) for step in range(1, 501)]
        plan = self.build()
        self.assertEqual(len(plan["groups"]), 2)
        self.assertIsNone(plan["groups"][0]["max_norm"])
        self.assertEqual(plan["groups"][1]["aliases"], ["p90", "p50", "p10", "p1", "zero"])
        self.assertEqual(plan["groups"][1]["max_norm"], 0.0)

    def test_refuses_overwrite_and_preserves_bytes(self):
        plan = self.build()
        output = self.root / "SYNTHETIC_TEST_PLAN.json"
        digest = grid.write_plan(plan, output)
        original = output.read_bytes()
        self.assertEqual(digest, hashlib.sha256(original).hexdigest())
        with self.assertRaises(FileExistsError):
            grid.write_plan(plan, output)
        self.assertEqual(output.read_bytes(), original)

    def test_refuses_symlink_output(self):
        target = self.root / "target.json"
        target.write_text("keep")
        linked = self.root / "linked.json"
        linked.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            grid.write_plan(self.build(), linked)
        self.assertEqual(target.read_text(), "keep")

    def test_duplicate_json_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            grid._load_json('{"l2": 1, "l2": 2}')

    def test_boolean_flags_are_not_integers(self):
        self.rows[0]["tensor_statistics"]["amplified_gradient"]["all_finite"] = 1
        with self.assertRaisesRegex(ValueError, "all_finite"):
            self.build()


if __name__ == "__main__":
    unittest.main()
