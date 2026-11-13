"""Synthetic full-loop integration gates; these do not validate real depth models.

Run with the frozen R2 helpers available, e.g.:
  python test_joint_training.py --r2-source /path/to/meeting_full_repeat_20260924_r2
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--r2-source", type=Path,
    default=ROOT.parent / "meeting_full_repeat_20260924_r2")
_args, _remaining = _parser.parse_known_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


joint = load_module("joint_training_test_target", ROOT / "joint_training.py")
R2_SOURCE, _ = joint.bind_r2_source(_args.r2_source)
fixtures = load_module("frozen_r2_synthetic_fixtures", R2_SOURCE / "test_training.py")
original = fixtures.runner
from adaptive_loss import foa_weights
from compare import tree_difference
from evidence import CallLedger, StateRecorder, rng_snapshot


class SyntheticDepth:
    def __init__(self):
        self.calls = 0

    def call(self, patch, context, views, ledger):
        self.calls += 1
        for _ in range(8):
            ledger.call("depth_clean_forwards", lambda: None)
            ledger.call("depth_adv_forwards", lambda: None)
        loss = .1 + 2e-5 * (patch - .7).square().mean()
        return loss, {"synthetic": True, "view_count": len(views),
                      "loss": float(loss.detach())}


class ForbiddenDepth:
    def call(self, *args, **kwargs):
        raise AssertionError("zero depth weight invoked the depth branch")


class JointLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PYTHONHASHSEED"] = str(original.SEED)
        original.configure_determinism()

    def run_once(self, output, runner, depth_weight=0.0, depth_objective=None):
        random.seed(original.SEED)
        np.random.seed(original.SEED)
        torch.manual_seed(original.SEED)
        output.mkdir()
        config = fixtures.Config()
        schedule = [SimpleNamespace(sample_id=f"train{i:03d}", split="train", dataset="synthetic")
                    for i in range(100)]
        context = SimpleNamespace(background=torch.zeros(1, 3, 16, 16),
            object_box=[7, 7, 8, 8], road_quad=[[7, 7], [9, 7], [9, 9], [7, 9]])
        budget = original.BUDGET if runner is original else joint.execution_budget(depth_weight)
        ledger = CallLedger(budget)
        recorder = StateRecorder(output, runner.CANDIDATE_ID)
        kwargs = {} if runner is original else {
            "depth_weight": depth_weight, "depth_objective": depth_objective}
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_optimization(parent=fixtures.Parent, config=config, output=output,
                pipeline=SimpleNamespace(scheduler=fixtures.Scheduler(), transformer=fixtures.Transformer()),
                prompt_pack=(torch.ones(1), torch.zeros(1), torch.ones(1), torch.zeros(1)),
                initial_patch=torch.ones(1, 3, 16, 16) * .1, schedule=schedule,
                context_for=lambda record: context, adapters={}, ledger=ledger, recorder=recorder,
                device=torch.device("cpu"), dtype=torch.bfloat16,
                eot_config=json.loads((R2_SOURCE / "eot_config.json").read_text()),
                loss_evaluator=fixtures.fake_losses, **kwargs)
        recorder.finish("complete", 100)
        return (*result, recorder, ledger)

    @staticmethod
    def payload(output, entry):
        return torch.load(output / entry["path"], map_location="cpu", weights_only=True)

    def assert_exact(self, left, right):
        self.assertIsNone(tree_difference(left, right))

    def test_zero_full_100_matches_frozen_r2_every_state_bit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.run_once(root / "original", original)
            new = self.run_once(root / "zero", joint, 0.0, ForbiddenDepth())
            self.assertEqual(old[4].completed, original.BUDGET)
            self.assertEqual(new[4].completed, original.BUDGET)
            self.assertEqual(len(old[3].entries), 409)
            self.assertEqual(len(new[3].entries), 409)
            self.assert_exact(old[0], new[0])
            for old_entry, new_entry in zip(old[3].entries, new[3].entries):
                self.assertEqual((old_entry["phase"], old_entry["global_step"]),
                                 (new_entry["phase"], new_entry["global_step"]))
                left = self.payload(root / "original", old_entry)
                right = self.payload(root / "zero", new_entry)
                self.assertEqual(left["metadata"]["candidate_id"], original.CANDIDATE_ID)
                self.assertEqual(right["metadata"]["candidate_id"], joint.CANDIDATE_ID)
                # This is the sole allowed change, explicitly checked above.
                left["metadata"]["candidate_id"] = right["metadata"]["candidate_id"]
                self.assert_exact(left, right)
            for old_row, new_row in zip(old[1], new[1]):
                # State-binding fingerprints include the declared candidate ID;
                # their complete payloads were checked above.
                self.assert_exact({k: v for k, v in old_row.items() if k != "state_bindings"},
                                  {k: v for k, v in new_row.items() if k != "state_bindings"})
            for name in old[2]:
                self.assert_exact(torch.load(root / "original" / name, weights_only=True),
                                  torch.load(root / "zero" / name, weights_only=True))

    def test_nonzero_depth_changes_shared_latent_and_preserves_two_detector_foa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero = self.run_once(root / "zero", joint, 0.0)
            depth = SyntheticDepth()
            active = self.run_once(root / "active", joint, 1.0, depth)
            self.assertEqual(depth.calls, 100)
            self.assertEqual(active[4].completed, joint.execution_budget(1.0))
            self.assertEqual(active[4].completed["autograd_calls"], 100)
            self.assertEqual(len(active[3].entries), 409)
            self.assertIsNotNone(tree_difference(zero[0], active[0]))
            previous = None
            depth_states = 0
            for entry in active[3].entries:
                state = self.payload(root / "active", entry)["state"]
                if entry["phase"] == "pre_backward":
                    depth_states += 1
                    self.assertEqual(tuple(state["raw_losses"].shape), (2,))
                    expected = state["detection_total_loss"] + state["depth_weight"] * state["depth_loss"]
                    self.assert_exact(expected, state["weighted_total_loss"])
                if entry["phase"] != "pre_update":
                    continue
                self.assert_exact(state["previous_raw_losses"], previous)
                weights, _ = foa_weights(state["raw_losses"], previous)
                self.assert_exact(weights, state["adaptive_weights"])
                previous = state["raw_losses"]
                z, gradient = state["latent_variable"], state["weighted_loss_gradient"]
                self.assertEqual(z.dtype, torch.bfloat16)
                self.assert_exact(z - (.05 * (100000.0 * gradient.to(z))), state["candidate_latent"])
                self.assertEqual(state["update_statistics"]["normalization"], "none")
                self.assertEqual(state["update_statistics"]["clamp"], "none")
            self.assertEqual(depth_states, 100)
            self.assertEqual(active[1][0]["adaptive_weights"], [1., 1.])
            for row in active[1]:
                self.assertEqual(row["training_models"], list(original.MODELS))
                self.assertEqual(row["training_depth_models"], list(joint.DEPTH_MODELS))
                self.assertEqual(row["depth_clean_forwards_this_step"], 8)
                self.assertEqual(row["depth_adv_forwards_this_step"], 8)

    def test_zero_never_imports_constructs_or_loads_depth_models(self):
        calls = []

        class Model:
            def eval(self):
                return self

            def requires_grad_(self, enabled):
                self.enabled = enabled
                return self

        def build(names, *args):
            calls.append(names)
            return {name: SimpleNamespace(model=Model()) for name in names}

        parent = SimpleNamespace(build_selected_adapters=build)
        with mock.patch.dict(sys.modules, {"depth_objective": None}):
            adapters, objective = joint.build_training_adapters(parent, "cpu", "manifest", None, 0.0)
        self.assertEqual(calls, [original.MODELS])
        self.assertEqual(tuple(adapters), original.MODELS)
        self.assertIsNone(objective)
        self.assertTrue(all(adapter.model.enabled is False for adapter in adapters.values()))

    def test_depth_weight_validation_and_explicit_call_budget(self):
        for invalid in (-1., float("nan"), float("inf"), True, "1"):
            with self.assertRaises((ValueError, TypeError)):
                joint.execution_budget(invalid)
        self.assertEqual(joint.execution_budget(0.0), original.BUDGET)
        self.assertEqual(joint.execution_budget(1.0), dict(original.BUDGET,
            depth_clean_forwards=800, depth_adv_forwards=800))

    def test_depth_loading_restores_all_rng_on_success_and_failure(self):
        def draw_from_all_rngs():
            random.random()
            np.random.random()
            torch.rand(7)
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    torch.rand(7, device=torch.device("cuda", index))

        class Model:
            def eval(self):
                return self

            def requires_grad_(self, enabled):
                return self

        for fail_construction in (False, True):
            with self.subTest(fail_construction=fail_construction):
                observed = {}

                def build(names, *args):
                    draw_from_all_rngs()
                    if names == original.MODELS:
                        # Detector initialization is deliberately not rolled back.
                        observed["after_detectors"] = rng_snapshot()
                    return {name: SimpleNamespace(model=Model()) for name in names}

                def construct(parent, adapters, beta):
                    self.assertEqual(tuple(adapters), joint.DEPTH_MODELS)
                    self.assertEqual(beta, .1)
                    draw_from_all_rngs()
                    if fail_construction:
                        raise RuntimeError("injected depth construction failure")
                    return "constructed-depth"

                parent = SimpleNamespace(build_selected_adapters=build)
                fake_module = SimpleNamespace(DepthObjective=construct)
                with mock.patch.dict(sys.modules, {"depth_objective": fake_module}):
                    if fail_construction:
                        with self.assertRaisesRegex(RuntimeError, "injected depth"):
                            joint.build_training_adapters(parent, "cpu", "manifest", None, 1.0)
                    else:
                        _, objective = joint.build_training_adapters(parent, "cpu", "manifest", None, 1.0)
                        self.assertEqual(objective, "constructed-depth")
                self.assert_exact(observed["after_detectors"], rng_snapshot())

    def test_asset_preflight_routes_depth_separately_and_enforces_readiness(self):
        calls = []

        def road_preflight(records, names, manifest, car, mask, weights_root):
            self.assertEqual(names, original.MODELS)
            calls.append("detectors")
            return {"inputs_ready": True, "checks": []}

        parent = SimpleNamespace(road_preflight=road_preflight)
        manifest = Path("synthetic-model-manifest.json")
        with mock.patch.dict(sys.modules, {"ours_road_v16": None}):
            _, depth_assets, ready = joint.preflight_training_assets(
                parent, [], manifest, "car", "mask", None, 0.0)
        self.assertIsNone(depth_assets)
        self.assertTrue(ready)
        self.assertEqual(calls, ["detectors"])
        for depth_ready in (True, False):
            with self.subTest(depth_ready=depth_ready):
                def depth_check(names, manifest_path):
                    self.assertEqual(names, joint.DEPTH_MODELS)
                    self.assertEqual(manifest_path, manifest)
                    calls.append("depth")
                    return {"inputs_ready": depth_ready, "checks": [], "applicable": True}

                with mock.patch.dict(sys.modules, {
                    "ours_road_v16": SimpleNamespace(depth_asset_checks=depth_check)}):
                    _, depth_assets, ready = joint.preflight_training_assets(
                        parent, [], manifest, "car", "mask", None, 1.0)
                self.assertEqual(ready, depth_ready)
                self.assertEqual(depth_assets["inputs_ready"], depth_ready)
                self.assertEqual(calls[-2:], ["detectors", "depth"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *_remaining])
