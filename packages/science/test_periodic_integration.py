"""CPU toy victims exercise the real periodic runner, not attack effectiveness.

All executions use the allowed single-outer 25-step diagnostic. Frozen helper
code is read-only. No pretrained weights, image datasets, GPU, or old outputs
are opened; only temporary test artifacts are written.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
for path in (ROOT.parent / "normalized-v29-c0-u1-lr-20260922/normalized-v29-c0-u1-lr005-20260922/parent_v17/03_方法/production",
             ROOT.parent / "r2_joint_depth_20261001_v2",
             ROOT.parent / "meeting_full_repeat_20260924_r2",
             ROOT.parent / "r2_joint_clip120_followup_20261004", ROOT):
    sys.path.insert(0, str(path))

import numpy as np
import torch
import torch.nn.functional as F

import batch_training
from depth_hinge_objective import DepthObjective, DEPTH_SEMANTICS
from evidence import CallLedger, fingerprint, rng_snapshot
from adaptive_loss import weighted_loss
from clip_update import clipped_scaled_update
from geometry_eot import sample_views
import sample_contract
import sample_evaluate
import sample_training as training


@dataclass
class FakeConfig:
    num_inference_steps: int = 100
    strength: float = .25
    latent_lr: float = .05
    step_feedback_mode: str = "latent_chain"
    guidance_scale: float = 4.5
    seed: int = 20260919
    prompt: str = "top-down aerial view of clean asphalt road surface"
    negative_prompt: str = "vehicle, car, obstacle"
    outer_rounds: int = 1

    @property
    def active_steps_per_round(self):
        return 25

    @property
    def adversarial_updates(self):
        return 25 * self.outer_rounds

    def to_dict(self):
        return dict(vars(self))


class FakeScheduler:
    def __init__(self, config=None):
        self.config = dict(config or {"schedule": "periodic_cpu_fixture"})
        self.step_index = None
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, count, device):
        self.timesteps = torch.arange(count - 1, -1, -1, device=device)
        self.sigmas = torch.linspace(1, 0, count + 1, device=device)
        self.step_index = 0


class FakePipeline:
    def __init__(self):
        self.scheduler = FakeScheduler()
        self.denoise_inputs = []


class FakeParent:
    def __init__(self):
        self.encode_inputs = []

    def encode_patch(self, pipeline, image, dtype):
        self.encode_inputs.append(fingerprint(image))
        return F.adaptive_avg_pool2d(image.mean(1, keepdim=True), (4, 4)).expand(
            -1, 16, -1, -1).to(dtype).clone()

    def decode_patch(self, pipeline, latent):
        # Different RGB latent channels allow non-collinear task gradients;
        # a scalar decoder would force a cancellation-only conflict fixture.
        return torch.sigmoid(.001 * latent[:, :3]).mean(
            (-2, -1), keepdim=True).expand(1, 3, 512, 512)

    def save_patch(self, path, image, metadata, raw_generator_rgb=None):
        canonical = image.float() if image.shape[-2:] == (64, 128) else F.interpolate(
            image.float(), (64, 128), mode="bilinear", align_corners=False)
        value = {"canonical_rgb": canonical.detach().clone(), "metadata": metadata}
        if raw_generator_rgb is not None:
            value["raw_generator_rgb"] = raw_generator_rgb.detach().clone()
        with Path(path).open("xb") as stream:
            torch.save(value, stream)

    class RoadRGBView:
        def __init__(self, image):
            self.image = image

        def render(self, context):
            rendered = context.benign + F.interpolate(
                self.image, (16, 16), mode="bilinear", align_corners=False)
            support = torch.zeros_like(context.object_mask)
            support[..., 11:14, 6:10] = 1.0
            return SimpleNamespace(image=rendered, support=support)


class FakeDepthModel(torch.nn.Module):
    def __init__(self, semantic, gain):
        super().__init__()
        self.training_output_semantics = semantic
        self.gain = torch.nn.Parameter(torch.tensor(gain))
        self.calls = 0

    def predict_for_training(self, image):
        self.calls += 1
        signal = image[:, :1] * self.gain
        sign = 1 if self.training_output_semantics == "relative_depth" else -1
        return torch.sinh(torch.asinh(torch.ones_like(signal)) + sign * signal)


class FakeDetector:
    def __init__(self, score):
        self.score = score
        self.clean_calls = 0

    def predict(self, image):
        if torch.is_grad_enabled():
            raise AssertionError("clean reference must have no graph")
        self.clean_calls += 1
        return [{"boxes": torch.tensor([[6., 5., 10., 9.]]),
                 "labels": torch.tensor([3]), "scores": torch.tensor([self.score])}]


def obsolete_foa(*args, **kwargs):
    raise AssertionError("old per-step FOA callback must not be reached")


def fake_denoise(pipeline, latent, timestep, prompt_pack, guidance_scale):
    pipeline.denoise_inputs.append(fingerprint(latent))
    pipeline.scheduler.step_index += 1
    with torch.no_grad():
        result = .99 * latent + float(timestep) * .00001
    return result, {"step_index": pipeline.scheduler.step_index}


def fake_detector_losses(parent, image, context, views, adapters, ledger):
    for _ in range(8):
        ledger.call("detector_forwards", lambda: None)
    losses = [(.1 + image.mean()).square(), (.2 + .3 * image.mean()).square()]
    models = {}
    for name, scale in zip(training.DETECTOR_MODELS, (.9, .65)):
        score = scale * (1.0 - image.mean()).detach()
        models[name] = {"boxes": torch.tensor([[6., 5., 10., 9.]]),
                        "labels": torch.tensor([3]), "scores_all": score.reshape(1)}
    return losses, {"views": [{"view_index": i,
        "target_box": [6., 5., 10., 9.], "models": models} for i in range(4)]}


def sample(name):
    return {"sample_id": name, "case_id": name, "dataset": "kitti",
            "source_split": "train", "source_sha256": "a" * 64,
            "source_path": "/not-opened/" + name}


def context(value=.3):
    return SimpleNamespace(object_box=(6, 5, 10, 9),
        benign=torch.zeros(1, 3, 16, 16), object_mask=torch.ones(1, 1, 16, 16),
        crop_value=value)


def initialization(scene):
    small = torch.full((1, 3, 64, 128), scene.crop_value)
    return small, F.interpolate(small, (512, 512), mode="bilinear", align_corners=False)


def fixture(science, pipeline=None):
    parent, pipeline = FakeParent(), pipeline or FakePipeline()
    detectors = {name: FakeDetector(score) for name, score in zip(
        training.DETECTOR_MODELS, (.9, .65))}
    models = {name: FakeDepthModel(DEPTH_SEMANTICS[name], gain)
              for name, gain in zip(training.DEPTH_MODELS, (.05, .4))}
    objective = DepthObjective(parent, models, beta=science["depth_beta"],
        depth_target=science["switches"]["depth_target"],
        epsilon=science["controller_config"]["epsilon"])
    kernels = training.TrainingKernels(fake_denoise, fake_detector_losses,
        obsolete_foa, weighted_loss, clipped_scaled_update, sample_views,
        fingerprint, rng_snapshot, lambda scheduler: {
            "config": scheduler.config, "step_index": scheduler.step_index,
            "timesteps": scheduler.timesteps, "sigmas": scheduler.sigmas}, CallLedger)
    return parent, pipeline, detectors, models, objective, kernels


def run_case(output, science, *, pipeline=None, name="case", value=.3):
    output.mkdir()
    parent, pipeline, detectors, models, objective, kernels = fixture(science, pipeline)
    scene = context(value)
    small, raw = initialization(scene)
    calls = []
    original_grad = torch.autograd.grad

    def tracked_grad(*args, **kwargs):
        calls.append((kwargs["retain_graph"], kwargs["create_graph"]))
        return original_grad(*args, **kwargs)

    with contextlib.redirect_stdout(io.StringIO()), patch.object(
            torch.autograd, "grad", tracked_grad):
        result = training.run_sample_optimization(parent=parent, config=FakeConfig(),
            output=output, pipeline=pipeline, prompt_pack=None, initial_patch=raw,
            initial_small=small, sample=sample(name), context=scene, adapters=detectors,
            depth_objective=objective, device=torch.device("cpu"), dtype=torch.float32,
            eot_config=science["eot"], kernels=kernels, plan_sha256="b" * 64,
            source_bindings={"fixture": "c" * 64}, scientific_config=science)
    rows = [json.loads(row) for row in (output / "attack_trace.jsonl").read_text().splitlines()]
    return SimpleNamespace(summary=result, rows=rows, parent=parent, pipeline=pipeline,
        detectors=detectors, models=models, objective=objective, calls=calls, output=output)


class PeriodicIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.groups = {name: run_case(cls.root / name,
            sample_contract.scientific_config_for(name)) for name in sample_contract.VARIANTS}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_six_groups_exact_calls_and_two_vjps(self):
        expected = {"detector_forwards": 202, "autograd_calls": 50,
            "transformer_forward_calls": 25, "vae_encode_calls": 2,
            "vae_decode_calls": 52, "noise_draws": 1,
            "depth_clean_forwards": 200, "depth_adv_forwards": 200}
        self.assertEqual(training.planned_budget(25), expected)
        self.assertEqual(sample_contract.planned_budget(25), expected)
        for name, result in self.groups.items():
            with self.subTest(variant=name):
                for section in ("planned", "attempted", "completed"):
                    self.assertEqual(result.summary["budget"][section], expected)
                self.assertEqual(result.calls, [(True, False), (False, False)] * 25)
                self.assertEqual(len(result.parent.encode_inputs), 2)
                self.assertEqual(len(result.pipeline.denoise_inputs), 25)
                self.assertEqual([model.calls for model in result.models.values()], [200, 200])
                self.assertEqual([model.clean_calls for model in result.detectors.values()], [1, 1])
                self.assertEqual(result.summary["combined_scalar_autograd_calls"], 0)

    def test_first_cycle_fixed_then_step_six_consumes_next_weights(self):
        for name, result in self.groups.items():
            with self.subTest(variant=name):
                initial = {"detector": [1., 1.], "depth": [.5, .5], "task": [.5, .5]}
                self.assertTrue(all(row["current_period_weights"] == initial
                                    for row in result.rows[:5]))
                cycle = result.rows[4]["cycle_summary"]
                self.assertEqual(cycle["R"]["task"], [1., 1.])
                self.assertEqual(result.rows[5]["current_period_weights"], cycle["next"])
                self.assertTrue(cycle["applied_next"])
                self.assertFalse(result.rows[24]["cycle_summary"]["applied_next"])
                self.assertEqual(result.summary["statistical_cycles"], 5)
                self.assertEqual(result.summary["weight_refreshes_used"], 4)
        for name in ("periodic_adaptive", "periodic_ema", "complete"):
            self.assertNotEqual(self.groups[name].rows[5]["detection_model_weights"], [1., 1.])

    def test_named_switches_reach_gradient_and_depth_helpers(self):
        for name, result in self.groups.items():
            expected = sample_contract.VARIANT_SWITCHES[name]
            self.assertEqual(result.objective.depth_target, expected["depth_target"])
            for row in result.rows:
                self.assertEqual(row["switches"], expected)
                allocation = row["task_allocation"]
                for key in ("task_balance", "norm_match", "conflict_guard"):
                    self.assertEqual(allocation[key], expected[key])
                self.assertEqual(allocation["additional_vjp_calls"], 0)
                self.assertTrue(allocation["reference_uses_current_model_weighted_task_gradients"])
                self.assertEqual(row["autograd_calls_this_step"], 2)
            manifest = json.loads((result.output / "state_evidence/MANIFEST.json").read_text())
            probe = next(row for row in manifest["fixed_update_math_checks"]
                         if row["global_step"] == 1)
            state = torch.load(result.output / probe["path"], weights_only=True)["state"]
            self.assertEqual(state["switches"], expected)
            self.assertEqual(state["current_period_weights"]["detector"], [1., 1.])

    def test_selected_depth_target_retains_overshoot_or_removes_pullback(self):
        old, hinge = self.groups["old_target_equal"], self.groups["hinge_target_equal"]
        # The toy inverse model already exceeds beta. Exact-beta pulls it back;
        # hinge has a live, exactly-zero model loss on the same first-step input.
        self.assertEqual(old.rows[0]["raw_losses"], hinge.rows[0]["raw_losses"])
        self.assertGreater(old.rows[0]["raw_depth_model_losses"]["dpt_large"], 0.)
        self.assertEqual(hinge.rows[0]["raw_depth_model_losses"]["dpt_large"], 0.)
        self.assertEqual(old.rows[0]["depth_deficit_details"],
                         hinge.rows[0]["depth_deficit_details"])

    def test_retained_reference_uses_actual_model_weighted_task_gradients(self):
        result = self.groups["periodic_ema"]
        manifest = json.loads((result.output / "state_evidence/MANIFEST.json").read_text())
        probe = next(row for row in manifest["fixed_update_math_checks"]
                     if row["global_step"] == 25)
        state = torch.load(result.output / probe["path"], weights_only=True)["state"]
        self.assertNotEqual(state["current_period_weights"]["detector"], [1., 1.])
        reference = state["detection_task_gradient"].float() + state["depth_task_gradient"].float()
        expected_norm = float(torch.linalg.vector_norm(reference.flatten()))
        allocation = state["task_allocation"]
        self.assertEqual(allocation["reference_gradient_norm"], expected_norm)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(state["joint_gradient"].flatten())),
                               expected_norm, delta=max(1e-12, expected_norm * 2e-6))
        self.assertFalse(allocation["model_weights_recomputed"])
        self.assertEqual(allocation["additional_vjp_calls"], 0)

    def test_evaluator_accepts_each_selected_scientific_variant(self):
        for name, result in self.groups.items():
            plan = {"_plan_sha256": "b" * 64, "source_bindings": {"fixture": "c" * 64},
                    "scientific_config": sample_contract.scientific_config_for(name),
                    "budget": {"updates_per_sample": 25}}
            self.assertIs(sample_evaluate.validate_summary(plan, sample("case"),
                                                           result.summary), result.summary)
        bad = dict(self.groups["complete"].summary,
                   scientific_config=sample_contract.scientific_config_for("old_target_equal"))
        with self.assertRaises(ValueError):
            sample_evaluate.validate_summary(plan, sample("case"), bad)

    def test_period_one_custom_override_is_applied(self):
        switches = {**sample_contract.SWITCHES, "period": 1, "model_adaptive": False,
                    "task_adaptive": False, "ema": False, "conflict_guard": False,
                    "norm_match": False, "task_balance": "raw_sum"}
        science = sample_contract.scientific_config_for("custom", switches=switches)
        result = run_case(self.root / "period_one", science)
        self.assertEqual(result.summary["statistical_cycles"], 25)
        self.assertEqual(result.summary["weight_refreshes_used"], 24)
        self.assertTrue(all(row["cycle_summary"] is not None for row in result.rows))
        self.assertTrue(all(row["detection_model_weights"] == [1., 1.]
                            for row in result.rows))
        with self.assertRaises(ValueError):
            sample_contract.scientific_config_for("complete", switches=switches)

    def test_real_batch_cases_reset_reused_pipeline_and_controller(self):
        science = sample_contract.scientific_config_for("complete")
        parent, pipeline, detectors, models, objective, kernels = fixture(science)
        # An unrelated earlier trajectory deliberately leaves scheduler/RNG state.
        pipeline.scheduler.step_index = 300
        random.random(); np.random.rand(); torch.rand(13)
        cases = [sample("batch_a"), sample("batch_b")]
        records = {case["sample_id"]: SimpleNamespace(source_path=case["source_path"])
                   for case in cases}
        plan = {"samples": cases, "scientific_config": science, "_plan_sha256": "b" * 64,
                "source_bindings": {"fixture": "c" * 64}, "shard_id": None,
                "optimization_on_original_validation": False,
                "budget": {"sample_count": 2, "updates_per_sample": 25, "total_updates": 50}}
        output = self.root / "batch"
        output.mkdir()
        with patch.object(batch_training, "record_for_sample",
                lambda p, case, rec: rec[case["sample_id"]]), contextlib.redirect_stdout(io.StringIO()):
            result = batch_training.run_batch_cases(plan=plan, output_root=output,
                records=records, parent=parent, config=FakeConfig(), pipeline=pipeline,
                prompt_pack=None, adapters=detectors, depth_objective=objective,
                device=torch.device("cpu"), dtype=torch.float32, kernels=kernels,
                context_for=lambda record: context(), initial_for=initialization,
                runtime_provenance={"real_cuda_run": False, "fixture": True})
        self.assertEqual(result["executed_updates"], 50)
        left = [json.loads(row) for row in (output / "samples/batch_a/train/attack_trace.jsonl").read_text().splitlines()]
        right = [json.loads(row) for row in (output / "samples/batch_b/train/attack_trace.jsonl").read_text().splitlines()]
        for a, b in zip(left, right):
            for key in ("state_bindings", "raw_losses", "raw_depth_model_losses",
                        "current_period_weights", "cycle_summary", "update_statistics"):
                self.assertEqual(a[key], b[key], key)
        self.assertEqual(left[4]["cycle_summary"]["R"]["task"], [1., 1.])
        self.assertEqual(right[4]["cycle_summary"]["R"]["task"], [1., 1.])
        self.assertEqual(pipeline.scheduler.step_index, 25)
        self.assertEqual([model.clean_calls for model in detectors.values()], [2, 2])
        for case in cases:
            runtime = json.loads((output / "samples" / case["case_id"] / "train/runtime_provenance.json").read_text())
            self.assertEqual(runtime["trajectory_reset"],
                             "fresh_latent_scheduler_periodic_controller_rng_per_sample")


if __name__ == "__main__":
    unittest.main()
