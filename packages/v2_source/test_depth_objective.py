"""CPU tests using actual R2 geometry and historical loss; no pretrained models.

Optional paths for a packaged test environment:
  R2_JOINT_R2_SOURCE: directory containing geometry_eot.py/evidence.py
  R2_JOINT_DEPTH_PRODUCTION: directory containing the frozen historical losses.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
R2_SOURCE = Path(os.environ.get("R2_JOINT_R2_SOURCE", str(ROOT.parent / "meeting_full_repeat_20260924_r2")))
DEPTH_PRODUCTION = Path(os.environ.get(
    "R2_JOINT_DEPTH_PRODUCTION",
    str(ROOT.parent / "normalized-v17-roadpatch-20260920" / "03_方法" / "production")))
for directory in (DEPTH_PRODUCTION, R2_SOURCE, ROOT):
    sys.path.insert(0, str(directory))

import torch
import torch.nn.functional as F

from depth_objective import DepthObjective, DEPTH_MODELS, DEPTH_SEMANTICS
from evidence import CallLedger, fingerprint
from geometry_eot import sample_views, transform_support_mask, transform_tensor
from losses import semantic_far_change, semantic_target_far_loss


class RoadView:
    """Toy renderer modifies a road region outside the toy vehicle mask."""
    def __init__(self, patch):
        self.patch = patch

    def render(self, context):
        support = torch.zeros_like(context.object_mask)
        support[..., 11:14, 6:10] = 1
        texture = F.interpolate(self.patch, size=context.benign.shape[-2:], mode="bilinear", align_corners=False)
        return SimpleNamespace(image=context.benign + .1 * texture * support, support=support)


class DepthModel(torch.nn.Module):
    """Positive local + global response gives the ROI a road receptive field."""
    def __init__(self, semantics, invalid=None, disconnected=False):
        super().__init__()
        self.training_output_semantics = semantics
        self.scale = torch.nn.Parameter(torch.tensor(.3))
        self.calls = []
        self.invalid = invalid
        self.disconnected = disconnected

    def predict_for_training(self, image):
        self.calls.append({"image": image.detach().clone(), "grad_enabled": torch.is_grad_enabled(),
                           "requires_grad": image.requires_grad})
        intensity = image.mean(dim=1, keepdim=True)
        response = self.scale * (intensity + .25 * intensity.mean(dim=(-2, -1), keepdim=True))
        result = 1 + response if self.training_output_semantics == "relative_depth" else 2 - response
        if self.invalid is not None:
            result = result.clone()
            result[..., 0, 0] = self.invalid
        return result.detach() if self.disconnected else result


def context():
    image = torch.linspace(.2, .6, 16 * 16).reshape(1, 1, 16, 16).repeat(1, 3, 1, 1)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 5:9, 6:10] = 1
    return SimpleNamespace(benign=image, object_mask=mask)


def views(kind="translation"):
    config = {"crop_side_fraction": [.9, 1.], "scale_range": [.95, 1.05],
              "translation_fraction": [-.04, .04], "rotation_degrees": [-4., 4.],
              "kinds": [kind], "padding_mode": "border", "random_views": 3,
              "max_attempts": 50}
    return sample_views(16, 16, [6, 5, 10, 9], [6, 11, 10, 14], seed=31, config=config)


def adapters(**kwargs):
    return {name: DepthModel(DEPTH_SEMANTICS[name], **kwargs) for name in DEPTH_MODELS}


def ledger(steps=1):
    return CallLedger({"depth_clean_forwards": 8 * steps, "depth_adv_forwards": 8 * steps})


class DepthObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_historical_loss_far_direction_and_target(self):
        clean = torch.full((1, 1, 3, 4), 2., dtype=torch.float64)
        roi = torch.ones_like(clean, dtype=torch.bool)
        for semantics, sign in (("relative_depth", 1), ("relative_inverse_depth", -1)):
            with self.subTest(semantics=semantics):
                adv = clean.clone().requires_grad_()
                loss = semantic_target_far_loss(adv, clean, roi, semantics=semantics, target_shift=.1)
                self.assertAlmostEqual(float(loss), .01, places=14)
                gradient, = torch.autograd.grad(loss, adv)
                self.assertTrue(bool((gradient * sign < 0).all()))
                updated = adv.detach() - .01 * gradient
                change = semantic_far_change(updated, clean, semantics)
                self.assertTrue(bool((change > 0).all()))
                scale = clean.mean()
                target = scale * torch.sinh(torch.asinh(clean / scale) + sign * .1)
                target.requires_grad_()
                target_loss = semantic_target_far_loss(target, clean, roi, semantics=semantics, target_shift=.1)
                self.assertLess(float(target_loss), 1e-28)

    def test_patch_gradient_and_exact_forward_budget(self):
        models = adapters()
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), models)
        patch = torch.full((1, 3, 2, 3), .15, requires_grad=True)
        book = ledger()
        loss, evidence = objective.call(patch, context(), views(), book)
        gradient, = torch.autograd.grad(loss, patch)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)
        book.require_complete()
        for model in models.values():
            self.assertFalse(model.training)
            self.assertFalse(model.scale.requires_grad)
            self.assertIsNone(model.scale.grad)
            self.assertEqual(len(model.calls), 8)
            self.assertEqual([x["grad_enabled"] for x in model.calls], [False, True] * 4)
        expected = sum(evidence["raw_model_losses"].values()) / 2
        self.assertAlmostEqual(float(loss), expected, places=7)
        json.dumps(evidence, allow_nan=False)
        self.assertEqual(evidence["depth_clean_forwards"], 8)
        self.assertEqual(evidence["depth_adv_forwards"], 8)
        self.assertLess(len(json.dumps(evidence)), 20000)

    def test_every_geometry_family_uses_exact_clean_adversarial_pair(self):
        for kind in ("crop", "scale", "translation", "rotation"):
            with self.subTest(kind=kind):
                models = adapters()
                objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), models)
                source = context()
                shared = views(kind)
                loss, evidence = objective.call(torch.zeros(1, 3, 2, 3, requires_grad=True), source, shared, ledger())
                self.assertAlmostEqual(float(loss), .01, places=7)
                for index, view in enumerate(shared):
                    expected = transform_tensor(source.benign, view)
                    for model in models.values():
                        self.assertTrue(torch.equal(model.calls[2 * index]["image"], expected))
                        self.assertTrue(torch.equal(model.calls[2 * index + 1]["image"], expected))
                    row = evidence["views"][index]
                    self.assertEqual(row["source_view_fingerprint"], fingerprint(view))
                    self.assertEqual(row["clean_input_fingerprint"], row["adversarial_input_fingerprint"])
                    for result in row["models"].values():
                        self.assertEqual(result["semantic_far_change_roi_mean"], 0.)

    def test_crop_replicated_padding_is_excluded_from_roi(self):
        source = context()
        source.object_mask.fill_(1)
        shared = views("crop")
        for row in shared[1:]:
            row.update(kind="crop", forward_matrix=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                       inverse_matrix=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            row["parameters"] = {"crop_xyxy": [1, 1, 15, 15], "pad_lrtb": [1, 1, 1, 1], "crop_resized": False}
            row["source_support"]["input_xyxy"] = [1., 1., 15., 15.]
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), adapters())
        _, evidence = objective.call(torch.zeros(1, 3, 2, 3), source, shared, ledger())
        for view, row in zip(shared[1:], evidence["views"][1:]):
            support = transform_support_mask(view).unsqueeze(0).bool()
            self.assertEqual(row["valid_roi_pixels"], 14 * 14)
            self.assertEqual(row["padding_pixels_excluded_from_roi"], 16 * 16 - 14 * 14)
            self.assertEqual(row["object_roi_fingerprint"], fingerprint(support))

    def test_no_clean_cache_reused_across_calls(self):
        models = adapters()
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), models)
        source = context()
        book = ledger(steps=2)
        _, first = objective.call(torch.zeros(1, 3, 2, 3), source, views(), book)
        source.benign = source.benign + .02
        _, second = objective.call(torch.zeros(1, 3, 2, 3), source, views(), book)
        book.require_complete()
        self.assertNotEqual(first["source_clean_fingerprint"], second["source_clean_fingerprint"])
        self.assertEqual(len(models["monodepth2"].calls), 16)

    def test_plain_adapters_freeze_encoder_decoder_and_model(self):
        mono_encoder = DepthModel("relative_depth")
        mono_decoder = torch.nn.Conv2d(1, 1, 1)
        dpt_model = DepthModel("relative_inverse_depth")
        wrapped = {
            "monodepth2": SimpleNamespace(
                encoder=mono_encoder, decoder=mono_decoder,
                training_output_semantics="relative_depth",
                predict_for_training=mono_encoder.predict_for_training),
            "dpt_large": SimpleNamespace(
                model=dpt_model, training_output_semantics="relative_inverse_depth",
                predict_for_training=dpt_model.predict_for_training),
        }
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), wrapped)
        for module in (mono_encoder, mono_decoder, dpt_model):
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        patch = torch.zeros(1, 3, 2, 3, requires_grad=True)
        loss, _ = objective.call(patch, context(), views(), ledger())
        gradient, = torch.autograd.grad(loss, patch)
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_probe_exposes_each_live_model_loss_without_changing_arithmetic(self):
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), adapters())
        patch = torch.full((1, 3, 2, 3), .15, requires_grad=True)
        source, shared = context(), views()
        per_model = {}
        book = ledger(steps=2)
        total, evidence = objective.call(patch, source, shared, book, model_losses_out=per_model)
        self.assertEqual(tuple(per_model), DEPTH_MODELS)
        self.assertTrue(torch.equal(total, torch.stack(list(per_model.values())).mean()))
        for model_id in DEPTH_MODELS:
            value = per_model[model_id]
            self.assertEqual(value.ndim, 0)
            self.assertTrue(value.requires_grad)
            self.assertIsNotNone(value.grad_fn)
            gradient, = torch.autograd.grad(value, patch, retain_graph=True)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)
        default_total, default_evidence = objective.call(patch, source, shared, book)
        self.assertTrue(torch.equal(total, default_total))
        self.assertEqual(evidence, default_evidence)
        book.require_complete()

    def test_probe_rejects_nonempty_or_nondict_before_forward(self):
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), adapters())
        for invalid in ({"old": torch.tensor(1., requires_grad=True)}, [], (), "", 0):
            with self.subTest(invalid_type=type(invalid).__name__):
                book = ledger()
                error = ValueError if isinstance(invalid, dict) else TypeError
                with self.assertRaises(error):
                    objective.call(torch.zeros(1, 3, 2, 3), context(), views(), book,
                                   model_losses_out=invalid)
                self.assertEqual(book.attempted, {"depth_clean_forwards": 0, "depth_adv_forwards": 0})

    def test_invalid_output_fails_without_clipping_or_filtering(self):
        for invalid in (-1., float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), adapters(invalid=invalid))
                book = ledger()
                with self.assertRaisesRegex(ValueError, "negative or non-finite"):
                    objective.call(torch.zeros(1, 3, 2, 3), context(), views(), book)
                self.assertEqual(book.completed["depth_clean_forwards"], 1)
                self.assertEqual(book.completed["depth_adv_forwards"], 0)

    def test_empty_roi_and_disconnected_prediction_fail(self):
        source = context()
        source.object_mask.zero_()
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), adapters())
        with self.assertRaisesRegex(ValueError, "ROI is empty"):
            objective.call(torch.zeros(1, 3, 2, 3), source, views(), ledger())
        objective = DepthObjective(SimpleNamespace(RoadRGBView=RoadView), adapters(disconnected=True))
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            objective.call(torch.zeros(1, 3, 2, 3, requires_grad=True), context(), views(), ledger())

    def test_zero_output_is_finite_in_historical_asinh_loss(self):
        clean = torch.zeros(1, 1, 3, 4)
        adversarial = clean.clone().requires_grad_()
        loss = semantic_target_far_loss(adversarial, clean, torch.ones_like(clean),
                                        semantics="relative_inverse_depth", target_shift=.1)
        gradient, = torch.autograd.grad(loss, adversarial)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_model_ids_semantics_beta_and_views_are_bound(self):
        parent = SimpleNamespace(RoadRGBView=RoadView)
        wrong = adapters()
        wrong["dpt"] = wrong.pop("dpt_large")
        with self.assertRaises(ValueError):
            DepthObjective(parent, wrong)
        wrong = adapters()
        wrong["dpt_large"].training_output_semantics = "metric_depth"
        with self.assertRaises(ValueError):
            DepthObjective(parent, wrong)
        for beta in (0., -1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                DepthObjective(parent, adapters(), beta)
        objective = DepthObjective(parent, adapters())
        with self.assertRaises(ValueError):
            objective.call(torch.zeros(1, 3, 2, 3), context(), views()[:3], ledger())
        wrong_views = copy.deepcopy(views())
        wrong_views[0]["kind"] = "rotation"
        with self.assertRaises(ValueError):
            objective.call(torch.zeros(1, 3, 2, 3), context(), wrong_views, ledger())


if __name__ == "__main__":
    unittest.main()
