"""CPU checks using frozen geometry/loss/spatial code and toy depth adapters.

No source images, pretrained weights, CUDA experiment, or model downloads.
Packaged tests can bind OURS_PERIODIC_R2_SOURCE, OURS_PERIODIC_DEPTH_SOURCE,
and OURS_PERIODIC_DEPTH_PRODUCTION to the corresponding frozen source roots.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
R2 = Path(os.environ.get("OURS_PERIODIC_R2_SOURCE", str(SOURCE / "meeting_full_repeat_20260924_r2")))
DEPTH = Path(os.environ.get("OURS_PERIODIC_DEPTH_SOURCE", str(SOURCE / "r2_joint_depth_20261001_v2")))
PRODUCTION = Path(os.environ.get("OURS_PERIODIC_DEPTH_PRODUCTION", str(SOURCE /
    "normalized-v29-c0-u1-lr-20260922/normalized-v29-c0-u1-lr005-20260922/parent_v17/03_方法/production")))
for path in (PRODUCTION, R2, DEPTH, ROOT):
    sys.path.insert(0, str(path))

import torch

from depth_hinge_objective import DepthObjective, DEPTH_MODELS, DEPTH_SEMANTICS, semantic_hinge_far_loss
from depth_objective import DepthObjective as HistoricalDepthObjective
from evidence import CallLedger, fingerprint
from geometry_eot import sample_views, transform_support_mask, transform_tensor
from losses import semantic_far_change, semantic_target_far_loss


class DirectRoadView:
    def __init__(self, patch):
        self.patch = patch

    def render(self, context):
        return SimpleNamespace(image=context.benign + self.patch,
                               support=torch.ones_like(context.object_mask))


class CoordinateModel(torch.nn.Module):
    """Positive toy outputs whose semantic far displacement is the RGB signal."""
    def __init__(self, semantics, *, disconnected=False, invalid=None):
        super().__init__()
        self.training_output_semantics = semantics
        self.gain = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.calls = []
        self.disconnected = disconnected
        self.invalid = invalid

    def predict_for_training(self, image):
        self.calls.append({"image": image.detach().clone(), "grad_enabled": torch.is_grad_enabled()})
        signal = image.mean(dim=1, keepdim=True) * self.gain
        sign = 1 if self.training_output_semantics == "relative_depth" else -1
        output = torch.sinh(torch.asinh(torch.ones_like(signal)) + sign * signal)
        if self.invalid is not None:
            output = output.clone()
            output[..., 0, 0] = self.invalid
        return output.detach() if self.disconnected else output


def models(**kwargs):
    return {name: CoordinateModel(DEPTH_SEMANTICS[name], **kwargs) for name in DEPTH_MODELS}


def context():
    return SimpleNamespace(benign=torch.zeros(1, 3, 16, 16, dtype=torch.float64),
                           object_mask=torch.ones(1, 1, 16, 16, dtype=torch.float64))


def views(kind="translation", identity=False):
    config = {"crop_side_fraction": [.9, 1.], "scale_range": [.95, 1.05],
              "translation_fraction": [-.04, .04], "rotation_degrees": [-4., 4.],
              "kinds": [kind], "padding_mode": "border", "random_views": 3,
              "max_attempts": 50}
    result = sample_views(16, 16, [6, 5, 10, 9], [6, 11, 10, 14], seed=31, config=config)
    if identity:
        result = [dict(copy.deepcopy(result[0]), index=index) for index in range(4)]
    return result


def book(calls=1):
    return CallLedger({"depth_clean_forwards": 8 * calls, "depth_adv_forwards": 8 * calls})


def patch(value=0.0):
    return torch.full((1, 3, 16, 16), value, dtype=torch.float64, requires_grad=True)


def displaced(clean, semantic, far_shift):
    sign = 1 if semantic == "relative_depth" else -1
    scale = clean.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(clean.dtype).eps)
    return scale * torch.sinh(torch.asinh(clean / scale) + sign * far_shift)


class DepthHingeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)

    def objective(self, *, depth_target="at_least_beta", beta=.1, epsilon=1e-8, **kwargs):
        return DepthObjective(SimpleNamespace(RoadRGBView=DirectRoadView), models(**kwargs),
                              beta=beta, depth_target=depth_target, epsilon=epsilon)

    def test_beyond_beta_has_zero_loss_and_no_pullback_in_both_semantics(self):
        clean = torch.full((1, 1, 2, 3), 2., dtype=torch.float64)
        roi = torch.ones_like(clean, dtype=torch.bool)
        for semantic in DEPTH_SEMANTICS.values():
            with self.subTest(semantic=semantic):
                adv = displaced(clean, semantic, .2).requires_grad_()
                hinge = semantic_hinge_far_loss(adv, clean, roi, semantics=semantic, target_shift=.1)
                gradient, = torch.autograd.grad(hinge, adv, retain_graph=True)
                self.assertTrue(hinge.requires_grad)
                self.assertEqual(float(hinge.detach()), 0.)
                self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
                exact = semantic_target_far_loss(adv, clean, roi, semantics=semantic, target_shift=.1)
                old_gradient, = torch.autograd.grad(exact, adv)
                self.assertGreater(float(exact.detach()), 0.)
                self.assertGreater(float(old_gradient.abs().sum()), 0.)

    def test_exact_floating_boundary_has_zero_residual_and_gradient(self):
        clean = torch.ones(1, 1, 2, 3, dtype=torch.float64)
        roi = torch.ones_like(clean, dtype=torch.bool)
        for semantic in DEPTH_SEMANTICS.values():
            adv = displaced(clean, semantic, .1).requires_grad_()
            # Bind the representable semantic boundary, rather than silently
            # treating a floating roundoff residual as an algorithmic failure.
            beta = float(semantic_far_change(adv.detach(), clean, semantic)[0, 0, 0, 0])
            value = semantic_hinge_far_loss(adv, clean, roi, semantics=semantic, target_shift=beta)
            gradient, = torch.autograd.grad(value, adv)
            self.assertEqual(float(value.detach()), 0.)
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_shortfall_gradient_moves_depth_and_inverse_depth_farther(self):
        clean = torch.full((1, 1, 2, 3), 2., dtype=torch.float64)
        roi = torch.ones_like(clean, dtype=torch.bool)
        for semantic, sign in (("relative_depth", 1), ("relative_inverse_depth", -1)):
            adv = clean.clone().requires_grad_()
            loss = semantic_hinge_far_loss(adv, clean, roi, semantics=semantic, target_shift=.1)
            gradient, = torch.autograd.grad(loss, adv)
            self.assertAlmostEqual(float(loss.detach()), .01, places=14)
            self.assertTrue(bool((gradient * sign < 0).all()))
            updated = adv.detach() - .01 * gradient
            self.assertTrue(bool((semantic_far_change(updated, clean, semantic) > 0).all()))

    def test_pixels_outside_roi_and_overshoot_have_zero_gradient(self):
        clean = torch.ones(1, 1, 2, 2, dtype=torch.float64)
        far = torch.tensor([[[[-.1, -.1], [.5, .2]]]], dtype=torch.float64)
        roi = torch.tensor([[[[True, False], [False, True]]]])
        for semantic in DEPTH_SEMANTICS.values():
            adv = displaced(clean, semantic, far).requires_grad_()
            loss = semantic_hinge_far_loss(adv, clean, roi, semantics=semantic, target_shift=.1)
            gradient, = torch.autograd.grad(loss, adv)
            self.assertAlmostEqual(float(loss.detach()), .02, places=14)
            self.assertTrue(torch.equal(gradient[~roi], torch.zeros_like(gradient[~roi])))
            self.assertEqual(float(gradient[0, 0, 1, 1]), 0.)
            self.assertNotEqual(float(gradient[0, 0, 0, 0]), 0.)

    def test_identity_control_is_rms_shortfall_not_mean_far_shift(self):
        value = patch()
        with torch.no_grad():
            value[..., :8, :] = -.2
            value[..., 8:, :] = .2
        objective = self.objective()
        loss, evidence = objective.call(value, context(), views(identity=True), book())
        expected = math.sqrt((.3**2 + 0.**2) / 2) / (.1 + 1e-8)
        for name in DEPTH_MODELS:
            row = evidence["views"][0]["models"][name]
            self.assertAlmostEqual(row["semantic_far_change_roi_mean"], 0., places=14)
            self.assertAlmostEqual(evidence["control_deficits"][name], expected, places=12)
            self.assertEqual(evidence["control_deficits"][name], row["control_deficit"])
            self.assertEqual(row["above_beta_pixel_count"], 128)
            self.assertEqual(row["below_beta_pixel_count"], 128)
            self.assertFalse(row["overshoot_pullback_enabled"])
        self.assertEqual(evidence["control_view_index"], 0)
        self.assertTrue(evidence["control_deficits_detached"])
        self.assertTrue(loss.requires_grad)
        json.dumps(evidence, allow_nan=False)

    def test_depth_deficit_above_three_is_not_clipped(self):
        _, evidence = self.objective().call(patch(-.5), context(), views(identity=True), book())
        self.assertFalse(evidence["control_deficits_clipped"])
        for value in evidence["control_deficits"].values():
            self.assertGreater(value, 3.)
            self.assertAlmostEqual(value, .6 / (.1 + 1e-8), places=12)

    def test_all_satisfied_pixels_keep_full_forward_budget_and_live_zero_graph(self):
        objective = self.objective()
        value = patch(.2)
        ledger = book()
        losses = {}
        total, evidence = objective.call(value, context(), views(identity=True), ledger, model_losses_out=losses)
        self.assertEqual(float(total.detach()), 0.)
        gradient, = torch.autograd.grad(total, value, retain_graph=True)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
        ledger.require_complete()
        self.assertEqual(evidence["depth_clean_forwards"], 8)
        self.assertEqual(evidence["depth_adv_forwards"], 8)
        self.assertEqual(evidence["control_forward_calls"], 0)
        for name, model in objective.adapters.items():
            self.assertEqual(len(model.calls), 8)
            self.assertEqual([r["grad_enabled"] for r in model.calls], [False, True] * 4)
            self.assertTrue(losses[name].requires_grad)
            self.assertIsNotNone(losses[name].grad_fn)
            self.assertEqual(evidence["control_deficits"][name], 0.)

    def test_model_losses_out_retains_two_live_graphs_and_equal_model_mean(self):
        objective = self.objective()
        value = patch(.02)
        losses = {}
        total, evidence = objective.call(value, context(), views(), book(), model_losses_out=losses)
        self.assertEqual(tuple(losses), DEPTH_MODELS)
        self.assertTrue(torch.equal(total, torch.stack(list(losses.values())).mean()))
        for name, loss in losses.items():
            gradient, = torch.autograd.grad(loss, value, retain_graph=True)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)
            self.assertEqual(float(loss.detach()), evidence["raw_model_losses"][name])
        for model in objective.adapters.values():
            self.assertFalse(model.training)
            self.assertFalse(model.gain.requires_grad)
            self.assertIsNone(model.gain.grad)

    def test_exact_beta_retains_historical_loss_and_patch_gradient_exactly(self):
        parent = SimpleNamespace(RoadRGBView=DirectRoadView)
        objective = self.objective(depth_target="exact_beta")
        historical = HistoricalDepthObjective(parent, models(), beta=.1)
        value = patch(.2)
        source, shared = context(), views()
        new_losses, old_losses = {}, {}
        current, evidence = objective.call(value, source, shared, book(), model_losses_out=new_losses)
        previous, original = historical.call(value, source, shared, book(), model_losses_out=old_losses)
        self.assertTrue(torch.equal(current, previous))
        self.assertEqual(evidence["raw_model_losses"], original["raw_model_losses"])
        current_grad, = torch.autograd.grad(current, value)
        previous_grad, = torch.autograd.grad(previous, value)
        self.assertTrue(torch.equal(current_grad, previous_grad))
        self.assertGreater(float(current_grad.abs().sum()), 0.)
        self.assertTrue(evidence["overshoot_pullback_enabled"])
        self.assertEqual(evidence["depth_target"], "exact_beta")
        # Controls still use the one-sided deficit, even for the exact target.
        for deficit in evidence["control_deficits"].values():
            self.assertEqual(deficit, 0.)

    def test_geometry_family_pairs_and_identity_controls_share_existing_forwards(self):
        for kind in ("crop", "scale", "translation", "rotation"):
            with self.subTest(kind=kind):
                objective = self.objective()
                source, shared = context(), views(kind)
                _, evidence = objective.call(patch(), source, shared, book())
                for index, view in enumerate(shared):
                    expected = transform_tensor(source.benign, view)
                    row = evidence["views"][index]
                    self.assertEqual(row["source_view_fingerprint"], fingerprint(view))
                    for model in objective.adapters.values():
                        self.assertTrue(torch.equal(model.calls[2 * index]["image"], expected))
                        self.assertTrue(torch.equal(model.calls[2 * index + 1]["image"], expected))
                for name in DEPTH_MODELS:
                    self.assertEqual(evidence["control_deficits"][name], evidence["views"][0]["models"][name]["control_deficit"])

    def test_border_padding_is_excluded_from_depth_roi(self):
        shared = views("crop")
        for row in shared[1:]:
            row.update(kind="crop", forward_matrix=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                       inverse_matrix=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            row["parameters"] = {"crop_xyxy": [1, 1, 15, 15], "pad_lrtb": [1, 1, 1, 1], "crop_resized": False}
            row["source_support"]["input_xyxy"] = [1., 1., 15., 15.]
        _, evidence = self.objective().call(patch(), context(), shared, book())
        for view, row in zip(shared[1:], evidence["views"][1:]):
            expected = transform_support_mask(view).unsqueeze(0).bool()
            self.assertEqual(row["valid_roi_pixels"], 196)
            self.assertEqual(row["padding_pixels_excluded_from_roi"], 60)
            self.assertEqual(row["object_roi_fingerprint"], fingerprint(expected))

    def test_no_clean_cache_across_calls(self):
        objective = self.objective()
        source, ledger = context(), book(calls=2)
        _, first = objective.call(patch(), source, views(), ledger)
        source.benign = source.benign + .05
        _, second = objective.call(patch(), source, views(), ledger)
        ledger.require_complete()
        self.assertNotEqual(first["source_clean_fingerprint"], second["source_clean_fingerprint"])
        self.assertEqual(len(objective.adapters["monodepth2"].calls), 16)

    def test_invalid_probe_container_fails_before_any_forward(self):
        objective = self.objective()
        for invalid in ({"old": torch.tensor(1., requires_grad=True)}, [], (), "", 0):
            ledger = book()
            error = ValueError if isinstance(invalid, dict) else TypeError
            with self.assertRaises(error):
                objective.call(patch(), context(), views(), ledger, model_losses_out=invalid)
            self.assertEqual(ledger.attempted, {"depth_clean_forwards": 0, "depth_adv_forwards": 0})

    def test_empty_roi_and_disconnected_model_fail_closed(self):
        source, ledger = context(), book()
        source.object_mask.zero_()
        with self.assertRaisesRegex(ValueError, "ROI is empty"):
            self.objective().call(patch(), source, views(), ledger)
        self.assertEqual(sum(ledger.attempted.values()), 0)
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.objective(disconnected=True).call(patch(), context(), views(), book())

    def test_negative_and_nonfinite_predictions_are_not_filtered_or_clamped(self):
        for value in (-1., float("nan"), float("inf")):
            ledger = book()
            with self.assertRaisesRegex(ValueError, "negative or non-finite"):
                self.objective(invalid=value).call(patch(), context(), views(), ledger)
            self.assertEqual(ledger.completed["depth_clean_forwards"], 1)
            self.assertEqual(ledger.completed["depth_adv_forwards"], 0)

    def test_zero_clean_scale_floor_remains_finite(self):
        clean = torch.zeros(1, 1, 2, 3)
        for semantic in DEPTH_SEMANTICS.values():
            adv = clean.clone().requires_grad_()
            loss = semantic_hinge_far_loss(adv, clean, torch.ones_like(clean, dtype=torch.bool),
                                          semantics=semantic, target_shift=.1)
            gradient, = torch.autograd.grad(loss, adv)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_invalid_target_epsilon_beta_and_view_contract_rejected(self):
        for target in ("", "both_sides", None):
            with self.assertRaises(ValueError):
                self.objective(depth_target=target)
        for field in ("beta", "epsilon"):
            for value in (0., -1., float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    self.objective(**{field: value})
            for value in (True, ".1"):
                with self.assertRaises(TypeError):
                    self.objective(**{field: value})
        objective = self.objective()
        with self.assertRaisesRegex(ValueError, "four indexed"):
            objective.call(patch(), context(), views()[:3], book())
        wrong = views()
        wrong[0]["kind"] = "rotation"
        with self.assertRaisesRegex(ValueError, "first shared"):
            objective.call(patch(), context(), wrong, book())


if __name__ == "__main__":
    unittest.main()
