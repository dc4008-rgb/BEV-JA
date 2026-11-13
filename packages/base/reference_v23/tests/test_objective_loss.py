"""CPU tests with real randomly initialized torchvision core detectors only."""
from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.ssd import SSD
from torchvision.ops import MultiScaleRoIAlign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import objective_loss as dispatch
from training import attack_objectives as objective
from training.state_evidence import CallLedger, CountingAdapter


def setUpModule():
    global old_threads
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(old_threads)


class TinyBackbone(nn.Module):
    def __init__(self, ssd=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=4, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(.3)
        self.out_channels = [8] if ssd else 8

    def forward(self, image):
        return OrderedDict([("0", self.dropout(self.bn(self.conv(image))).relu())])


class Adapter:
    def __init__(self, model_id, model):
        self.model_id, self.model, self.calls = model_id, model, 0

    def predict(self, image):
        self.calls += 1
        return self.model([image[0]])


def tiny_adapter(model_id="ssd300"):
    with torch.random.fork_rng():
        torch.manual_seed(721)
        if model_id == "fasterrcnn":
            model = FasterRCNN(
                TinyBackbone(), num_classes=9, min_size=64, max_size=64,
                rpn_anchor_generator=AnchorGenerator(((8, 16, 32),), ((.5, 1., 2.),)),
                box_roi_pool=MultiScaleRoIAlign(["0"], output_size=3, sampling_ratio=2),
                rpn_pre_nms_top_n_train=40, rpn_pre_nms_top_n_test=30,
                rpn_post_nms_top_n_train=20, rpn_post_nms_top_n_test=15,
                rpn_batch_size_per_image=32, box_batch_size_per_image=16,
                box_score_thresh=0., box_detections_per_img=30)
        else:
            model = SSD(TinyBackbone(ssd=True), DefaultBoxGenerator([[2]], scales=[.25, .5]),
                        size=(64, 64), num_classes=9, score_thresh=0.,
                        topk_candidates=20, detections_per_img=20)
    return Adapter(model_id, model.eval().requires_grad_(False))


def source_image(graph=True):
    return torch.linspace(.05, .95, 3 * 48 * 64).reshape(1, 3, 48, 64).requires_grad_(graph)


def run_config():
    return SimpleNamespace(method_id="ours", train_models=("fasterrcnn", "ssd300"), depth_direction="far")


def context():
    return SimpleNamespace(object_box=(16, 12, 40, 36), object_mask=torch.ones((1, 1, 48, 64)))


def ledger_for(phase, count=1):
    name = {"calibration": "calibration_model_forwards", "train_pre": "optimization_pre_model_forwards",
            "train_post": "optimization_post_model_forwards"}[phase]
    ledger = CallLedger({name: count, "victim_model_forwards_total": count})
    ledger.phase = phase
    return ledger


class ContextualPhaseTests(unittest.TestCase):
    def test_real_both_models_loss_gradient_and_matching_stats_exactly_match_frozen_baseline(self):
        for model_id in run_config().train_models:
            with self.subTest(model=model_id):
                adapter, image = tiny_adapter(model_id), source_image()
                before, stats = objective.frozen_reference().contextual_loss(adapter, image, context().object_box)
                after, evidence, diagnostics = objective.capture_contextual_loss(
                    adapter, image, context().object_box, require_attack_graph=True, phase="train_pre")
                self.assertTrue(torch.equal(before, after))
                for key, value in stats.items():
                    self.assertEqual(diagnostics[key], value)
                old_gradient = torch.autograd.grad(before, image)[0]
                new_gradient = torch.autograd.grad(after, image)[0]
                self.assertTrue(torch.equal(old_gradient, new_gradient))
                self.assertGreater(int(torch.count_nonzero(new_gradient)), 0)
                self.assertEqual(adapter.calls, 0)
                self.assertEqual(evidence["matched_logits"].shape[0], diagnostics["matched_prediction_count"])

    def test_car3_only_does_not_suppress_vehicle_group_sum(self):
        for model_id in run_config().train_models:
            adapter = tiny_adapter(model_id)
            loss, evidence, diagnostics = objective.capture_contextual_loss(
                adapter, source_image(), context().object_box, require_attack_graph=True, phase="train_pre")
            probabilities = evidence["matched_logits"].softmax(-1)
            self.assertTrue(torch.equal(loss, probabilities[:, 3].mean()))
            self.assertGreater(float(probabilities[:, [3, 6, 8]].sum(-1).mean().detach()), float(loss.detach()))
            self.assertEqual(diagnostics["class_index"], 3)
            self.assertNotIn("vehicle_classes", diagnostics)

    def test_no_grad_both_phases_both_models_preserve_value_and_no_artificial_graph(self):
        for model_id in run_config().train_models:
            adapter, image = tiny_adapter(model_id), source_image()
            before = objective.capture_contextual_loss(adapter, image, context().object_box,
                                                       require_attack_graph=True, phase="train_pre")
            for phase in ("calibration", "train_post"):
                detached = image.detach()
                with torch.no_grad():
                    after = objective.capture_contextual_loss(adapter, detached, context().object_box,
                                                              require_attack_graph=False, phase=phase)
                self.assertTrue(torch.equal(before[0], after[0]))
                for key, value in before[1].items():
                    self.assertTrue(torch.equal(value, after[1][key]), key)
                    self.assertFalse(after[1][key].requires_grad)
                self.assertFalse(after[0].requires_grad)
                self.assertFalse(detached.requires_grad)
                self.assertEqual(after[2]["phase"], phase)

    def test_one_backbone_no_predict_and_frozen_state_rng_unchanged(self):
        for model_id in run_config().train_models:
            adapter = tiny_adapter(model_id)
            state = {key: value.clone() for key, value in adapter.model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            calls = []
            handle = adapter.model.backbone.register_forward_hook(lambda *_: calls.append(1))
            try:
                loss, _, _ = objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                                               require_attack_graph=True, phase="train_pre")
            finally:
                handle.remove()
            self.assertEqual(calls, [1])
            self.assertEqual(adapter.calls, 0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in state.items():
                self.assertTrue(torch.equal(value, adapter.model.state_dict()[key]))
            self.assertTrue(all(not module.training for module in adapter.model.modules()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in adapter.model.parameters()))

    def test_frcnn_explicit_gt_survives_empty_rpn_and_matches_reference(self):
        adapter, image = tiny_adapter("fasterrcnn"), source_image()
        with patch.object(adapter.model.rpn, "forward", return_value=([torch.empty((0, 4))], {})):
            old, info = objective.frozen_reference().contextual_loss(adapter, image, context().object_box)
            new, evidence, diag = objective.capture_contextual_loss(
                adapter, image, context().object_box, require_attack_graph=True, phase="train_pre")
        self.assertEqual(diag["matched_prediction_count"], 1)
        self.assertEqual(diag["explicit_gt_or_forced_match_count"], 1)
        self.assertEqual(diag["matched_ious"], [1.])
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, image)[0], torch.autograd.grad(new, image)[0]))

    def test_ssd_forced_best_anchor_below_threshold_matches_reference(self):
        adapter, image = tiny_adapter(), source_image()
        target = (0., 0., .01, .01)
        old, info = objective.frozen_reference().contextual_loss(adapter, image, target)
        new, evidence, diag = objective.capture_contextual_loss(
            adapter, image, target, require_attack_graph=True, phase="train_pre")
        self.assertLess(diag["matched_iou_max"], .5)
        self.assertEqual(diag["explicit_gt_or_forced_match_count"], 1)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, image)[0], torch.autograd.grad(new, image)[0]))

    def test_phase_contract_and_detached_pre_rejected_before_backbone(self):
        adapter = tiny_adapter()
        calls = []
        handle = adapter.model.backbone.register_forward_hook(lambda *_: calls.append(1))
        try:
            for flag, phase in ((False, "train_pre"), (True, "calibration"), (1, "train_pre"), (False, "invalid")):
                with self.assertRaises(ValueError):
                    objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                                       require_attack_graph=flag, phase=phase)
            with self.assertRaises(RuntimeError):
                objective.capture_contextual_loss(adapter, source_image(False), context().object_box,
                                                   require_attack_graph=True, phase="train_pre")
            with torch.no_grad(), self.assertRaises(RuntimeError):
                objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                                   require_attack_graph=True, phase="train_pre")
            for phase in ("calibration", "train_post"):
                with self.assertRaisesRegex(RuntimeError, "caller no_grad"):
                    objective.capture_contextual_loss(adapter, source_image(False), context().object_box,
                                                       require_attack_graph=False, phase=phase)
        finally:
            handle.remove()
        self.assertEqual(calls, [])

    def test_invalid_image_target_model_state_and_matcher_fail_closed(self):
        adapter = tiny_adapter()
        for image, target in ((source_image().double(), context().object_box),
                              (torch.full((1, 3, 48, 64), float("nan")), context().object_box),
                              (source_image(), (1, 1, 0, 0)), (source_image(), (0, 0, 100, 100))):
            with self.assertRaises(ValueError):
                objective.capture_contextual_loss(adapter, image, target, require_attack_graph=True, phase="train_pre")
        for attr in ("high_threshold", "low_threshold"):
            with patch.object(adapter.model.proposal_matcher, attr, .6), self.assertRaisesRegex(ValueError, "threshold"):
                objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                                   require_attack_graph=True, phase="train_pre")
        parameter = next(adapter.model.parameters())
        parameter.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "parameters"):
            objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                               require_attack_graph=True, phase="train_pre")
        parameter.requires_grad_(False)
        parameter.grad = torch.zeros_like(parameter)
        with self.assertRaisesRegex(RuntimeError, "pre-existing"):
            objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                               require_attack_graph=True, phase="train_pre")
        parameter.grad = None
        adapter.model.backbone.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            objective.capture_contextual_loss(adapter, source_image(), context().object_box,
                                               require_attack_graph=True, phase="train_pre")

    def test_failure_after_backbone_restores_flags_and_preserves_actual_callback_prefix(self):
        for model_id in run_config().train_models:
            adapter = tiny_adapter(model_id)
            ledger = ledger_for("train_pre")
            wrapped = CountingAdapter(adapter, ledger)
            failing = adapter.model.head if model_id == "ssd300" else adapter.model.roi_heads.box_predictor
            calls = []
            handle = adapter.model.backbone.register_forward_hook(lambda *_: calls.append(1))
            try:
                with patch.object(failing, "forward", side_effect=RuntimeError("after backbone")), \
                     self.assertRaisesRegex(RuntimeError, "after backbone"):
                    dispatch.model_loss(run_config(), {}, model_id, wrapped, context(), source_image(),
                                        torch.ones((1, 1, 48, 64)), {}, "train_fixture",
                                        attack_objective="C3", phase="train_pre")
            finally:
                handle.remove()
            self.assertEqual(calls, [1])
            self.assertEqual(adapter.calls, 0)
            self.assertTrue(all(value == 1 for value in ledger.attempted.values()))
            self.assertTrue(all(value == 0 for value in ledger.completed.values()))
            self.assertTrue(all(not m.training for m in adapter.model.modules()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in adapter.model.parameters()))


class DispatchTests(unittest.TestCase):
    def test_unknown_identity_rejected_without_call(self):
        adapter = tiny_adapter()
        args = (run_config(), {}, "ssd300", adapter, context(), source_image(), torch.ones(1), {}, "train")
        for value in ("c0", "c3", "C1", "C2", None):
            with self.assertRaisesRegex(ValueError, "exactly C0 or C3"):
                dispatch.model_loss(*args, attack_objective=value, phase="train_pre")
        self.assertEqual(adapter.calls, 0)

    def test_c0_both_models_value_gradient_and_original_object_unmodified(self):
        for model_id in run_config().train_models:
            adapter, image = tiny_adapter(model_id), source_image()
            args = (run_config(), {}, model_id, adapter, context(), image, torch.ones(1), {}, "train")
            original = dispatch._parent().model_loss(*args)
            result = dispatch.model_loss(*args, attack_objective="C0", phase="train_pre")
            self.assertTrue(torch.equal(original.total, result.total))
            self.assertEqual(original.scalars(), result.scalars())
            self.assertTrue(torch.equal(torch.autograd.grad(original.total, image)[0],
                                        torch.autograd.grad(result.total, image)[0]))
            with patch.object(dispatch._parent(), "model_loss", return_value=original) as frozen:
                self.assertIs(dispatch.model_loss(*args, attack_objective="C0", phase="train_pre"), original)
            frozen.assert_called_once_with(*args)

    def test_c3_both_models_all_phases_count_one_callback_and_one_real_backbone(self):
        for model_id in run_config().train_models:
            adapter = tiny_adapter(model_id)
            for phase in objective.PHASES:
                ledger = ledger_for(phase)
                wrapped = CountingAdapter(adapter, ledger)
                calls = []
                handle = adapter.model.backbone.register_forward_hook(lambda *_: calls.append(1))
                try:
                    with torch.set_grad_enabled(phase == "train_pre"):
                        result = dispatch.model_loss(run_config(), {}, model_id, wrapped, context(), source_image(),
                                                     torch.ones(1), {}, "train", attack_objective="C3", phase=phase)
                finally:
                    handle.remove()
                ledger.require_complete()
                self.assertEqual(calls, [1])
                self.assertEqual(result.diagnostics["model_id"], model_id)
                self.assertEqual(result.diagnostics["objective"], "contextual_car_class_probability")
                self.assertEqual(result.total.requires_grad, phase == "train_pre")
                self.assertEqual(adapter.calls, 0)
                json.dumps(result.diagnostics, allow_nan=False)

    def test_c3_wrong_model_or_method_rejected(self):
        adapter = tiny_adapter()
        args = (run_config(), {}, "fasterrcnn", adapter, context(), source_image(), torch.ones(1), {}, "train")
        with self.assertRaisesRegex(ValueError, "matching"):
            dispatch.model_loss(*args, attack_objective="C3", phase="train_pre")
        args = (SimpleNamespace(method_id="baseline"), *args[1:])
        with self.assertRaisesRegex(ValueError, "ours"):
            dispatch.model_loss(*args, attack_objective="C3", phase="train_pre")


class CalibrationTests(unittest.TestCase):
    def fixtures(self):
        parent = dispatch._parent()
        from road_patch_contract import RoadRGBView
        crop = torch.full((1, 3, 64, 128), .25)
        carrier = RoadRGBView(crop)
        records = dispatch._fixed_calibrators()
        rendered = SimpleNamespace(image=source_image(False), support=torch.ones((1, 1, 48, 64)))
        ctx = SimpleNamespace(background=source_image(False), road_quad=torch.zeros((4, 2)))
        args = (run_config(), {}, carrier, {"fasterrcnn": object(), "ssd300": object()},
                records, torch.zeros(1), torch.zeros(1), torch.device("cpu"), 20260919, set(), {})
        return parent, crop, carrier, records, rendered, ctx, args

    def test_real_two_detector_calibration_both_variants_eight_callbacks_no_backward(self):
        for variant in ("C0", "C3"):
            parent, crop, carrier, records, rendered, ctx, args = self.fixtures()
            ctx.object_box = context().object_box
            ctx.object_mask = context().object_mask
            ledger = ledger_for("calibration", 8)
            adapters = {name: tiny_adapter(name) for name in run_config().train_models}
            wrapped = {name: CountingAdapter(value, ledger) for name, value in adapters.items()}
            actual_calls = {name: [] for name in adapters}
            handles = [value.model.backbone.register_forward_hook(
                lambda *_, name=name: actual_calls[name].append(1)) for name, value in adapters.items()]
            try:
                with patch.object(parent, "build_context", return_value=ctx), \
                     patch.object(parent, "extract_road_patch", return_value=crop.clone()), \
                     patch.object(carrier, "render", return_value=rendered):
                    scales, evidence = dispatch.calibrate_normalizers(
                        *(*args[:3], wrapped, *args[4:]), attack_objective=variant)
            finally:
                for handle in handles:
                    handle.remove()
            ledger.require_complete()
            self.assertEqual({name: len(calls) for name, calls in actual_calls.items()},
                             {"fasterrcnn": 4, "ssd300": 4})
            self.assertEqual(evidence["model_loss_attempts"], 8)
            self.assertEqual(evidence["model_loss_completed"], 8)
            self.assertEqual(evidence["autograd_calls"], 0)
            self.assertFalse(rendered.image.requires_grad)
            for name, adapter in adapters.items():
                image = rendered.image.detach().requires_grad_(True)
                if variant == "C3":
                    expected, _ = objective.frozen_reference().contextual_loss(adapter, image, ctx.object_box)
                else:
                    expected = parent.model_loss(run_config(), {}, name, adapter, ctx,
                                                 image, rendered.support, {}, "synthetic_train").total
                self.assertEqual(scales[name], max(abs(float(expected.detach())), 1e-6))
                for row in evidence["samples"]:
                    self.assertEqual(row["models"][name]["raw_loss"], float(expected.detach()))
                    if variant == "C3":
                        self.assertFalse(row["models"][name]["diagnostics"]["loss_requires_grad"])
                        self.assertEqual(row["models"][name]["diagnostics"]["class_index"], 3)
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_fixed_four_mean_floor_no_grad_no_eot_complete_evidence(self):
        parent, crop, carrier, records, rendered, ctx, args = self.fixtures()
        visited = []

        def loss(*items, attack_objective, phase):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(phase, "calibration")
            self.assertEqual(attack_objective, "C3")
            model_id, sample_id = items[2], items[-1]
            index = [r.sample_id for r in records].index(sample_id)
            value = (.2 * (index + 1)) if model_id == "fasterrcnn" else 0.
            visited.append((sample_id, model_id))
            tensor = torch.tensor(value)
            return dispatch.CapturedLoss(tensor, {"synthetic": tensor})

        with patch.object(parent, "build_context", return_value=ctx), \
             patch.object(parent, "extract_road_patch", return_value=crop.clone()), \
             patch.object(carrier, "render", return_value=rendered) as render, \
             patch.object(dispatch, "model_loss", side_effect=loss):
            scales, evidence = dispatch.calibrate_normalizers(*args, attack_objective="C3")
        self.assertEqual(len(visited), 8)
        self.assertAlmostEqual(scales["fasterrcnn"], .5)
        self.assertEqual(scales["ssd300"], 1e-6)
        self.assertTrue(evidence["models"]["ssd300"]["floor_applied"])
        self.assertFalse(evidence["models"]["fasterrcnn"]["floor_applied"])
        self.assertEqual(evidence["sample_ids"], [r.sample_id for r in records])
        self.assertEqual(evidence["victim_forward_calls"], 8)
        self.assertEqual(evidence["autograd_calls"], 0)
        self.assertTrue(all(call.kwargs["eot"] is None for call in render.call_args_list))
        self.assertTrue(torch.equal(carrier.rgb, crop))
        json.dumps(evidence, allow_nan=False)

    def test_forbidden_split_altered_identity_order_or_count_before_source_read(self):
        parent, _, _, records, _, _, args = self.fixtures()
        alterations = [records[:3], records[::-1],
                       (replace(records[0], split="validation"), *records[1:]),
                       (replace(records[0], split="test"), *records[1:]),
                       (replace(records[0], source_sha256="bad"), *records[1:]),
                       (records[0], records[0], records[2], records[3])]
        with patch.object(parent, "build_context") as build:
            for changed in alterations:
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    current = (*args[:4], changed, *args[5:])
                    dispatch.calibrate_normalizers(*current, attack_objective="C0")
        build.assert_not_called()

    def test_changed_optimized_crop_rejected_before_any_model_forward(self):
        parent, crop, _, _, _, ctx, args = self.fixtures()
        with patch.object(parent, "build_context", return_value=ctx), \
             patch.object(parent, "extract_road_patch", return_value=crop + .01), \
             patch.object(dispatch, "model_loss") as loss, self.assertRaisesRegex(dispatch.CalibrationFailure, "crop"):
            dispatch.calibrate_normalizers(*args, attack_objective="C3")
        loss.assert_not_called()

    def test_nonfinite_calibration_retained_as_failure_not_floored(self):
        parent, crop, carrier, _, rendered, ctx, args = self.fixtures()
        tensor = torch.tensor(float("nan"))
        with patch.object(parent, "build_context", return_value=ctx), \
             patch.object(parent, "extract_road_patch", return_value=crop), \
             patch.object(carrier, "render", return_value=rendered), \
             patch.object(dispatch, "model_loss", return_value=dispatch.CapturedLoss(tensor, {})), \
             self.assertRaisesRegex(dispatch.CalibrationFailure, "non-finite") as failed:
            dispatch.calibrate_normalizers(*args, attack_objective="C0")
        prefix = failed.exception.progress_evidence
        self.assertEqual(prefix["model_loss_attempts"], 1)
        self.assertEqual(prefix["model_loss_completed"], 1)
        self.assertEqual(prefix["current"]["stage"], "validate_returned_loss")
        self.assertEqual(prefix["samples"][0]["models"]["fasterrcnn"]["raw_loss_nonfinite"], "nan")
        json.dumps(prefix, allow_nan=False)

    def test_mid_calibration_failure_keeps_prefix_and_does_not_guess_forward_count(self):
        parent, crop, carrier, _, rendered, ctx, args = self.fixtures()
        tensor = torch.tensor(.4)
        with patch.object(parent, "build_context", return_value=ctx), \
             patch.object(parent, "extract_road_patch", return_value=crop), \
             patch.object(carrier, "render", return_value=rendered), \
             patch.object(dispatch, "model_loss", side_effect=[
                 dispatch.CapturedLoss(tensor, {}), RuntimeError("after a possible forward")]), \
             self.assertRaises(dispatch.CalibrationFailure) as failed:
            dispatch.calibrate_normalizers(*args, attack_objective="C3")
        prefix = failed.exception.progress_evidence
        self.assertEqual(prefix["model_loss_attempts"], 2)
        self.assertEqual(prefix["model_loss_completed"], 1)
        self.assertEqual(prefix["current"]["model_id"], "ssd300")
        self.assertEqual(prefix["current"]["stage"], "model_loss")
        self.assertEqual(prefix["samples"][0]["models"]["fasterrcnn"]["raw_loss"], float(tensor))
        self.assertNotIn("victim_forward_calls", prefix)
        json.dumps(prefix, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
