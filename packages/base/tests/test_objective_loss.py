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
from training import ssd_objectives as objective
from training import inputs
from training import evaluate_validation as evaluation
import importlib.util
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



class FixedLossTests(unittest.TestCase):
    def test_native_ssd_source_is_byte_identical_to_frozen_v21(self):
        self.assertEqual((ROOT/'training/ssd_objectives.py').read_bytes(),
                         (ROOT/'reference_v21/training/ssd_objectives.py').read_bytes())

    def reference(self):
        name='_v25_reference_v21_loss'
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,ROOT/'reference_v21/training/objective_loss.py')
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return sys.modules[name]

    def args(self,model,adapter,image):
        return (run_config(),{},model,adapter,context(),image,torch.ones_like(image[:,:1]),{},'synthetic')

    def test_u1_matches_original_v21_loss_and_input_gradient_exactly(self):
        for model in ('fasterrcnn','ssd300'):
            adapter=tiny_adapter(model); image=source_image()
            expected=self.reference().model_loss(*self.args(model,adapter,image),ssd_objective=inputs.MATHEMATICAL_OBJECTIVE,phase='train_pre')
            gradient=torch.autograd.grad(expected.total,image,retain_graph=True)[0]
            for variant in ('U1',):
                with self.subTest(model=model,variant=variant):
                    actual=dispatch.model_loss(*self.args(model,adapter,image),update_variant=variant,phase='train_pre')
                    self.assertTrue(torch.equal(actual.total,expected.total))
                    self.assertEqual(actual.scalars(),expected.scalars())
                    self.assertTrue(torch.equal(torch.autograd.grad(actual.total,image,retain_graph=True)[0],gradient))
                    evaluation.validate_objective_diagnostics(actual.diagnostics,model)
                    self.assertEqual(actual.diagnostics['training_target_classes'],[3,6,8])
            self.assertEqual(adapter.calls,2)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in adapter.model.parameters()))

    def test_all_phases_use_one_counted_native_predict_and_no_hidden_forward(self):
        for model in ('fasterrcnn','ssd300'):
            adapter=tiny_adapter(model)
            before={k:v.clone() for k,v in adapter.model.state_dict().items()}
            for phase in ('calibration','train_pre','train_post'):
                image=source_image(phase=='train_pre');ledger=ledger_for(phase)
                wrapped=CountingAdapter(adapter,ledger)
                with torch.set_grad_enabled(phase=='train_pre'):
                    result=dispatch.model_loss(*self.args(model,wrapped,image),update_variant='U1',phase=phase)
                ledger.require_complete()
                self.assertEqual(result.total.requires_grad,phase=='train_pre')
                evaluation.validate_objective_diagnostics(result.diagnostics,model)
                for module in adapter.model.modules():self.assertFalse(module._forward_hooks)
            self.assertEqual(adapter.calls,3)
            self.assertTrue(all(torch.equal(before[k],v) for k,v in adapter.model.state_dict().items()))

    def test_c1_still_means_ssd_only_and_vehicle_group_not_car_only(self):
        # Independent formula verifies all three classes and unchanged FR branch.
        logits=torch.tensor([[0.,0.,0.,1.,0.,0.,2.,0.,3.]],requires_grad=True)
        anchors=torch.tensor([[0.,0.,10.,10.]])
        loss,_,stats=objective.native_matched_vehicle_objective(logits,anchors,anchors,
            objective.SSDMatcher(.5),[1],require_attack_graph=True,phase='train_pre')
        expected=logits.softmax(-1)[0,[3,6,8]].sum()
        self.assertTrue(torch.equal(loss,expected))
        self.assertGreater(float(loss.detach()),float(logits.softmax(-1)[0,3].detach()))
        adapter=tiny_adapter('fasterrcnn');image=source_image()
        actual=dispatch.model_loss(*self.args('fasterrcnn',adapter,image),update_variant='U1',phase='train_pre')
        original=dispatch._parent().model_loss(*self.args('fasterrcnn',adapter,image))
        self.assertTrue(torch.equal(actual.total,original.total))
        self.assertEqual(actual.diagnostics['loss_branch'],'C0_post_nms')

    def test_no_grad_preupdate_and_wrong_variant_are_rejected(self):
        adapter=tiny_adapter()
        with torch.no_grad(),self.assertRaises(RuntimeError):
            dispatch.model_loss(*self.args('ssd300',adapter,source_image()),update_variant='U1',phase='train_pre')
        with self.assertRaises(ValueError):
            dispatch.model_loss(*self.args('ssd300',adapter,source_image()),update_variant='C3',phase='train_pre')
        self.assertEqual(adapter.calls,0)

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

    def test_fixed_four_mean_floor_no_grad_no_eot_complete_evidence(self):
        parent, crop, carrier, records, rendered, ctx, args = self.fixtures()
        visited = []

        def loss(*items, update_variant, phase):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(phase, "calibration")
            self.assertEqual(update_variant, "U1")
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
            scales, evidence = dispatch.calibrate_normalizers(*args, update_variant="U1")
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
                    dispatch.calibrate_normalizers(*current, update_variant="U1")
        build.assert_not_called()

    def test_changed_optimized_crop_rejected_before_any_model_forward(self):
        parent, crop, _, _, _, ctx, args = self.fixtures()
        with patch.object(parent, "build_context", return_value=ctx), \
             patch.object(parent, "extract_road_patch", return_value=crop + .01), \
             patch.object(dispatch, "model_loss") as loss, self.assertRaisesRegex(dispatch.CalibrationFailure, "crop"):
            dispatch.calibrate_normalizers(*args, update_variant="U1")
        loss.assert_not_called()

    def test_nonfinite_calibration_retained_as_failure_not_floored(self):
        parent, crop, carrier, _, rendered, ctx, args = self.fixtures()
        tensor = torch.tensor(float("nan"))
        with patch.object(parent, "build_context", return_value=ctx), \
             patch.object(parent, "extract_road_patch", return_value=crop), \
             patch.object(carrier, "render", return_value=rendered), \
             patch.object(dispatch, "model_loss", return_value=dispatch.CapturedLoss(tensor, {})), \
             self.assertRaisesRegex(dispatch.CalibrationFailure, "non-finite") as failed:
            dispatch.calibrate_normalizers(*args, update_variant="U1")
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
            dispatch.calibrate_normalizers(*args, update_variant="U1")
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
