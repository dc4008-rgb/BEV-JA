"""Synthetic CPU structural tests; random tiny detectors, not attack efficacy.

Uses actual torchvision FasterRCNN/SSD native heads and losses without weights,
downloads, datasets, or access to any test split.
"""
from collections import OrderedDict
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.ops import MultiScaleRoIAlign

from road_detection_objectives import (
    contextual_loss, frozen_detector_branch, make_robust_targets, robust_loss,
)


class TinyBackbone(nn.Module):
    def __init__(self, ssd=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=4, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(0.3)
        self.out_channels = [8] if ssd else 8

    def forward(self, value):
        return OrderedDict([("0", self.dropout(self.bn(self.conv(value))).relu())])


def tiny_adapter(model_id):
    torch.manual_seed(19)
    if model_id == "fasterrcnn":
        model = FasterRCNN(
            TinyBackbone(), num_classes=5, min_size=64, max_size=64,
            rpn_anchor_generator=AnchorGenerator(((8, 16, 32),), ((0.5, 1.0, 2.0),)),
            box_roi_pool=MultiScaleRoIAlign(["0"], output_size=3, sampling_ratio=2),
            rpn_pre_nms_top_n_train=40, rpn_pre_nms_top_n_test=30,
            rpn_post_nms_top_n_train=20, rpn_post_nms_top_n_test=15,
            rpn_batch_size_per_image=32, box_batch_size_per_image=16,
        )
    else:
        model = SSD(TinyBackbone(ssd=True), DefaultBoxGenerator([[2]], scales=[0.25, 0.5]),
                    size=(64, 64), num_classes=5)
    model.eval()
    return SimpleNamespace(model_id=model_id, model=model)


class DetectorObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20260919)
        self.image = (0.2 + 0.6 * torch.rand(1, 3, 64, 64)).requires_grad_()
        self.box = [22., 20., 42., 44.]
        self.prediction = [{
            "boxes": torch.tensor([[21., 20., 42., 44.], [2., 2., 9., 9.], [45., 2., 58., 12.]]),
            "scores": torch.tensor([0.9, 0.7, 0.2]),
            "labels": torch.tensor([3, 1, 4]),
        }]

    def test_fixed_targets_keep_background_and_replace_target(self):
        targets = make_robust_targets(self.prediction, self.box, "cpu")
        self.assertEqual(targets[0]["labels"].tolist(), [1, 3])
        torch.testing.assert_close(targets[0]["boxes"], torch.tensor([[2., 2., 9., 9.], self.box]))
        self.prediction[0]["boxes"].fill_(0)
        self.assertTrue(bool((targets[0]["boxes"][:, 2:] > targets[0]["boxes"][:, :2]).all()))
        self.assertFalse(targets[0]["boxes"].requires_grad)

    def test_empty_clean_predictions_still_have_known_car(self):
        targets = make_robust_targets({"boxes": torch.empty(0, 4), "scores": torch.empty(0),
                                       "labels": torch.empty(0, dtype=torch.long)}, self.box, "cpu")
        self.assertEqual(targets[0]["labels"].tolist(), [3])

    def test_invalid_clean_predictions_fail(self):
        self.prediction[0]["scores"][0] = float("nan")
        with self.assertRaises(ValueError):
            make_robust_targets(self.prediction, self.box, "cpu")

    def test_real_fasterrcnn_contextual(self):
        self._check_contextual("fasterrcnn")

    def test_real_ssd_contextual(self):
        self._check_contextual("ssd300")

    def _check_contextual(self, model_id):
        adapter = tiny_adapter(model_id)
        loss, stats = contextual_loss(adapter, self.image, self.box)
        self.assertGreater(stats["matched_prediction_count"], 0)
        self.assertEqual(stats["backbone_forward_calls"], 1)
        self.assertEqual(stats["direction"], "minimize")
        self.assertGreaterEqual(float(loss), 0.)
        self.assertLessEqual(float(loss), 1.)
        gradient = torch.autograd.grad(loss, self.image)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
        # Actual one-gradient descent lowers this objective on this synthetic
        # fixture. This is a mathematical direction test, not road efficacy.
        moved = (self.image.detach() - 1e-4 * gradient.sign()).requires_grad_()
        next_loss, _ = contextual_loss(adapter, moved, self.box)
        self.assertLess(float(next_loss), float(loss))

    def test_real_fasterrcnn_native_losses(self):
        self._check_native("fasterrcnn")

    def test_real_ssd_native_losses(self):
        self._check_native("ssd300")

    def _check_native(self, model_id):
        adapter = tiny_adapter(model_id)
        targets = make_robust_targets(self.prediction, self.box, "cpu")
        before = {name: value.clone() for name, value in adapter.model.named_buffers()}
        # Mixed modes and mixed parameter flags must survive the attack call.
        adapter.model.backbone.bn.training = True
        next(adapter.model.parameters()).requires_grad_(False)
        modes = [module.training for module in adapter.model.modules()]
        flags = [parameter.requires_grad for parameter in adapter.model.parameters()]
        seen = []
        hook = adapter.model.backbone.register_forward_pre_hook(
            lambda module, inputs: seen.append((module.bn.training, module.dropout.training)))
        torch.manual_seed(111)
        loss, stats = robust_loss(adapter, self.image, targets)
        hook.remove()
        self.assertEqual(seen, [(False, False)])
        self.assertEqual(modes, [module.training for module in adapter.model.modules()])
        self.assertEqual(flags, [parameter.requires_grad for parameter in adapter.model.parameters()])
        for name, value in adapter.model.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        gradient = torch.autograd.grad(loss, self.image)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
        self.assertEqual(stats["direction"], "maximize")
        expected_keys = ({"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
                         if model_id == "fasterrcnn" else {"bbox_regression", "classification"})
        self.assertEqual(set(stats["loss_components"]), expected_keys)
        # Independent direct native-forward reference, same RPN sampler seed.
        torch.manual_seed(111)
        with frozen_detector_branch(adapter.model, native_losses=True):
            reference = sum(adapter.model([self.image[0]], targets).values())
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        moved = (self.image.detach() + 1e-4 * gradient.sign()).requires_grad_()
        torch.manual_seed(111)
        next_loss, _ = robust_loss(adapter, moved, targets)
        self.assertGreater(float(next_loss), float(loss))

    def test_frozen_context_restores_after_exception(self):
        adapter = tiny_adapter("fasterrcnn")
        adapter.model.train()
        adapter.model.backbone.dropout.eval()
        modes = [m.training for m in adapter.model.modules()]
        flags = [p.requires_grad for p in adapter.model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with frozen_detector_branch(adapter.model, native_losses=True):
                raise RuntimeError("fixture")
        self.assertEqual(modes, [m.training for m in adapter.model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in adapter.model.parameters()])

    def test_detached_image_and_unsupported_model_fail(self):
        adapter = tiny_adapter("ssd300")
        with self.assertRaises(RuntimeError):
            contextual_loss(adapter, self.image.detach(), self.box)
        adapter.model_id = "fcos"
        with self.assertRaises(ValueError):
            contextual_loss(adapter, self.image, self.box)

    def test_native_labels_must_be_fixed(self):
        targets = make_robust_targets(self.prediction, self.box, "cpu")
        targets[0]["boxes"].requires_grad_(True)
        with self.assertRaises(ValueError):
            robust_loss(tiny_adapter("ssd300"), self.image, targets)


if __name__ == "__main__":
    unittest.main()
