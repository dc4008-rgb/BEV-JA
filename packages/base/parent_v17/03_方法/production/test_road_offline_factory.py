"""Offline factory structure/guard checks; no pretrained weights or inference."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torchvision.models._api import WeightsEnum
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights, SSD300_VGG16_Weights,
)
from torchvision.ops.misc import FrozenBatchNorm2d

ROOT = Path(__file__).resolve().parents[2]
for directory in ("01_公共数据集", "02_模型"):
    sys.path.insert(0, str(ROOT / directory))

import model_adapters
import road_runtime

MANIFEST = ROOT / "02_模型/model_manifest.json"
ENUMS = {"fasterrcnn": FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
         "ssd300": SSD300_VGG16_Weights.COCO_V1,
         "maskrcnn": MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
         "fcos": FCOS_ResNet50_FPN_Weights.COCO_V1}


class OfflineFactoryTests(unittest.TestCase):
    def mocked_local_state(self, state):
        stack = ExitStack()
        stack.enter_context(patch.object(road_runtime, "checked_file", return_value={"status": "pass"}))
        loader = stack.enter_context(patch.object(torch, "load", return_value=state))
        stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("network access forbidden")))
        return stack, loader

    def test_all_four_factory_bindings_restore_getter(self):
        original = WeightsEnum.get_state_dict
        for model_id, weights in ENUMS.items():
            with self.subTest(model=model_id):
                state = {"synthetic_structural_marker": torch.zeros(1)}
                returned = object()
                def factory(requested_id, device):
                    self.assertEqual(requested_id, model_id)
                    self.assertIs(weights.get_state_dict(progress=False, check_hash=True), state)
                    return returned
                stack, loader = self.mocked_local_state(state)
                with stack, patch.object(model_adapters, "build_detection_adapter", side_effect=factory):
                    actual = road_runtime.build_offline_detector(model_id, torch.device("cpu"), MANIFEST)
                    expected_path, _ = road_runtime.detector_file(model_id, MANIFEST)
                    loader.assert_called_once_with(expected_path, map_location="cpu", weights_only=True)
                    self.assertIs(actual, returned)
                self.assertIs(WeightsEnum.get_state_dict, original)

    def test_unbound_additional_missing_checkpoint_requests_rejected(self):
        original = WeightsEnum.get_state_dict
        def wrong_enum(*_):
            return ENUMS["ssd300"].get_state_dict()
        def duplicate(*_):
            ENUMS["fasterrcnn"].get_state_dict()
            return ENUMS["fasterrcnn"].get_state_dict()
        for factory in (wrong_enum, duplicate, lambda *_: object()):
            with self.subTest(factory=factory.__name__):
                stack, _ = self.mocked_local_state({})
                with stack, patch.object(model_adapters, "build_detection_adapter", side_effect=factory):
                    with self.assertRaises(RuntimeError):
                        road_runtime.build_offline_detector("fasterrcnn", torch.device("cpu"), MANIFEST)
                self.assertIs(WeightsEnum.get_state_dict, original)

    def test_factory_exception_restores_getter(self):
        original = WeightsEnum.get_state_dict
        def failed(*_):
            ENUMS["fasterrcnn"].get_state_dict()
            raise ValueError("synthetic constructor failure")
        stack, _ = self.mocked_local_state({})
        with stack, patch.object(model_adapters, "build_detection_adapter", side_effect=failed):
            with self.assertRaisesRegex(ValueError, "synthetic constructor"):
                road_runtime.build_offline_detector("fasterrcnn", torch.device("cpu"), MANIFEST)
        self.assertIs(WeightsEnum.get_state_dict, original)

    def test_checkpoint_hash_failure_stops_before_loading_or_constructing(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "fake.pth"
            file.write_bytes(b"not a checkpoint")
            with patch.object(road_runtime, "detector_file", return_value=(file, "0" * 64)), \
                    patch.object(torch, "load") as loader, \
                    patch.object(model_adapters, "build_detection_adapter") as factory:
                with self.assertRaisesRegex(RuntimeError, "hash_mismatch"):
                    road_runtime.build_offline_detector("fasterrcnn", torch.device("cpu"), MANIFEST)
                loader.assert_not_called()
                factory.assert_not_called()

    def test_manifest_tampering_stops_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "model_manifest.json"
            file.write_text(MANIFEST.read_text() + "\n")
            with patch.object(torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "reviewed pretrained identities"):
                    road_runtime.build_offline_detector("fasterrcnn", torch.device("cpu"), file)
                loader.assert_not_called()

    def test_real_fasterrcnn_constructor_preserves_frozen_bn_and_eps(self):
        # Execute the actual factory, but stub checkpoint loading: random model
        # parameters are never used for inference/gradients or claimed pretrained.
        torch.set_num_threads(1)
        state = {"synthetic_structural_marker": torch.zeros(1)}
        stack, _ = self.mocked_local_state(state)
        with stack, patch.object(torch.nn.Module, "load_state_dict", return_value=None) as state_loader:
            adapter = road_runtime.build_offline_detector("fasterrcnn", torch.device("cpu"), MANIFEST)
            state_loader.assert_called_once_with(state)
        frozen = [m for m in adapter.model.modules() if isinstance(m, FrozenBatchNorm2d)]
        self.assertEqual(len(frozen), 53)
        self.assertEqual({m.eps for m in frozen}, {0.0})
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in adapter.model.modules()))
        self.assertEqual(tuple(adapter.model.transform.min_size), (320,))
        self.assertEqual(adapter.model.transform.max_size, 1024)
        self.assertFalse(adapter.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.model.parameters()))


if __name__ == "__main__":
    unittest.main()
