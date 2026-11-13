from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch as mock_patch

ROOT = Path(__file__).resolve().parents[2]
for part in ("03_方法/production", "01_公共数据集", "02_模型"):
    sys.path.insert(0, str(ROOT / part))
import torch
from road_runtime import (patch_update, patch_brightness, training_records, checked_file,
                          build_offline_detector, preflight, CORE)


class RoadRuntimeTest(unittest.TestCase):
    def test_opposite_optimization_directions(self):
        # A convex scalar objective exposes accidental sign/optimizer swapping.
        for method in ("contextual", "robust_dpatch"):
            value = torch.nn.Parameter(torch.full((1, 3, 4, 8), .5))
            before = float(value.square().sum())
            grad = torch.autograd.grad(value.square().sum(), value)[0]
            optimizer = torch.optim.Adam([value], lr=.01) if method == "contextual" else None
            patch_update(value, grad, method, .01, optimizer)
            after = float(value.detach().square().sum())
            self.assertLess(after, before) if method == "contextual" else self.assertGreater(after, before)

    def test_projection_and_failure_are_explicit(self):
        value = torch.nn.Parameter(torch.ones(1, 3, 4, 8))
        patch_update(value, torch.ones_like(value), "robust_dpatch", .1)
        self.assertTrue(torch.equal(value, torch.ones_like(value)))
        for grad in (torch.zeros_like(value), torch.full_like(value, float("nan"))):
            with self.assertRaises(RuntimeError):
                patch_update(value, grad, "robust_dpatch", .1)

    def test_eot_reproducibility_and_gradient(self):
        value = torch.full((1, 3, 64, 128), .4, requires_grad=True)
        a, factor_a = patch_brightness(value, torch.Generator().manual_seed(22))
        b, factor_b = patch_brightness(value, torch.Generator().manual_seed(22))
        self.assertTrue(torch.equal(a, b)); self.assertEqual(factor_a, factor_b)
        self.assertGreaterEqual(factor_a, .9); self.assertLessEqual(factor_a, 1.1)
        self.assertGreater(float(torch.autograd.grad(a.sum(), value)[0].abs().sum()), 0)

    def test_split_identity_and_explicit_prefix_remap(self):
        manifest = ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv"
        records = training_records(manifest)
        self.assertEqual(len(records), 240)
        self.assertEqual({r.split for r in records}, {"train", "validation"})
        mapped = training_records(manifest, Path("/root"), Path("/alternate"))
        self.assertEqual([r.source_sha256 for r in records], [r.source_sha256 for r in mapped])
        self.assertTrue(all(str(r.source_path).startswith("/alternate/") for r in mapped))
        with self.assertRaises(ValueError):
            training_records(manifest, Path("/root"), None)
        with self.assertRaises(ValueError):
            training_records(Path("missing_test_manifest.csv"))
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / "train.csv"
            altered.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                training_records(altered)

    def test_no_implicit_model_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Loader must reject absent bytes before constructing a model or
            # asking torchvision to retrieve a pretrained checkpoint.
            with mock_patch("torch.hub.download_url_to_file", side_effect=AssertionError("network attempted")):
                with self.assertRaises(RuntimeError):
                    build_offline_detector("fasterrcnn", torch.device("cpu"),
                                           ROOT / "02_模型/model_manifest.json", Path(tmp))

    def test_hash_mismatch_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset"
            self.assertEqual(checked_file(path, "0"*64)["status"], "missing")
            path.write_bytes(b"incorrect")
            self.assertEqual(checked_file(path, "0"*64)["status"], "hash_mismatch")


if __name__ == "__main__":
    unittest.main()
