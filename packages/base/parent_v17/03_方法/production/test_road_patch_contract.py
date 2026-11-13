"""Synthetic CPU contract checks, never real train/test images or model weights."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_公共数据集"))

from advrm_insertion import insert_car, load_car
from carriers import SceneContext
from road_patch_contract import (
    CANONICAL_HW, PATCH_SCHEMA, RENDERER_CONTRACT, RoadRGBView,
    apply_patch_brightness, canonicalize_rgb, load_patch, sample_patch_brightness, save_patch,
)


class RoadPatchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(20260919)
        cls.background = torch.rand(1, 3, 320, 1024, generator=generator)
        car_path = ROOT / "01_公共数据集/car_00.jpg"
        mask_path = ROOT / "01_公共数据集/car_00_mask.jpg"
        expected = {
            car_path: "f0edb7f67cb4b697511d68ae39adc1a53bb0ce0c32a87b1306d16c488ea0f5d7",
            mask_path: "c285e5c88cc48f1fca8194f2c587251192750bc433c701843c5ac57c32b396b1",
        }
        for file, digest in expected.items():
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise RuntimeError("frozen vehicle fixture changed: " + str(file))
        car, mask = load_car(car_path, mask_path, torch.device("cpu"))
        with (ROOT / "01_公共数据集/runtime_v1/train_validation_manifest.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if not {row["split"] for row in rows} <= {"train", "validation"}:
            raise RuntimeError("test data forbidden in fixture manifest")
        cls.contexts = {}
        for domain in ("kitti", "nuscenes"):
            row = next(row for row in rows if row["dataset"] == domain and row["split"] == "train")
            quad = tuple(tuple(point) for point in json.loads(row["road_quad_norm"]))
            benign = insert_car(cls.background, car, mask, quad)
            cls.contexts[domain] = SceneContext(cls.background, benign.image, car, mask,
                                              benign.mask, benign.geometry.box_xyxy, quad)

    def test_training_save_load_render_pixel_exact_both_domains(self):
        raw = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(7))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "patch.pt"
            save_patch(file, raw, {"split": "train", "seed": 7}, raw_generator_rgb=raw)
            restored = load_patch(file)
            self.assertEqual(tuple(restored.shape[-2:]), CANONICAL_HW)
            self.assertTrue(torch.equal(restored, canonicalize_rgb(raw)))
            for domain, context in self.contexts.items():
                with self.subTest(domain=domain):
                    training = RoadRGBView(raw).render(context, training=True)
                    evaluation = RoadRGBView(restored).render(context)
                    self.assertTrue(torch.equal(training.image, evaluation.image))
                    self.assertTrue(torch.equal(training.support, evaluation.support))
                    support = evaluation.support > 0
                    outside = (~support).expand_as(evaluation.image)
                    self.assertTrue(torch.equal(evaluation.image[outside], context.benign[outside]))
                    left, top, right, bottom = context.object_box
                    self.assertTrue(torch.equal(evaluation.image[..., top:bottom, left:right],
                                                context.benign[..., top:bottom, left:right]))
                    self.assertFalse(bool((support & context.object_mask.bool()).any()))

    def test_gradient_survives_downsampling_and_projection(self):
        for domain, context in self.contexts.items():
            with self.subTest(domain=domain):
                raw = torch.full((1, 3, 512, 512), .5, requires_grad=True)
                image = RoadRGBView(raw).render(context).image
                gradient, = torch.autograd.grad(image.square().mean(), raw)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_canonical_input_is_not_resampled(self):
        rgb = torch.rand(1, 3, *CANONICAL_HW)
        self.assertIs(canonicalize_rgb(rgb), rgb)

    def test_invalid_rgb_rejected(self):
        inputs = [torch.zeros(1, 3, 2, 2, dtype=torch.uint8), torch.zeros(3, 4, 5),
                  torch.zeros(2, 3, 4, 5), torch.zeros(1, 4, 4, 5),
                  torch.zeros(1, 3, 0, 5), torch.full((1, 3, 2, 2), float("nan")),
                  torch.full((1, 3, 2, 2), float("inf")), torch.full((1, 3, 2, 2), 1.001),
                  torch.full((1, 3, 2, 2), -.001)]
        for rgb in inputs:
            with self.subTest(shape=rgb.shape, dtype=rgb.dtype), self.assertRaises(ValueError):
                canonicalize_rgb(rgb)

    def test_no_legacy_or_ambiguous_scaled_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "bad.pt"
            payloads = [{"canonical_rgb": torch.zeros(1, 3, 512, 512)},
                        {"schema": PATCH_SCHEMA, "renderer_contract": RENDERER_CONTRACT,
                         "metadata": {}, "canonical_rgb": torch.zeros(1, 3, 512, 512)},
                        {"schema": PATCH_SCHEMA, "renderer_contract": RENDERER_CONTRACT,
                         "metadata": {}, "canonical_rgb": torch.zeros(1, 3, 64, 128).double()},
                        {"schema": PATCH_SCHEMA, "renderer_contract": {},
                         "metadata": {}, "canonical_rgb": torch.zeros(1, 3, 64, 128)}]
            for index, payload in enumerate(payloads):
                torch.save(payload, file)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_patch(file)

    def test_fractional_alpha_below_legacy_half_threshold_rejected_in_box(self):
        context = self.contexts["kitti"]
        # Use a box pixel deliberately absent from the car mask: full box is protected.
        context = replace(context, object_mask=torch.zeros_like(context.object_mask))
        left, top, _, _ = context.object_box
        alpha = torch.zeros_like(context.object_mask)
        alpha[..., top, left] = 1e-8
        with patch("road_patch_contract.perspective_project", return_value=(torch.zeros_like(context.benign), alpha)):
            with self.assertRaisesRegex(RuntimeError, "any nonzero"):
                RoadRGBView(torch.full((1, 3, 64, 128), .5)).render(context)

    def test_mask_outside_box_also_protected(self):
        context = self.contexts["kitti"]
        mask = context.object_mask.clone()
        mask[..., 310, 100] = 1
        alpha = torch.zeros_like(mask)
        alpha[..., 310, 100] = 1e-8
        with patch("road_patch_contract.perspective_project", return_value=(torch.zeros_like(context.benign), alpha)):
            with self.assertRaisesRegex(RuntimeError, "any nonzero"):
                RoadRGBView(torch.full((1, 3, 64, 128), .5)).render(replace(context, object_mask=mask))

    def test_brightness_eot_reproducible_patch_only_and_nonzero_gradient(self):
        rgb = torch.full((1, 3, 64, 128), .5, requires_grad=True)
        first_generator = torch.Generator().manual_seed(13)
        second_generator = torch.Generator().manual_seed(13)
        factors = [sample_patch_brightness(first_generator) for _ in range(2)]
        self.assertEqual(factors, [sample_patch_brightness(second_generator) for _ in range(2)])
        self.assertTrue(all(.9 <= factor <= 1.1 for factor in factors))
        for domain, context in self.contexts.items():
            with self.subTest(domain=domain):
                original = RoadRGBView(rgb).render(context)
                transformed = RoadRGBView(apply_patch_brightness(rgb, factors[0])).render(context)
                direct = RoadRGBView(rgb).render(context, training=True, eot={"brightness_factor": [.9, 1.1]},
                                                 generator=torch.Generator().manual_seed(13))
                self.assertTrue(torch.equal(direct.image, transformed.image))
                self.assertTrue(torch.equal(original.support, transformed.support))
                self.assertFalse(torch.equal(original.image, transformed.image))
                outside = (transformed.support == 0).expand_as(transformed.image)
                self.assertTrue(torch.equal(transformed.image[outside], context.benign[outside]))
                gradient, = torch.autograd.grad(transformed.image.mean(), rgb)
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_eval_eot_or_geometry_eot_rejected(self):
        view = RoadRGBView(torch.full((1, 3, 64, 128), .5))
        for kwargs in ({"eot": {"brightness_factor": [.9, 1.1]}},
                       {"eot": {"perspective_vertex": [-.01, .01]}, "training": True}):
            with self.assertRaises(ValueError):
                view.render(self.contexts["kitti"], **kwargs)

    def test_raw_provenance_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_patch(Path(directory) / "bad.pt", torch.zeros(1, 3, 64, 128),
                           raw_generator_rgb=torch.ones(1, 3, 512, 512))


if __name__ == "__main__":
    unittest.main()
