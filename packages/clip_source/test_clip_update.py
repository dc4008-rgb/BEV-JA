"""CPU contract tests; no model loading, training, or CUDA equivalence claim."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import torch

from clip_update import clipped_scaled_update


_HISTORICAL_PATH = Path(__file__).resolve().parent.parent / "meeting_full_repeat_20260924_r2" / "scale_update.py"
_SPEC = importlib.util.spec_from_file_location("historical_scale_update_for_clip_test", _HISTORICAL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HISTORICAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HISTORICAL)


class ClipUpdateTests(unittest.TestCase):
    def assert_historical_equal(self, dtype: torch.dtype, max_norm: float | None) -> None:
        latent = torch.linspace(-2, 2, 512).reshape(1, 2, 16, 16).to(dtype)
        gradient = torch.linspace(-0.00017, 0.00013, 512).reshape_as(latent)
        old_candidate, old_tensors, _ = _HISTORICAL.scaled_update(latent, gradient, 100000.0, 0.05)
        candidate, tensors, metadata = clipped_scaled_update(latent, gradient, max_norm=max_norm)
        self.assertTrue(torch.equal(candidate, old_candidate))
        for key, tensor in old_tensors.items():
            self.assertTrue(torch.equal(tensors[key], tensor), key)
        self.assertFalse(metadata["clipped"])
        self.assertEqual(metadata["clip_scale"], 1.0)
        self.assertEqual(tensors["amplified_gradient"].data_ptr(), tensors["clipped_gradient"].data_ptr())

    def test_off_float32_exact_historical(self):
        self.assert_historical_equal(torch.float32, None)

    def test_off_bfloat16_exact_historical(self):
        self.assert_historical_equal(torch.bfloat16, None)

    def test_inactive_float32_exact_historical(self):
        self.assert_historical_equal(torch.float32, 10000.0)

    def test_inactive_bfloat16_exact_historical(self):
        self.assert_historical_equal(torch.bfloat16, 10000.0)

    def test_known_vector_and_after_multiplier_threshold(self):
        candidate, tensors, meta = clipped_scaled_update(torch.tensor([5., 5.]), torch.tensor([3., 4.]), coefficient=2, lr=.5, max_norm=5)
        self.assertTrue(torch.equal(tensors["amplified_gradient"], torch.tensor([6., 8.])))
        self.assertTrue(torch.equal(tensors["clipped_gradient"], torch.tensor([3., 4.])))
        self.assertTrue(torch.equal(candidate, torch.tensor([3.5, 3.])))
        self.assertEqual(meta["norm_before"], 10.)
        self.assertEqual(meta["norm_after"], 5.)
        self.assertEqual(meta["clip_scale"], .5)
        self.assertTrue(meta["clipped"])

    def test_global_tensor_norm_not_per_channel_or_element_clip(self):
        latent = torch.zeros(2, 2)
        gradient = torch.full((2, 2), 3.)
        _, tensors, meta = clipped_scaled_update(latent, gradient, coefficient=1, lr=1, max_norm=3)
        self.assertTrue(torch.equal(tensors["clipped_gradient"], torch.full((2, 2), 1.5)))
        self.assertEqual(meta["norm_before"], 6.)
        self.assertEqual(meta["norm_after"], 3.)
        self.assertIsNone(meta["realized_to_latent_rms_ratio"])

    def test_equal_threshold_is_inactive(self):
        _, tensors, meta = clipped_scaled_update(torch.ones(2), torch.tensor([3., 4.]), coefficient=1, max_norm=5)
        self.assertFalse(meta["clipped"])
        self.assertTrue(torch.equal(tensors["amplified_gradient"], tensors["clipped_gradient"]))

    def test_zero_threshold_cancels_only_update(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                latent = torch.tensor([1., -2.], dtype=dtype)
                candidate, tensors, meta = clipped_scaled_update(latent, torch.tensor([3., -4.]), max_norm=0)
                self.assertTrue(torch.equal(candidate, latent))
                self.assertEqual(torch.count_nonzero(tensors["proposed_update"]).item(), 0)
                self.assertEqual(meta["norm_after"], 0.)
                self.assertEqual(meta["clip_scale"], 0.)
                self.assertTrue(meta["clipped"])
                self.assertEqual(meta["changed_count"], 0)

    def test_zero_gradient_zero_threshold_stays_inactive(self):
        _, _, meta = clipped_scaled_update(torch.ones(2), torch.zeros(2), max_norm=0)
        self.assertFalse(meta["clipped"])
        self.assertEqual(meta["clip_scale"], 1.)
        self.assertEqual(meta["norm_before"], 0.)

    def test_zero_endpoint_handles_fp32_norm_underflow(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                latent = torch.zeros(2, dtype=dtype)
                gradient = torch.tensor([1e-35, -1e-35], dtype=dtype)
                candidate, tensors, meta = clipped_scaled_update(latent, gradient, max_norm=0.)
                self.assertGreater(torch.count_nonzero(tensors['amplified_gradient']).item(), 0)
                self.assertEqual(meta['norm_before'], 0.)
                self.assertTrue(meta['clipped'])
                self.assertTrue(torch.equal(candidate, latent))
                self.assertEqual(meta['clip_scale'], 0.)

    def test_active_bfloat16_uses_fp32_scale_then_cast(self):
        latent = torch.tensor([1., 2., 3., 4.], dtype=torch.bfloat16)
        gradient = torch.tensor([.00113, -.00279, .00319, -.00731])
        _, tensors, meta = clipped_scaled_update(latent, gradient, max_norm=17.3)
        amplified = 100000. * gradient.to(latent)
        expected_scale = torch.tensor(17.3, dtype=torch.float32) / torch.linalg.vector_norm(amplified.float())
        expected = (amplified.float() * expected_scale).to(torch.bfloat16)
        self.assertTrue(torch.equal(tensors["clipped_gradient"], expected))
        self.assertEqual(meta["clip_scale"], expected_scale.item())
        self.assertEqual(meta["norm_after"], torch.linalg.vector_norm(expected.float()).item())

    def test_inputs_rng_and_autograd_unchanged(self):
        latent = torch.tensor([1., 2.], requires_grad=True)
        gradient = torch.tensor([3., 4.], requires_grad=True)
        before_l, before_g = latent.detach().clone(), gradient.detach().clone()
        rng = torch.random.get_rng_state().clone()
        for threshold in (None, 1., 1e9, 0.):
            candidate, tensors, _ = clipped_scaled_update(latent, gradient, max_norm=threshold)
            self.assertTrue(torch.equal(latent, before_l))
            self.assertTrue(torch.equal(gradient, before_g))
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertNotEqual(candidate.data_ptr(), latent.data_ptr())
            for tensor in tensors.values():
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)
        self.assertIsNone(latent.grad)
        self.assertIsNone(gradient.grad)

    def test_metadata_rms_and_json(self):
        _, tensors, meta = clipped_scaled_update(torch.tensor([3., 4.]), torch.tensor([1., 0.]), coefficient=2., lr=.5, max_norm=1.)
        self.assertAlmostEqual(meta["latent_rms"], 5. / 2 ** .5, places=6)
        self.assertAlmostEqual(meta["proposed_update_rms"], .5 / 2 ** .5, places=6)
        self.assertAlmostEqual(meta["realized_update_rms"], .5 / 2 ** .5, places=6)
        self.assertAlmostEqual(meta["realized_to_latent_rms_ratio"], .1, places=6)
        self.assertEqual(meta["changed_count"], 1)
        self.assertEqual(meta["unchanged_count"], 1)
        self.assertEqual(meta["changed_fraction"], .5)
        self.assertEqual(meta["clip_stage"], "after_coefficient_before_lr")
        self.assertEqual(meta["norm_accumulation_dtype"], "torch.float32")
        self.assertTrue(torch.equal(tensors["realized_delta_float32"], torch.tensor([.5, 0.])))
        json.dumps(meta, allow_nan=False)

    def test_noncontiguous_and_scalar_tensors(self):
        latent = torch.ones(2, 3).T
        gradient = torch.arange(6.).reshape(2, 3).T
        _, tensors, meta = clipped_scaled_update(latent, gradient, coefficient=1., max_norm=2.)
        self.assertEqual(tensors["clipped_gradient"].shape, latent.shape)
        self.assertTrue(meta["clipped"])
        candidate, _, _ = clipped_scaled_update(torch.tensor(2.), torch.tensor(3.), coefficient=1., lr=1., max_norm=1.)
        self.assertEqual(candidate.item(), 1.)

    def test_invalid_thresholds(self):
        for value in (True, False, "1", torch.tensor(1.), complex(1, 0)):
            with self.subTest(value=repr(value)), self.assertRaises(TypeError):
                clipped_scaled_update(torch.ones(1), torch.ones(1), max_norm=value)
        for value in (-1., float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                clipped_scaled_update(torch.ones(1), torch.ones(1), max_norm=value)

    def test_invalid_coefficient_and_lr(self):
        for parameter in ("coefficient", "lr"):
            for value in (True, "1", torch.tensor(1.)):
                with self.subTest(parameter=parameter, value=repr(value)), self.assertRaises(TypeError):
                    clipped_scaled_update(torch.ones(1), torch.ones(1), **{parameter: value})
            for value in (0., -1., float("nan"), float("inf")):
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    clipped_scaled_update(torch.ones(1), torch.ones(1), **{parameter: value})

    def test_invalid_tensor_inputs(self):
        invalid = ([1.], torch.ones(1, dtype=torch.int64), torch.ones(1, dtype=torch.complex64), torch.eye(2).to_sparse())
        for value in invalid:
            for slot in (0, 1):
                args = [torch.ones(1), torch.ones(1)]
                args[slot] = value
                with self.subTest(slot=slot, value=repr(value)), self.assertRaises(TypeError):
                    clipped_scaled_update(*args)

    def test_empty_shape_meta_and_nonfinite_rejected(self):
        cases = [(torch.empty(0), torch.empty(0)), (torch.ones(2), torch.ones(3)), (torch.ones(1, device="meta"), torch.ones(1))]
        for args in cases:
            with self.subTest(args=repr(args)), self.assertRaises(ValueError):
                clipped_scaled_update(*args)
        for value in (float("nan"), float("inf"), -float("inf")):
            for slot in (0, 1):
                args = [torch.ones(1), torch.ones(1)]
                args[slot] = torch.tensor([value])
                with self.subTest(slot=slot, value=value), self.assertRaises(ValueError):
                    clipped_scaled_update(*args)

    def test_cast_and_amplification_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, "cast_gradient"):
            clipped_scaled_update(torch.ones(1, dtype=torch.float16), torch.tensor([1e10]))
        with self.assertRaisesRegex(ValueError, "amplified_gradient"):
            clipped_scaled_update(torch.ones(1), torch.tensor([1e38]), max_norm=1.)

    def test_fp32_norm_overflow_rejected_not_hidden_by_clip(self):
        with self.assertRaisesRegex(ValueError, "norm_before"):
            clipped_scaled_update(torch.ones(4), torch.full((4,), 1e20), coefficient=1., max_norm=1.)

    def test_update_and_candidate_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, "proposed_update"):
            clipped_scaled_update(torch.ones(1), torch.ones(1), coefficient=1., lr=1e39)
        with self.assertRaisesRegex(ValueError, "candidate_latent"):
            clipped_scaled_update(torch.tensor([-3e38]), torch.ones(1), coefficient=1., lr=3e38)


if __name__ == "__main__":
    unittest.main(verbosity=2)
