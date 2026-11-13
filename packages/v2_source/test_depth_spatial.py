"""CPU semantic/bitwise checks and optional CUDA native-forward checks."""
from __future__ import annotations

import json
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from depth_spatial import DepthSpatialScope


class DepthSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)

    def check_operation(self, function, shape=(1, 2, 4, 5), dtype=torch.float64):
        x = (torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape) / 17).requires_grad_(True)
        reference = function(x)
        # Fixed nonuniform output VJP; not a trivial sum-only gradient check.
        upstream = torch.linspace(-.7, .9, reference.numel(), dtype=dtype).reshape(reference.shape)
        expected, = torch.autograd.grad(reference, x, upstream)
        y = x.detach().clone().requires_grad_(True)
        with DepthSpatialScope("dpt_large") as scope:
            actual = function(y)
        with mock.patch("torch.autograd.grad", wraps=torch.autograd.grad) as grad:
            observed, = torch.autograd.grad(actual, y, upstream)
            self.assertEqual(grad.call_count, 1, "custom backward used a nested autograd.grad")
        self.assertTrue(torch.equal(reference.detach().contiguous().view(torch.uint8),
                                    actual.detach().contiguous().view(torch.uint8)))
        self.assertTrue(torch.equal(expected.contiguous().view(torch.uint8), observed.contiguous().view(torch.uint8)))
        self.assertTrue(scope.counts)
        json.dumps(scope.receipt(), allow_nan=False)
        return scope

    def test_all_resize_modes_size_scale_and_corner_contracts(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("nearest", "nearest-exact", "bilinear", "bicubic"):
                aligns = (None,) if mode.startswith("nearest") else (False, True)
                for align in aligns:
                    for settings in ({"size": (3, 8)}, {"scale_factor": (1.7, .8)},
                                     {"scale_factor": (1.7, .8), "recompute_scale_factor": True}):
                        with self.subTest(dtype=dtype, mode=mode, align=align, settings=settings):
                            self.check_operation(lambda x: F.interpolate(x, mode=mode, align_corners=align, **settings), dtype=dtype)

    def test_reflection_replication_and_module_padding(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("reflect", "replicate"):
                with self.subTest(dtype=dtype, mode=mode):
                    self.check_operation(lambda x: F.pad(x, (1, 2, 1, 0), mode=mode), dtype=dtype)
            self.check_operation(torch.nn.ReflectionPad2d(1), dtype=dtype)

    def test_composed_depth_path_gradient_and_repeatability(self):
        def forward(x):
            x = F.interpolate(x, size=(7, 8), mode="bicubic", align_corners=False)
            x = torch.nn.ReflectionPad2d(1)(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            return F.interpolate(x, size=(6, 7), mode="bilinear", align_corners=True)
        first = self.check_operation(forward)
        second = self.check_operation(forward)
        self.assertEqual(first.receipt(), second.receipt())
        self.assertEqual(sum(first.counts.values()), 4)

    def test_gradcheck_first_derivative(self):
        x = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4).div(20).requires_grad_(True)
        def forward(value):
            with DepthSpatialScope("monodepth2"):
                return F.pad(F.interpolate(value, size=(4, 5), mode="bicubic", align_corners=False),
                             (1, 0, 0, 1), mode="reflect")
        self.assertTrue(torch.autograd.gradcheck(forward, (x,), eps=1e-6, atol=1e-6, rtol=1e-5))

    def test_scope_preserves_globals_and_no_grad_path(self):
        original_interpolate, original_pad = F.interpolate, F.pad
        x = torch.ones(1, 1, 3, 4, requires_grad=True)
        with torch.no_grad(), DepthSpatialScope("monodepth2") as scope:
            out = F.interpolate(x, scale_factor=2, mode="nearest")
            F.pad(out, (1, 1, 1, 1), mode="reflect")
        self.assertIs(F.interpolate, original_interpolate)
        self.assertIs(F.pad, original_pad)
        self.assertFalse(out.requires_grad)
        self.assertEqual(scope.counts, {"interpolate_nearest_native": 1, "pad_reflect_native": 1})
        outside = F.interpolate(x, scale_factor=2, mode="nearest")
        self.assertNotIn("ResizeCPUBackward", type(outside.grad_fn).__name__)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_exceptions_restore_scope_and_unsupported_modes_fail_closed(self):
        x = torch.ones(1, 1, 3, 4, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "unsupported differentiable"):
            with DepthSpatialScope("dpt_large"):
                F.interpolate(x, size=(5, 6), mode="bilinear", align_corners=False, antialias=True)
        outside = F.interpolate(x, size=(5, 6), mode="bilinear", align_corners=False)
        self.assertNotIn("ResizeCPUBackward", type(outside.grad_fn).__name__)
        with self.assertRaises(ValueError):
            DepthSpatialScope("fasterrcnn")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_matches_native_bits_and_cpu_vjp(self):
        cpu = torch.linspace(-.3, .8, 20, dtype=torch.float32).reshape(1, 1, 4, 5).requires_grad_(True)
        for mode in ("bicubic", "bilinear"):
            reference_cpu = F.interpolate(cpu, size=(7, 9), mode=mode, align_corners=False)
            upstream = torch.linspace(-1., 1., 63).reshape(1, 1, 7, 9)
            expected, = torch.autograd.grad(reference_cpu, cpu, upstream)
            x = cpu.detach().cuda().requires_grad_(True)
            with torch.no_grad():
                native = F.interpolate(x, size=(7, 9), mode=mode, align_corners=False)
            with DepthSpatialScope("dpt_large"):
                output = F.interpolate(x, size=(7, 9), mode=mode, align_corners=False)
            observed, = torch.autograd.grad(output, x, upstream.cuda())
            self.assertTrue(torch.equal(native.view(torch.uint8), output.detach().view(torch.uint8)))
            self.assertTrue(torch.equal(expected.view(torch.uint8), observed.cpu().view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
