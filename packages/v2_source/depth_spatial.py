"""Depth-adapter-only spatial VJPs: native forward, single-thread CPU backward.

Use ``with DepthSpatialScope(model_id): adapter.predict_for_training(image)``.
TorchFunctionMode affects only this thread and lexical forward-call scope; it
does not replace any global torch/F function. The custom nodes survive scope
exit, so the eventual single production backward uses their CPU ATen VJPs.

Supported paths are 4-D nearest/nearest-exact/bilinear/bicubic resize without
antialias, and 4-D reflection/replication padding of the last two dimensions.
No detector, VAE, renderer, loss, forward device or global determinism setting
is changed. Unsupported differentiable resize modes fail closed.
"""
from __future__ import annotations

from collections import Counter
import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.overrides import TorchFunctionMode

_NATIVE_INTERPOLATE = F.interpolate
_NATIVE_PAD = F.pad
_RESIZE_MODES = ("nearest", "nearest-exact", "bilinear", "bicubic")
_MODEL_IDS = ("monodepth2", "dpt_large")


def _require_strict_policy():
    if (torch.get_num_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError("depth CPU spatial VJP requires single-thread strict determinism")


def _check_input(input):
    if input.ndim != 4 or input.dtype not in (torch.float32, torch.float64):
        raise ValueError("depth spatial VJP requires 4-D float32/float64 input")
    if input.device.type not in ("cpu", "cuda"):
        raise ValueError("depth spatial VJP supports CPU validation or CUDA production")


def _scales(scale_factor, recompute_scale_factor):
    if scale_factor is None or recompute_scale_factor:
        return (None, None)
    values = tuple(scale_factor) if isinstance(scale_factor, (tuple, list)) else (scale_factor,) * 2
    if len(values) != 2 or not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
        raise ValueError("depth resize requires two finite positive scales")
    return tuple(float(v) for v in values)


class _ResizeCPUBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias):
        _require_strict_policy()
        _check_input(input)
        if mode not in _RESIZE_MODES or antialias:
            raise ValueError("unsupported differentiable depth resize policy")
        # Exactly the same native forward call, on the original device/dtype.
        output = _NATIVE_INTERPOLATE(input, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)
        ctx.input_size = tuple(input.shape)
        ctx.output_size = tuple(output.shape[-2:])
        ctx.input_device = input.device
        ctx.input_dtype = input.dtype
        ctx.mode = mode
        ctx.align_corners = bool(align_corners)
        ctx.scales = _scales(scale_factor, recompute_scale_factor)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        _require_strict_policy()
        grad = grad_output.detach().to(device="cpu", dtype=ctx.input_dtype).contiguous()
        arguments = (grad, ctx.output_size, ctx.input_size)
        if ctx.mode == "bilinear":
            result = torch.ops.aten.upsample_bilinear2d_backward.default(
                *arguments, ctx.align_corners, *ctx.scales)
        elif ctx.mode == "bicubic":
            result = torch.ops.aten.upsample_bicubic2d_backward.default(
                *arguments, ctx.align_corners, *ctx.scales)
        elif ctx.mode == "nearest":
            result = torch.ops.aten.upsample_nearest2d_backward.default(*arguments, *ctx.scales)
        else:
            result = torch.ops.aten._upsample_nearest_exact2d_backward.default(*arguments, *ctx.scales)
        return result.to(device=ctx.input_device, dtype=ctx.input_dtype), None, None, None, None, None, None


class _PadCPUBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pad, mode, value):
        _require_strict_policy()
        _check_input(input)
        if mode not in ("reflect", "replicate") or len(pad) != 4:
            raise ValueError("depth pad VJP is limited to 2-D reflection/replication")
        output = _NATIVE_PAD(input, pad, mode=mode, value=value)
        ctx.input_size = tuple(input.shape)
        ctx.input_device = input.device
        ctx.input_dtype = input.dtype
        ctx.pad = tuple(pad)
        ctx.mode = mode
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        _require_strict_policy()
        grad = grad_output.detach().to(device="cpu", dtype=ctx.input_dtype).contiguous()
        # Both ATen pad VJPs depend on the input shape, not input values. A zero
        # tensor supplies that shape without retaining every GPU feature map.
        shape_carrier = torch.zeros(ctx.input_size, dtype=ctx.input_dtype, device="cpu")
        operator = (torch.ops.aten.reflection_pad2d_backward.default if ctx.mode == "reflect"
                    else torch.ops.aten.replication_pad2d_backward.default)
        result = operator(grad, shape_carrier, ctx.pad)
        return result.to(device=ctx.input_device, dtype=ctx.input_dtype), None, None, None


def _interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                 recompute_scale_factor=None, antialias=False):
    return _ResizeCPUBackward.apply(input, size, scale_factor, mode, align_corners,
                                   recompute_scale_factor, antialias)


def _pad(input, pad, mode="constant", value=None):
    return _PadCPUBackward.apply(input, pad, mode, value)


class DepthSpatialScope(TorchFunctionMode):
    """A fresh instance surrounds exactly one depth adapter forward call.

    Clean/no-grad forwards use the original operation with no custom node.
    Call ``receipt()`` after leaving the context for JSON audit counters.
    """
    def __init__(self, model_id):
        if model_id not in _MODEL_IDS:
            raise ValueError("depth spatial scope only accepts the two source depth models")
        super().__init__()
        self.model_id = model_id
        self.counts = Counter()

    def __enter__(self):
        _require_strict_policy()
        return super().__enter__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is _NATIVE_INTERPOLATE or func is _NATIVE_PAD:
            input = args[0] if args else kwargs["input"]
            differentiable = torch.is_grad_enabled() and input.requires_grad
            if func is _NATIVE_INTERPOLATE:
                mode = kwargs.get("mode", args[3] if len(args) > 3 else "nearest")
                operation = "interpolate_" + mode
                if differentiable:
                    self.counts[operation + "_cpu_backward"] += 1
                    return _interpolate(*args, **kwargs)
            else:
                mode = kwargs.get("mode", args[2] if len(args) > 2 else "constant")
                operation = "pad_" + mode
                if differentiable and mode in ("reflect", "replicate"):
                    self.counts[operation + "_cpu_backward"] += 1
                    return _pad(*args, **kwargs)
            self.counts[operation + "_native"] += 1
        return func(*args, **kwargs)

    def receipt(self):
        return {"schema": "depth-spatial-native-forward-cpu-vjp-v1", "model_id": self.model_id,
                "scope": "one depth adapter forward; thread-local TorchFunctionMode",
                "forward": "original native operation/device/dtype/arguments",
                "backward": "single-thread CPU ATen; copied to original device/dtype",
                "global_function_monkeypatch": False, "global_determinism_changed": False,
                "nested_autograd_grad_calls": 0, "operation_counts": dict(sorted(self.counts.items()))}
