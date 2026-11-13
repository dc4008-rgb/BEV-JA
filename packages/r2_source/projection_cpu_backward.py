"""Narrow road RGB projection fix candidate: native forward, CPU ATen backward.

Only the road contract's local perspective_project alias is replaced explicitly.
No global functional/operator patch, no detector/VAE/Transformer substitution.
The homography, forward grid, GPU RGB/alpha forward values and renderer remain
unchanged. The learnable input is RGB only; learnable geometry is rejected.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fixed_grid_cpu_backward(grad_output, input, grid):
    """Exact same bilinear/zeros/align_corners=True VJP, deterministic CPU backend.

    Uses ATen directly instead of a nested autograd.grad; production still makes
    one top-level autograd.grad call per weighted scalar attack loss. Float64 is
    permitted for standalone gradcheck; actual road RGB/input grid are float32.
    """
    if grid.requires_grad:
        raise ValueError('learnable projection grid is outside this scoped candidate')
    if input.dtype != grid.dtype or grad_output.dtype != input.dtype:
        raise TypeError('road sampler VJP must preserve matching input/grid/gradient dtype')
    if input.dtype not in (torch.float32, torch.float64):
        raise TypeError('CPU road sampler supports float32 production or float64 verification')
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('single-thread strict deterministic CPU backward required')
    with torch.no_grad():
        cpu_input = input.detach().to(device='cpu')
        cpu_grid = grid.detach().to(device='cpu')
        cpu_grad = grad_output.detach().to(device='cpu')
        result, unused_grid_gradient = torch.ops.aten.grid_sampler_2d_backward.default(
            cpu_grad, cpu_input, cpu_grid, 0, 0, True, [True, False])
    return result.to(device=input.device, dtype=input.dtype)


class _FixedRoadRGBGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        if input.ndim != 4 or grid.ndim != 4 or input.shape[0] != 1 or input.shape[1] != 3 or grid.shape[0] != 1 or grid.shape[-1] != 2:
            raise ValueError('expected one RGB road input and one normalized 2D grid')
        if grid.requires_grad:
            raise ValueError('learnable grid is forbidden for RGB-only road backward candidate')
        if input.device != grid.device or input.dtype != grid.dtype:
            raise ValueError('input and grid must share dtype/device')
        if input.device.type == 'cuda' and input.dtype != torch.float32:
            raise ValueError('production road renderer must remain float32')
        ctx.save_for_backward(input, grid)
        # Identical original call: keep native CUDA/cuDNN forward, do not move it.
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise RuntimeError('grid derivative must not be requested')
        grad_input = fixed_grid_cpu_backward(grad_output, input, grid) if ctx.needs_input_grad[0] else None
        return grad_input, None


def native_rgb_sample(input, grid):
    return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def cpu_backward_rgb_sample(input, grid):
    return _FixedRoadRGBGridSample.apply(input, grid)


def make_projection(*, rgb_sampler=cpu_backward_rgb_sample, observer=None):
    """Build a local RoadRGBView projection callable; preserve parent arithmetic.

    `observer` exists solely for fixed-state diagnostic node capture; production
    should omit it. It sees canonical RGB, fixed grid and projected RGB tensor.
    """
    import carriers
    def perspective_project(patch, alpha, destination_xy):
        carriers._validate_rgb(patch)
        if alpha.shape != (1, 1, *patch.shape[-2:]):
            raise ValueError('patch alpha has an incompatible shape')
        if destination_xy.requires_grad or alpha.requires_grad:
            raise ValueError('candidate is scoped to fixed road geometry and nonlearnable alpha')
        destination = destination_xy.to(device=patch.device, dtype=patch.dtype)
        homography = carriers._safe_homography(destination)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(carriers.SCENE_HEIGHT, device=patch.device, dtype=patch.dtype),
            torch.arange(carriers.SCENE_WIDTH, device=patch.device, dtype=patch.dtype), indexing='ij')
        homogeneous = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1)
        mapped = homogeneous @ homography.T
        denominator = mapped[..., 2:]
        epsilon = torch.full_like(denominator, 1e-8)
        denominator = torch.where(denominator.abs() < 1e-8,
            torch.where(denominator < 0, -epsilon, epsilon), denominator)
        grid = (mapped[..., :2] / denominator).unsqueeze(0)
        if grid.requires_grad:
            raise RuntimeError('road grid unexpectedly requires gradient')
        projected = rgb_sampler(patch, grid)
        projected_alpha = F.grid_sample(alpha, grid, mode='bilinear', padding_mode='zeros', align_corners=True).clamp(0., 1.)
        if observer is not None:
            observer(patch, grid, projected)
        return projected, projected_alpha
    return perspective_project


def install_road_rgb_projection_override():
    """Explicitly replace only the contract's alias, return old callable to restore."""
    import road_patch_contract
    original = road_patch_contract.perspective_project
    road_patch_contract.perspective_project = make_projection()
    return original
