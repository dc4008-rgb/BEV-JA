"""Production carrier parameterizations for the normalized adaptations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from advrm_insertion import (
    SCENE_HEIGHT,
    SCENE_WIDTH,
    insert_car,
    insertion_geometry,
)


@dataclass(frozen=True)
class SceneContext:
    background: torch.Tensor
    benign: torch.Tensor
    car: torch.Tensor
    car_mask: torch.Tensor
    object_mask: torch.Tensor
    object_box: tuple[int, int, int, int]
    road_quad: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CarrierRender:
    image: torch.Tensor
    support: torch.Tensor
    canonical_rgb: torch.Tensor
    kind: str


def logit(value: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    clipped = value.clamp(epsilon, 1.0 - epsilon)
    return torch.log(clipped / (1.0 - clipped))


def _validate_rgb(value: torch.Tensor) -> None:
    if value.ndim != 4 or value.shape[:2] != (1, 3):
        raise ValueError(f"RGB tensor must have shape [1,3,H,W], got {value.shape}")
    if not torch.isfinite(value).all():
        raise ValueError("RGB tensor contains non-finite values")


def _safe_homography(destination_xy: torch.Tensor) -> torch.Tensor:
    source_uv = destination_xy.new_tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    rows = []
    values = []
    for destination, source in zip(destination_xy, source_uv):
        x_coord, y_coord = destination.unbind()
        u_coord, v_coord = source.unbind()
        zero = x_coord.new_zeros(())
        one = x_coord.new_ones(())
        rows.extend(
            [
                torch.stack(
                    [
                        x_coord,
                        y_coord,
                        one,
                        zero,
                        zero,
                        zero,
                        -u_coord * x_coord,
                        -u_coord * y_coord,
                    ]
                ),
                torch.stack(
                    [
                        zero,
                        zero,
                        zero,
                        x_coord,
                        y_coord,
                        one,
                        -v_coord * x_coord,
                        -v_coord * y_coord,
                    ]
                ),
            ]
        )
        values.extend([u_coord, v_coord])
    coefficients = torch.linalg.solve(torch.stack(rows), torch.stack(values))
    return torch.cat([coefficients, coefficients.new_ones(1)]).view(3, 3)


def perspective_project(
    patch: torch.Tensor,
    alpha: torch.Tensor,
    destination_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_rgb(patch)
    if alpha.shape != (1, 1, *patch.shape[-2:]):
        raise ValueError("patch alpha has an incompatible shape")
    destination = destination_xy.to(device=patch.device, dtype=patch.dtype)
    homography = _safe_homography(destination)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(SCENE_HEIGHT, device=patch.device, dtype=patch.dtype),
        torch.arange(SCENE_WIDTH, device=patch.device, dtype=patch.dtype),
        indexing="ij",
    )
    homogeneous = torch.stack(
        [x_coords, y_coords, torch.ones_like(x_coords)], dim=-1
    )
    mapped = homogeneous @ homography.T
    denominator = mapped[..., 2:]
    epsilon = torch.full_like(denominator, 1e-8)
    denominator = torch.where(
        denominator.abs() < 1e-8,
        torch.where(denominator < 0, -epsilon, epsilon),
        denominator,
    )
    grid = (mapped[..., :2] / denominator).unsqueeze(0)
    projected = F.grid_sample(
        patch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    projected_alpha = F.grid_sample(
        alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).clamp(0.0, 1.0)
    return projected, projected_alpha


def road_patch_destination(
    road_quad: tuple[tuple[float, float], ...],
    *,
    top_fraction: float = 0.58,
    bottom_fraction: float = 0.92,
    width_fraction: float = 0.72,
) -> torch.Tensor:
    if not 0 < top_fraction < bottom_fraction < 1 or not 0 < width_fraction <= 1:
        raise ValueError("invalid road patch fractions")
    points = [(x * SCENE_WIDTH, y * SCENE_HEIGHT) for x, y in road_quad]
    top_left, top_right, bottom_right, bottom_left = points

    def interpolate(top: tuple[float, float], bottom: tuple[float, float], fraction: float) -> tuple[float, float]:
        return (
            top[0] + fraction * (bottom[0] - top[0]),
            top[1] + fraction * (bottom[1] - top[1]),
        )

    left_top = interpolate(top_left, bottom_left, top_fraction)
    right_top = interpolate(top_right, bottom_right, top_fraction)
    left_bottom = interpolate(top_left, bottom_left, bottom_fraction)
    right_bottom = interpolate(top_right, bottom_right, bottom_fraction)

    def shrink(left: tuple[float, float], right: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
        center = (0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1]))
        return (
            (
                center[0] + width_fraction * (left[0] - center[0]),
                center[1] + width_fraction * (left[1] - center[1]),
            ),
            (
                center[0] + width_fraction * (right[0] - center[0]),
                center[1] + width_fraction * (right[1] - center[1]),
            ),
        )

    left_top, right_top = shrink(left_top, right_top)
    left_bottom, right_bottom = shrink(left_bottom, right_bottom)
    return torch.tensor([left_top, right_top, right_bottom, left_bottom])


def extract_road_patch(
    scene: torch.Tensor,
    road_quad: tuple[tuple[float, float], ...],
    *,
    size: tuple[int, int] = (64, 128),
) -> torch.Tensor:
    _validate_rgb(scene)
    destination = road_patch_destination(road_quad).to(scene)
    top_left, top_right, bottom_right, bottom_left = destination
    height, width = size
    vertical = torch.linspace(0.0, 1.0, height, device=scene.device, dtype=scene.dtype)
    horizontal = torch.linspace(0.0, 1.0, width, device=scene.device, dtype=scene.dtype)
    vertical_grid, horizontal_grid = torch.meshgrid(
        vertical, horizontal, indexing="ij"
    )
    top = top_left[None, None] + horizontal_grid[..., None] * (
        top_right - top_left
    )[None, None]
    bottom = bottom_left[None, None] + horizontal_grid[..., None] * (
        bottom_right - bottom_left
    )[None, None]
    coordinates = top + vertical_grid[..., None] * (bottom - top)
    grid = coordinates.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (SCENE_WIDTH - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (SCENE_HEIGHT - 1) - 1.0
    return F.grid_sample(
        scene,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def apply_eot(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    spec: dict[str, Any] | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not spec:
        return rgb, alpha

    def sample(bounds: list[float] | tuple[float, float], default: float) -> float:
        if bounds is None:
            return default
        low, high = map(float, bounds)
        return float(
            torch.empty((), device=rgb.device).uniform_(low, high, generator=generator)
        )

    contrast = sample(spec.get("contrast"), 1.0)
    brightness = sample(spec.get("brightness"), 0.0)
    transformed = ((rgb - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)
    noise_bounds = spec.get("noise")
    if noise_bounds is not None:
        low, high = map(float, noise_bounds)
        noise = torch.empty_like(transformed).uniform_(low, high, generator=generator)
        transformed = (transformed + noise).clamp(0.0, 1.0)

    angle = math.radians(sample(spec.get("rotation_degrees"), 0.0))
    scale = sample(spec.get("scale"), 1.0)
    if angle or scale != 1.0:
        cosine = math.cos(angle) / max(scale, 1e-6)
        sine = math.sin(angle) / max(scale, 1e-6)
        theta = rgb.new_tensor(
            [[[cosine, -sine, 0.0], [sine, cosine, 0.0]]]
        )
        grid = F.affine_grid(theta, transformed.shape, align_corners=False)
        transformed = F.grid_sample(
            transformed, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        alpha = F.grid_sample(
            alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
    return transformed.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


class BaseCarrier(torch.nn.Module):
    kind: str

    def canonical_rgb(self) -> torch.Tensor:
        raise NotImplementedError

    def regularization_rgb(self) -> torch.Tensor:
        return self.canonical_rgb()

    def project_linf_(self, radius: float) -> None:
        raise TypeError(f"{type(self).__name__} does not support RGB L-infinity projection")

    def bim_step_(self, learning_rate: float) -> None:
        with torch.no_grad():
            for parameter in carrier_trainable_parameters(self):
                parameter.add_(parameter.grad.sign(), alpha=-learning_rate)
                parameter.grad = None

    def render(
        self,
        context: SceneContext,
        *,
        eot: dict[str, Any] | None,
        generator: torch.Generator,
    ) -> CarrierRender:
        rgb = self.canonical_rgb()
        _validate_rgb(rgb)
        if self.kind == "vehicle":
            alpha = torch.ones_like(rgb[:, :1])
            rgb, alpha = apply_eot(rgb, alpha, eot, generator)
            left, top, right, bottom = context.object_box
            object_size = (bottom - top, right - left)
            resized_rgb = F.interpolate(rgb, size=object_size, mode="nearest")
            resized_alpha = F.interpolate(alpha, size=object_size, mode="nearest")
            clean_crop = context.benign[..., top:bottom, left:right]
            replacement = resized_rgb * resized_alpha + clean_crop * (1.0 - resized_alpha)
            canvas = F.pad(
                replacement,
                (left, context.benign.shape[-1] - right, top, context.benign.shape[-2] - bottom),
            )
            image = torch.where(context.object_mask.bool(), canvas, context.benign)
            return CarrierRender(image, context.object_mask, rgb, self.kind)
        if self.kind == "road":
            alpha = torch.ones_like(rgb[:, :1])
            rgb, alpha = apply_eot(rgb, alpha, eot, generator)
            destination = road_patch_destination(context.road_quad).to(rgb)
            perspective_bounds = eot.get("perspective_vertex") if eot else None
            if perspective_bounds is not None:
                low, high = map(float, perspective_bounds)
                jitter = torch.empty_like(destination).uniform_(
                    low, high, generator=generator
                )
                extent = destination.amax(dim=0) - destination.amin(dim=0)
                destination = destination + jitter * extent
            projected, projected_alpha = perspective_project(
                rgb, alpha, destination
            )
            patched_background = (
                context.background * (1.0 - projected_alpha)
                + projected * projected_alpha
            ).clamp(0.0, 1.0)
            inserted = insert_car(
                patched_background,
                context.car,
                context.car_mask,
                context.road_quad,
            )
            if int(((projected_alpha > 0.5) & inserted.mask.bool()).sum()) != 0:
                raise RuntimeError("road carrier overlaps the inserted vehicle")
            return CarrierRender(
                inserted.image, projected_alpha, rgb, self.kind
            )
        raise ValueError(f"unsupported carrier kind: {self.kind}")


class PixelVehicleCarrier(BaseCarrier):
    kind = "vehicle"

    def __init__(self, car: torch.Tensor) -> None:
        super().__init__()
        initial = F.interpolate(car, size=(96, 160), mode="bilinear", align_corners=False)
        self.logits = torch.nn.Parameter(logit(initial))
        self.register_buffer("initial", initial.detach())

    def canonical_rgb(self) -> torch.Tensor:
        return self.logits.sigmoid()

    def project_linf_(self, radius: float) -> None:
        with torch.no_grad():
            projected = self.canonical_rgb().clamp(
                (self.initial - radius).clamp(0.0, 1.0),
                (self.initial + radius).clamp(0.0, 1.0),
            )
            self.logits.copy_(logit(projected))


class BlockVehicleCarrier(BaseCarrier):
    kind = "vehicle"

    def __init__(self, car: torch.Tensor) -> None:
        super().__init__()
        initial = F.interpolate(car, size=(12, 20), mode="area")
        self.logits = torch.nn.Parameter(logit(initial))
        self.register_buffer(
            "initial",
            F.interpolate(initial, size=(96, 160), mode="nearest").detach(),
        )

    def canonical_rgb(self) -> torch.Tensor:
        return F.interpolate(
            self.logits.sigmoid(), size=(96, 160), mode="nearest"
        )

    def regularization_rgb(self) -> torch.Tensor:
        return self.logits.sigmoid()


class CoordinateVehicleCarrier(BaseCarrier):
    kind = "vehicle"

    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260905)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, 3),
            )
        y_coord, x_coord = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 96),
            torch.linspace(-1.0, 1.0, 160),
            indexing="ij",
        )
        self.register_buffer("coordinates", torch.stack([x_coord, y_coord], dim=-1))
        self.register_buffer("initial", torch.full((1, 3, 96, 160), 0.5))

    def canonical_rgb(self) -> torch.Tensor:
        rgb = self.network(self.coordinates).sigmoid()
        return rgb.permute(2, 0, 1).unsqueeze(0)


class RoadPatchCarrier(BaseCarrier):
    kind = "road"

    def __init__(self, initial_patch: torch.Tensor) -> None:
        super().__init__()
        initial = F.interpolate(
            initial_patch, size=(64, 128), mode="bilinear", align_corners=False
        )
        self.logits = torch.nn.Parameter(logit(initial))
        self.register_buffer("initial", initial.detach())

    def canonical_rgb(self) -> torch.Tensor:
        return self.logits.sigmoid()

    def project_linf_(self, radius: float) -> None:
        with torch.no_grad():
            projected = self.canonical_rgb().clamp(
                (self.initial - radius).clamp(0.0, 1.0),
                (self.initial + radius).clamp(0.0, 1.0),
            )
            self.logits.copy_(logit(projected))


class AdvRMRoadPatchCarrier(BaseCarrier):
    kind = "road"

    def __init__(self, initial_patch: torch.Tensor) -> None:
        super().__init__()
        initial = F.interpolate(
            initial_patch, size=(64, 128), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        self.rgb = torch.nn.Parameter(initial.detach().clone())
        self.register_buffer("initial", initial.detach())

    def canonical_rgb(self) -> torch.Tensor:
        return self.rgb

    def bim_step_(self, learning_rate: float) -> None:
        if self.rgb.grad is None:
            raise RuntimeError("AdvRM RGB gradient is missing")
        with torch.no_grad():
            self.rgb.sub_(learning_rate * self.rgb.grad.sign()).clamp_(0.0, 1.0)
            self.rgb.grad = None


class APARoadPatchCarrier(BaseCarrier):
    kind = "road"

    def __init__(self, initial_patch: torch.Tensor) -> None:
        super().__init__()
        initial = F.interpolate(
            initial_patch, size=(64, 128), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        self.rgb = torch.nn.Parameter(initial.detach().clone())
        self.register_buffer("initial", initial.detach())

    def canonical_rgb(self) -> torch.Tensor:
        return self.rgb

    def clamp_rgb_(self) -> None:
        with torch.no_grad():
            self.rgb.clamp_(0.0, 1.0)


class SAAMRoadPatchCarrier(BaseCarrier):
    kind = "road"

    def __init__(self, initial_patch: torch.Tensor) -> None:
        super().__init__()
        initial = F.interpolate(
            initial_patch, size=(64, 128), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        self.rgb = torch.nn.Parameter(initial.detach().clone())
        self.register_buffer("initial", initial.detach())

    def canonical_rgb(self) -> torch.Tensor:
        return self.rgb

    def project_linf_(self, radius: float) -> None:
        with torch.no_grad():
            self.rgb.clamp_(
                (self.initial - radius).clamp(0.0, 1.0),
                (self.initial + radius).clamp(0.0, 1.0),
            )


class GaussianVehicleCarrier(BaseCarrier):
    kind = "vehicle"

    def __init__(self, count: int = 32) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(20260905)
        self.means = torch.nn.Parameter(
            torch.empty(count, 2).uniform_(-0.8, 0.8, generator=generator)
        )
        self.log_scales = torch.nn.Parameter(torch.full((count, 2), -1.7))
        self.color_logits = torch.nn.Parameter(torch.zeros(count, 3))
        self.opacity_logits = torch.nn.Parameter(torch.zeros(count))
        self.register_buffer("initial_means", self.means.detach().clone())
        self.register_buffer("initial_log_scales", self.log_scales.detach().clone())
        self.register_buffer("initial_colors", self.color_logits.sigmoid().detach().clone())
        self.register_buffer("initial", torch.full((1, 3, 96, 160), 0.5))

    def canonical_rgb(self) -> torch.Tensor:
        y_coord, x_coord = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 96, device=self.means.device),
            torch.linspace(-1.0, 1.0, 160, device=self.means.device),
            indexing="ij",
        )
        scales = self.log_scales.exp().clamp_min(1e-3)
        distance = (
            (x_coord[None] - self.means[:, 0, None, None]).square()
            / scales[:, 0, None, None].square()
            + (y_coord[None] - self.means[:, 1, None, None]).square()
            / scales[:, 1, None, None].square()
        )
        alpha = (
            self.opacity_logits.sigmoid()[:, None, None]
            * torch.exp(-0.5 * distance)
        ).clamp(max=0.95)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:1]), 1.0 - alpha + 1e-8], dim=0),
            dim=0,
        )[:-1]
        weights = alpha * transmittance
        colors = self.color_logits.sigmoid()
        rgb = (weights[:, None] * colors[:, :, None, None]).sum(dim=0, keepdim=True)
        support = weights.sum(dim=0, keepdim=True).unsqueeze(0)
        return rgb + 0.5 * (1.0 - support)

    def project_linf_(self, radius: float) -> None:
        with torch.no_grad():
            colors = self.color_logits.sigmoid().clamp(
                (self.initial_colors - radius).clamp(0.0, 1.0),
                (self.initial_colors + radius).clamp(0.0, 1.0),
            )
            self.color_logits.copy_(logit(colors))

    def color_budget_loss(self, radius: float) -> torch.Tensor:
        if radius < 0:
            raise ValueError("color L-infinity radius must be non-negative")
        maximum_shift = (
            self.color_logits.sigmoid() - self.initial_colors
        ).abs().amax()
        return (maximum_shift - radius).clamp_min(0.0)

    def shape_loss(self) -> torch.Tensor:
        return (self.means - self.initial_means).square().mean() + (
            self.log_scales - self.initial_log_scales
        ).square().mean()


class FrozenRgbCarrier(BaseCarrier):
    def __init__(self, rgb: torch.Tensor, kind: str) -> None:
        super().__init__()
        if kind not in {"vehicle", "road"}:
            raise ValueError(f"unsupported frozen carrier kind: {kind}")
        _validate_rgb(rgb)
        self.kind = kind
        self.register_buffer("rgb", rgb.detach().clone())

    def canonical_rgb(self) -> torch.Tensor:
        return self.rgb


class VAELatentRoadCarrier(BaseCarrier):
    kind = "road"
    uses_diffusion_denoising = False

    def __init__(
        self,
        snapshot: Path,
        device: torch.device,
        initial_patch: torch.Tensor,
    ) -> None:
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            snapshot,
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device).eval().requires_grad_(False)
        initial = F.interpolate(
            initial_patch.to(device),
            size=(64, 128),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            encoded = self.vae.encode(initial * 2.0 - 1.0).latent_dist.mode()
            latent = (
                encoded - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        self.latent = torch.nn.Parameter(latent)
        self.register_buffer("initial", initial.detach())

    def canonical_rgb(self) -> torch.Tensor:
        decoded = self.vae.decode(
            self.latent / self.vae.config.scaling_factor
            + self.vae.config.shift_factor,
            return_dict=False,
        )[0]
        return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


def build_carrier(
    carrier_id: str,
    *,
    car: torch.Tensor,
    car_mask: torch.Tensor,
    initial_patch: torch.Tensor,
    device: torch.device,
    sd35_snapshot: Path | None = None,
) -> BaseCarrier:
    if carrier_id == "vehicle_rgb":
        carrier: BaseCarrier = PixelVehicleCarrier(car.detach().cpu())
    elif carrier_id == "block_vehicle_texture":
        carrier = BlockVehicleCarrier(car.detach().cpu())
    elif carrier_id == "coordinate_mlp_vehicle":
        carrier = CoordinateVehicleCarrier()
    elif carrier_id == "road_patch":
        carrier = RoadPatchCarrier(initial_patch)
    elif carrier_id == "advrm_rgb_road_patch":
        carrier = AdvRMRoadPatchCarrier(initial_patch)
    elif carrier_id == "apa_rgb_road_patch":
        carrier = APARoadPatchCarrier(initial_patch)
    elif carrier_id == "saam_rgb_road_patch":
        carrier = SAAMRoadPatchCarrier(initial_patch)
    elif carrier_id == "gaussian_vehicle_texture":
        carrier = GaussianVehicleCarrier()
    elif carrier_id == "sd35_vae_latent_road_patch":
        if sd35_snapshot is None:
            raise ValueError("SD3.5 snapshot is required for VAE-latent carrier")
        carrier = VAELatentRoadCarrier(sd35_snapshot, device, initial_patch)
    else:
        raise ValueError(f"unsupported carrier: {carrier_id}")
    if carrier.kind == "vehicle":
        initialize_vehicle_reference(carrier, car, car_mask)
    return carrier.to(device)


def initialize_vehicle_reference(
    carrier: BaseCarrier,
    car: torch.Tensor,
    car_mask: torch.Tensor,
    *,
    steps: int = 200,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    if carrier.kind != "vehicle" or isinstance(carrier, FrozenRgbCarrier):
        raise ValueError("initialization requires a trainable vehicle carrier")
    _validate_rgb(car)
    if car.min() < 0 or car.max() > 1:
        raise ValueError("vehicle asset RGB is outside [0,1]")
    if car_mask.shape != (1, 1, *car.shape[-2:]):
        raise ValueError("vehicle asset mask has an incompatible shape")
    if not torch.isfinite(car_mask).all() or car_mask.min() < 0 or car_mask.max() > 1:
        raise ValueError("vehicle asset mask must be finite and in [0,1]")
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("fit steps and learning rate must be positive")
    parameters = carrier_trainable_parameters(carrier)
    target = F.interpolate(
        car.detach().to(parameters[0]), size=(96, 160), mode="bilinear", align_corners=False,
    )
    mask = F.interpolate(car_mask.detach().to(target), size=(96, 160), mode="nearest")
    denominator = 3 * mask.sum()
    if denominator <= 0:
        raise ValueError("vehicle asset mask has no canonical support")

    def reconstruction_loss() -> torch.Tensor:
        return ((carrier.canonical_rgb() - target).square() * mask).sum() / denominator

    before = float(reconstruction_loss().detach())
    fit_steps = steps if isinstance(carrier, (GaussianVehicleCarrier, CoordinateVehicleCarrier)) else 0
    if fit_steps:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(fit_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite vehicle reconstruction loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise RuntimeError("invalid vehicle reconstruction gradient")
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        reference = carrier.canonical_rgb().detach().clone()
        _validate_rgb(reference)
        carrier.initial.copy_(reference)
        if isinstance(carrier, GaussianVehicleCarrier):
            carrier.initial_means.copy_(carrier.means)
            carrier.initial_log_scales.copy_(carrier.log_scales)
            carrier.initial_colors.copy_(carrier.color_logits.sigmoid())
        after = float(reconstruction_loss())
    report = {
        "source": "fixed_vehicle_asset_only",
        "representation": "2d_appearance_adaptation",
        "fit_steps": fit_steps,
        "fit_learning_rate": learning_rate if fit_steps else None,
        "fit_seed": 20260905 if fit_steps else None,
        "fit_device": str(parameters[0].device),
        "objective": "masked_canonical_rgb_mse",
        "reconstruction_mse_before": before,
        "reconstruction_mse_after": after,
        "reference": "actual_zero_step_canonical_rgb",
        "victim_models_used": False,
        "scene_images_used": False,
    }
    carrier.initialization_report = report
    return report


def carrier_trainable_parameters(carrier: BaseCarrier) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in carrier.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("carrier exposes no trainable parameters")
    return parameters


def carrier_metadata(carrier: BaseCarrier) -> dict[str, Any]:
    parameters = carrier_trainable_parameters(carrier)
    return {
        "class": type(carrier).__name__,
        "kind": carrier.kind,
        "trainable_tensors": len(parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }


def carrier_shape_loss(carrier: BaseCarrier) -> torch.Tensor:
    if isinstance(carrier, GaussianVehicleCarrier):
        return carrier.shape_loss()
    return carrier.canonical_rgb().sum() * 0.0


def carrier_color_budget_loss(carrier: BaseCarrier, radius: float) -> torch.Tensor:
    if isinstance(carrier, GaussianVehicleCarrier):
        return carrier.color_budget_loss(radius)
    raise TypeError("color-budget loss requires the Gaussian vehicle carrier")


def project_method_constraints(
    carrier: BaseCarrier,
    method_id: str,
    contract: dict[str, Any],
) -> None:
    regularizers = contract.get("regularizers", {})
    if method_id in {"upc", "saam"}:
        if method_id == "saam" and not isinstance(carrier, SAAMRoadPatchCarrier):
            raise TypeError("SAAM requires the direct-RGB road-patch carrier")
        carrier.project_linf_(float(regularizers["linf_radius"]))
    elif method_id == "apa":
        if not isinstance(carrier, APARoadPatchCarrier):
            raise TypeError("APA requires the direct-RGB road-patch carrier")
        carrier.clamp_rgb_()