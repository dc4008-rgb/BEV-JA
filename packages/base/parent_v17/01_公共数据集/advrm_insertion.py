"""Deterministic AdvRM vehicle insertion without legacy experiment state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms as T


SCENE_HEIGHT = 320
SCENE_WIDTH = 1024
INSERT_HEIGHT = 230
CAR_ROAD_WIDTH_RATIO = 0.65


@dataclass(frozen=True)
class InsertionGeometry:
    object_height: int
    object_width: int
    bottom_y: int
    center_x: int

    @property
    def box_xyxy(self) -> tuple[int, int, int, int]:
        half_width = round(self.object_width / 2)
        left = self.center_x - half_width
        return (
            left,
            self.bottom_y - self.object_height,
            left + self.object_width,
            self.bottom_y,
        )


@dataclass(frozen=True)
class InsertionResult:
    image: torch.Tensor
    mask: torch.Tensor
    geometry: InsertionGeometry


def load_scene(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = T.Compose(
        [T.Resize([SCENE_HEIGHT, SCENE_WIDTH]), T.ToTensor()]
    )
    return transform(image).unsqueeze(0).to(device)


def load_car(
    image_path: Path, mask_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    if len(mask.split()) > 1:
        mask = ImageOps.grayscale(mask)
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    mask_tensor = T.ToTensor()(mask).unsqueeze(0).to(device)
    mask_tensor = mask_tensor.gt(0.5).to(image_tensor.dtype)
    return image_tensor, mask_tensor


def parse_road_quad(raw: str) -> tuple[tuple[float, float], ...]:
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("road_quad_norm must contain four points")
    points = tuple((float(point[0]), float(point[1])) for point in values)
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points):
        raise ValueError("normalized road point is outside the image")
    return points


def canvas_points(
    normalized: tuple[tuple[float, float], ...]
) -> tuple[tuple[int, int], ...]:
    """Return TL, TR, BR, BL integer points on AdvRM's 1024x320 canvas."""
    return tuple(
        (round(x * SCENE_WIDTH), round(y * SCENE_HEIGHT))
        for x, y in normalized
    )


def insertion_geometry(
    normalized: tuple[tuple[float, float], ...],
    car_shape: tuple[int, int],
    *,
    insert_height: int = INSERT_HEIGHT,
    road_width_ratio: float = CAR_ROAD_WIDTH_RATIO,
) -> InsertionGeometry:
    top_left, top_right, bottom_right, bottom_left = canvas_points(normalized)
    road_top = top_left[1]
    road_bottom = bottom_left[1]
    road_height = road_bottom - road_top
    if road_height <= 0 or not road_top <= insert_height <= road_bottom:
        raise ValueError("insert height is outside the road quadrilateral")
    top_width = top_right[0] - top_left[0]
    bottom_width = bottom_right[0] - bottom_left[0]
    if top_width <= 0 or bottom_width <= 0:
        raise ValueError("road quadrilateral has non-positive width")

    vertical_ratio = (insert_height - road_top) / road_height
    road_width = round(top_width + vertical_ratio * (bottom_width - top_width))
    top_center_x = round((top_left[0] + top_right[0]) / 2)
    bottom_center_x = round((bottom_left[0] + bottom_right[0]) / 2)
    center_x = round(
        (insert_height - road_top)
        / road_height
        * (bottom_center_x - top_center_x)
        + top_center_x
    )

    source_height, source_width = car_shape
    object_width = round(road_width * road_width_ratio)
    object_height = round(source_height / source_width * object_width)
    geometry = InsertionGeometry(
        object_height=object_height,
        object_width=object_width,
        bottom_y=insert_height,
        center_x=center_x,
    )
    left, top, right, bottom = geometry.box_xyxy
    if object_height < 1 or object_width < 1:
        raise ValueError("inserted vehicle has an empty size")
    if not (0 <= left < right <= SCENE_WIDTH and 0 <= top < bottom <= SCENE_HEIGHT):
        raise ValueError(f"inserted vehicle is outside the canvas: {geometry}")
    return geometry


def insert_car(
    scene: torch.Tensor,
    car: torch.Tensor,
    car_mask: torch.Tensor,
    road_quad_norm: tuple[tuple[float, float], ...],
    *,
    insert_height: int = INSERT_HEIGHT,
    road_width_ratio: float = CAR_ROAD_WIDTH_RATIO,
) -> InsertionResult:
    if scene.shape != (1, 3, SCENE_HEIGHT, SCENE_WIDTH):
        raise ValueError(f"unexpected scene shape: {tuple(scene.shape)}")
    if car.ndim != 4 or car.shape[:2] != (1, 3):
        raise ValueError(f"unexpected car shape: {tuple(car.shape)}")
    if car_mask.ndim != 4 or car_mask.shape[:2] != (1, 1):
        raise ValueError(f"unexpected car mask shape: {tuple(car_mask.shape)}")
    if car.shape[-2:] != car_mask.shape[-2:]:
        raise ValueError("car and car mask dimensions differ")
    if scene.device != car.device or scene.device != car_mask.device:
        raise ValueError("scene, car, and mask must use the same device")

    geometry = insertion_geometry(
        road_quad_norm,
        (car.shape[-2], car.shape[-1]),
        insert_height=insert_height,
        road_width_ratio=road_width_ratio,
    )
    resized_car = F.interpolate(
        car, size=(geometry.object_height, geometry.object_width)
    )
    resized_mask = F.interpolate(
        car_mask, size=(geometry.object_height, geometry.object_width)
    )
    resized_mask = resized_mask.ge(0.5).to(scene.dtype)

    full_mask = torch.zeros(
        (1, 1, SCENE_HEIGHT, SCENE_WIDTH),
        dtype=scene.dtype,
        device=scene.device,
    )
    full_car = torch.zeros_like(scene)
    left, top, right, bottom = geometry.box_xyxy
    full_mask[:, :, top:bottom, left:right] = resized_mask
    full_car[:, :, top:bottom, left:right] = resized_car
    composited = scene * (1.0 - full_mask) + full_car * full_mask
    return InsertionResult(composited, full_mask, geometry)