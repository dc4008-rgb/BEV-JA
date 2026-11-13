"""Local-only inputs for the v16 road-patch development candidate.

No downloads, test set discovery, or automatic remapping by basename.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from advrm_insertion import insert_car, load_scene, parse_road_quad
from carriers import SceneContext
from data import load_training_manifest, sha256, verify_source

ROOT = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST_SHA256 = "38ae14801d2e330236bd7b408f08ef29c5f93b4e802c6a8f2414a5a352ceb23f"
CAR_SHA256 = "f0edb7f67cb4b697511d68ae39adc1a53bb0ce0c32a87b1306d16c488ea0f5d7"
MASK_SHA256 = "c285e5c88cc48f1fca8194f2c587251192750bc433c701843c5ac57c32b396b1"
MODEL_MANIFEST_SHA256 = "d994315183884c678c35e1b4cdb03c0b459b74ba27fd16bc95e3c17717c0b5dc"
CORE = ("fasterrcnn", "ssd300")
TRANSFER = ("maskrcnn", "fcos")


def training_records(manifest: Path, source_prefix=None, source_root=None):
    if "test" in Path(manifest).name.lower():
        raise ValueError("test manifests are forbidden")
    if sha256(manifest) != TRAIN_MANIFEST_SHA256:
        raise ValueError("training manifest identity differs from the reviewed train/validation split")
    records = load_training_manifest(manifest)
    if bool(source_prefix) != bool(source_root):
        raise ValueError("path remapping requires both source-prefix and source-root")
    if source_prefix:
        records = [replace(r, source_path=Path(source_root) / r.source_path.relative_to(source_prefix))
                   for r in records]
    return records


def checked_file(path, expected):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "status": "missing", "expected_sha256": expected}
    actual = sha256(path)
    return {"path": str(path), "status": "pass" if actual == expected else "hash_mismatch",
            "sha256": actual, "expected_sha256": expected}


def detector_entry(model_id, manifest):
    if model_id not in CORE + TRANSFER:
        raise ValueError("unsupported road detector: " + model_id)
    if sha256(Path(manifest)) != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest differs from the reviewed pretrained identities")
    return next(x for x in json.loads(Path(manifest).read_text())["models"] if x["id"] == model_id)


def detector_file(model_id, manifest, weights_root=None):
    item = detector_entry(model_id, manifest)["artifacts"][0]
    original = Path(item["path"])
    path = original if weights_root is None else Path(weights_root) / original.name
    return path, item["sha256"]


def preflight(records, model_ids, manifest, car, car_mask, weights_root=None):
    checks = [checked_file(car, CAR_SHA256), checked_file(car_mask, MASK_SHA256)]
    for model_id in model_ids:
        path, expected = detector_file(model_id, manifest, weights_root)
        checks.append(dict(checked_file(path, expected), model_id=model_id))
    for record in records:
        checks.append(dict(checked_file(record.source_path, record.source_sha256),
                           sample_id=record.sample_id, split=record.split))
    return {"inputs_ready": all(c["status"] == "pass" for c in checks),
            "checks": checks, "test_data_opened": False}


def build_offline_detector(model_id, device, manifest, weights_root=None):
    from model_adapters import build_detection_adapter
    from torchvision.models._api import WeightsEnum
    from unittest.mock import patch
    path, expected = detector_file(model_id, manifest, weights_root)
    result = checked_file(path, expected)
    if result["status"] != "pass":
        raise RuntimeError("local pretrained detector unavailable: " + str(result))
    entry = detector_entry(model_id, manifest)
    state = torch.load(path, map_location="cpu", weights_only=True)
    calls = []

    def bound_state(weights, *args, **kwargs):
        if weights.url != entry["weights_url"] or calls:
            raise RuntimeError("constructor requested an unbound or additional checkpoint")
        calls.append(weights.url)
        return state

    # Keep the original weights=COCO constructor: weights=None would change
    # Faster R-CNN FrozenBatchNorm and epsilon policies despite equal tensors.
    # Intercept only acquisition, use strict original loading, and forbid IO.
    # Construction is sequential; this scoped class patch must not run in threads.
    with patch.object(WeightsEnum, "get_state_dict", bound_state), patch(
        "torch.hub.download_url_to_file", side_effect=RuntimeError("implicit network download forbidden")
    ):
        adapter = build_detection_adapter(model_id, device)
    if len(calls) != 1:
        raise RuntimeError("constructor did not consume the bound pretrained checkpoint")
    return adapter


def build_context(record, car, car_mask, device):
    if record.split not in ("train", "validation"):
        raise ValueError("test scene forbidden in development")
    verify_source(record)
    background = load_scene(record.source_path, device)
    quad = parse_road_quad(record.road_quad_norm)
    inserted = insert_car(background, car, car_mask, quad)
    return SceneContext(background, inserted.image, car, car_mask,
                        inserted.mask, inserted.geometry.box_xyxy, quad)


def patch_brightness(rgb, generator, bounds=(0.9, 1.1)):
    """EOT varies printed texture only: scene, car and patch support stay fixed."""
    from road_patch_contract import sample_patch_brightness, apply_patch_brightness
    factor = sample_patch_brightness(generator, bounds)
    return apply_patch_brightness(rgb, factor), factor


def patch_update(patch, gradient, method, learning_rate, optimizer=None):
    if not torch.isfinite(gradient).all():
        raise RuntimeError("nonfinite patch gradient")
    if gradient.shape != patch.shape or not gradient.detach().abs().any():
        raise RuntimeError("empty or zero patch gradient; retain failure evidence")
    if not 0 < learning_rate <= 1:
        raise ValueError("invalid RGB learning rate")
    if method == "contextual":
        if optimizer is None:
            raise ValueError("Contextual needs its Adam optimizer")
        patch.grad = gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elif method == "robust_dpatch":
        with torch.no_grad():
            patch.add_(gradient.sign(), alpha=learning_rate)
    else:
        raise ValueError("unknown method")
    with torch.no_grad():
        patch.clamp_(0, 1)
