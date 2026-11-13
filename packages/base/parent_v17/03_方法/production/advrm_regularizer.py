"""Frozen VGG feature regularization used by the released AdvRM objective."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from losses import advrm_feature_losses, advrm_residual_tv


STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
CONTENT_LAYER = "conv4_2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AdvRMFeatureExtractor(torch.nn.Module):
    def __init__(self, features: torch.nn.Sequential) -> None:
        super().__init__()
        self.features = features.eval().requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        value = (rgb - mean) / std
        outputs = {}
        pool = 1
        convolution = 0
        for layer in self.features:
            if isinstance(layer, torch.nn.Conv2d):
                convolution += 1
                name = f"conv{pool}_{convolution}"
            elif isinstance(layer, torch.nn.ReLU):
                layer = torch.nn.ReLU(inplace=False)
                name = f"relu{pool}_{convolution}"
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f"pool_{pool}"
                pool += 1
                convolution = 0
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = f"bn{pool}_{convolution}"
            else:
                raise RuntimeError(f"unsupported VGG feature layer: {type(layer).__name__}")
            value = layer(value)
            if name in STYLE_LAYERS or name == CONTENT_LAYER:
                outputs[name] = value
            if set(outputs) == set(STYLE_LAYERS) | {CONTENT_LAYER}:
                break
        missing = sorted((set(STYLE_LAYERS) | {CONTENT_LAYER}) - outputs.keys())
        if missing:
            raise RuntimeError(f"VGG19 feature extractor is missing layers: {missing}")
        return outputs


class AdvRMRegularizer:
    def __init__(self, extractor: torch.nn.Module, reference: torch.Tensor) -> None:
        self.extractor = extractor.eval().requires_grad_(False)
        with torch.no_grad():
            self.targets = {
                name: value.detach()
                for name, value in self.extractor(reference).items()
            }
        self.reference = reference.detach()

    def losses(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        style, content = advrm_feature_losses(
            self.extractor(rgb),
            self.targets,
            style_layers=STYLE_LAYERS,
            content_layer=CONTENT_LAYER,
        )
        return {
            "style": style,
            "content": content,
            "tv": advrm_residual_tv(rgb, self.reference),
        }


def build_advrm_regularizer(
    reference: torch.Tensor,
    checkpoint: Path,
) -> AdvRMRegularizer:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen AdvRM VGG19 checkpoint is missing: {checkpoint}")
    checksum = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    if not digest.startswith("dcbb9e9d"):
        raise ValueError(f"AdvRM VGG19 checkpoint SHA-256 mismatch: {digest}")
    from torchvision.models import vgg19

    model = vgg19(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    extractor = AdvRMFeatureExtractor(model.features).to(reference.device)
    return AdvRMRegularizer(extractor, reference)