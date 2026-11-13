"""Historical far-depth objective on the R2 runner's paired geometry views.

Import contract: the caller first adds the frozen R2 source directory (providing
``geometry_eot`` and ``evidence``) and the historical production directory
(providing ``losses``) to sys.path. No fallback algorithm, model loading, RNG
draws, calibration division, gradient normalization, or cross-view cache lives
here. The two adapters must expose differentiable ``predict_for_training``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from depth_spatial import DepthSpatialScope
from evidence import fingerprint
from geometry_eot import transform_support_mask, transform_tensor
from losses import semantic_far_change, semantic_target_far_loss


DEPTH_MODELS = ("monodepth2", "dpt_large")
DEPTH_SEMANTICS = {
    "monodepth2": "relative_depth",
    "dpt_large": "relative_inverse_depth",
}
ROI_POLICY = "nonzero_transformed_object_mask_and_geometric_source_support"


def _depth_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(name + " must be a floating tensor")
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(name + " must have shape [B,1,H,W]")
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(name + " contains negative or non-finite depth")
    return value


def _small_stats(value, roi):
    """Return plain numbers only; no full depth map enters saved evidence."""
    x = value.detach()
    selected = x[roi]
    return {
        "dtype": str(x.dtype), "shape": list(x.shape),
        "minimum": float(x.min()), "maximum": float(x.max()),
        "mean": float(x.double().mean()),
        "zero_pixels": int((x == 0).sum()),
        "roi_mean": float(selected.double().mean()),
        "roi_zero_pixels": int((selected == 0).sum()),
    }


class DepthObjective:
    """Equal mean of two model losses, each averaged over four shared views."""

    def __init__(self, parent, adapters, beta=.1):
        if not isinstance(adapters, Mapping) or set(adapters) != set(DEPTH_MODELS):
            raise ValueError("depth adapters must identify exactly monodepth2 and dpt_large")
        if not math.isfinite(float(beta)) or float(beta) <= 0:
            raise ValueError("depth beta must be finite and positive")
        self.parent = parent
        self.adapters = dict(adapters)
        self.beta = float(beta)
        for model_id in DEPTH_MODELS:
            adapter = self.adapters[model_id]
            if getattr(adapter, "training_output_semantics", None) != DEPTH_SEMANTICS[model_id]:
                raise ValueError("unexpected training output semantics: " + model_id)
            if not callable(getattr(adapter, "predict_for_training", None)):
                raise ValueError("missing differentiable training prediction: " + model_id)
            modules = ([adapter] if isinstance(adapter, torch.nn.Module) else
                       [value for value in vars(adapter).values()
                        if isinstance(value, torch.nn.Module)])
            if not modules:
                raise ValueError("depth adapter has no inspectable model to freeze: " + model_id)
            for module in modules:
                module.eval().requires_grad_(False)

    def call(self, patch, context, views, ledger, *, model_losses_out=None):
        """Optionally expose each model's live scalar graph for an isolated probe.

        The caller owns this diagnostic dict and must supply a new empty dict for
        each call. Default production calls retain no extra model-loss handles.
        """
        if model_losses_out is not None:
            if not isinstance(model_losses_out, dict):
                raise TypeError("model_losses_out must be an empty dict or None")
            if model_losses_out:
                raise ValueError("model_losses_out must be empty; old graphs cannot be reused")
        if len(views) != 4 or [view.get("index") for view in views] != [0, 1, 2, 3]:
            raise ValueError("depth requires the same four indexed R2 geometry views")
        if views[0].get("kind") != "identity":
            raise ValueError("first shared geometry view must be identity")
        clean_image = context.benign.detach()
        rendered = self.parent.RoadRGBView(patch).render(context)
        adversarial_image = rendered.image
        if clean_image.shape != adversarial_image.shape or clean_image.ndim != 4:
            raise ValueError("paired depth RGB shapes must match [B,3,H,W]")
        if clean_image.shape[1] != 3:
            raise ValueError("depth requires RGB images")
        if not bool(torch.isfinite(clean_image).all()) or not bool(torch.isfinite(adversarial_image).all()):
            raise ValueError("non-finite paired depth RGB")
        mask = context.object_mask.detach().to(device=clean_image.device, dtype=clean_image.dtype)
        if mask.shape != clean_image[:, :1].shape or not bool(torch.isfinite(mask).all()):
            raise ValueError("object mask must be finite [B,1,H,W] on the scene canvas")
        if bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError("object mask values must lie in [0,1]")
        values = {model_id: [] for model_id in DEPTH_MODELS}
        details = []
        for view in views:
            with torch.no_grad():
                clean_view = transform_tensor(clean_image, view)
                transformed_mask = transform_tensor(mask, view).gt(0)
                real_support = transform_support_mask(
                    view, device=mask.device, dtype=mask.dtype).unsqueeze(0).bool()
                roi = transformed_mask & real_support
            if not bool(roi.any()):
                raise ValueError("depth object ROI is empty after geometry/padding exclusion")
            adversarial_view = transform_tensor(adversarial_image, view)
            row = {
                "view_index": int(view["index"]), "view_kind": str(view["kind"]),
                "source_view_fingerprint": fingerprint(view),
                "clean_input_fingerprint": fingerprint(clean_view),
                "adversarial_input_fingerprint": fingerprint(adversarial_view),
                "object_roi_fingerprint": fingerprint(roi),
                "valid_roi_pixels": int(roi.sum()),
                "padding_pixels_excluded_from_roi": int((transformed_mask & ~real_support).sum()),
                "source_support_pixels": int(real_support.sum()),
                "models": {},
            }
            for model_id in DEPTH_MODELS:
                adapter = self.adapters[model_id]
                with torch.no_grad(), DepthSpatialScope(model_id) as clean_spatial:
                    clean = ledger.call("depth_clean_forwards", adapter.predict_for_training, clean_view)
                    clean = _depth_tensor(clean, model_id + " clean").detach()
                with DepthSpatialScope(model_id) as adv_spatial:
                    adversarial = ledger.call("depth_adv_forwards", adapter.predict_for_training, adversarial_view)
                adversarial = _depth_tensor(adversarial, model_id + " adversarial")
                if clean.shape != roi.shape or adversarial.shape != clean.shape:
                    raise ValueError("depth training output must match the shared scene/ROI canvas")
                if clean.device != roi.device or adversarial.device != roi.device:
                    raise ValueError("depth outputs and shared ROI must be on the same device")
                if clean.dtype != adversarial.dtype:
                    raise ValueError("paired clean/adversarial depth output dtypes differ")
                if adversarial_view.requires_grad and not adversarial.requires_grad:
                    raise RuntimeError("depth adapter disconnected the adversarial image graph: " + model_id)
                semantics = DEPTH_SEMANTICS[model_id]
                loss = semantic_target_far_loss(
                    adversarial, clean, roi, semantics=semantics, target_shift=self.beta)
                if loss.ndim or not bool(torch.isfinite(loss)):
                    raise ValueError("depth loss must be a finite scalar")
                values[model_id].append(loss)
                with torch.no_grad():
                    change = semantic_far_change(adversarial.detach(), clean, semantics)
                    scale = clean.abs().mean(dim=(-2, -1), keepdim=True)
                    scale = scale.clamp_min(torch.finfo(clean.dtype).eps)
                    row["models"][model_id] = {
                        "training_output_semantics": semantics, "raw_loss": float(loss.detach()),
                        "spatial_clean": clean_spatial.receipt(), "spatial_adversarial": adv_spatial.receipt(),
                        "semantic_far_change_roi_mean": float(change[roi].double().mean()),
                        "clean_asinh_scale": scale.reshape(-1).cpu().tolist(),
                        "clean": _small_stats(clean, roi),
                        "adversarial": _small_stats(adversarial, roi),
                    }
            details.append(row)
        model_losses = [torch.stack(values[model_id]).mean() for model_id in DEPTH_MODELS]
        total = torch.stack(model_losses).mean()
        evidence = {
            "schema": "r2_joint_depth_objective_v1", "depth_beta": self.beta,
            "model_order": list(DEPTH_MODELS), "model_weights": [0.5, 0.5],
            "aggregation": "mean_of_two_model_means_over_four_shared_views",
            "roi_policy": ROI_POLICY,
            "asinh_scale_policy": "historical_full_transformed_clean_prediction_mean_dtype_epsilon_floor",
            "clean_reference": "same_view_same_model_prediction_on_benign_scene_with_inserted_car",
            "reference_cache": "none_recompute_each_model_and_view",
            "source_clean_fingerprint": fingerprint(clean_image),
            "source_adversarial_fingerprint": fingerprint(adversarial_image),
            "raw_model_losses": {model_id: float(value.detach())
                                 for model_id, value in zip(DEPTH_MODELS, model_losses)},
            "raw_depth_loss": float(total.detach()), "views": details,
            "depth_clean_forwards": 8, "depth_adv_forwards": 8,
        }
        if model_losses_out is not None:
            model_losses_out.update(zip(DEPTH_MODELS, model_losses))
        return total, evidence

    __call__ = call
