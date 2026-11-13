"""Source-only SD3.5 TWO-CLIP/NO-T5 kernel for a NEW two-case diagnostic.

This module does not load weights, start training, read old results, or operate
SSH/Gates. A caller must admit a new, verified, locally loaded two-CLIP/no-T5 SD3 pipeline
and bind its snapshot, source and resource provenance before use. This is an
explicit two-CLIP conditioning adaptation: T5 and tokenizer_3 must be None.
Diffusers' official zero-T5-sequence branch is retained, not imputed as a real
T5 encoder execution or claimed equivalent to full-T5 conditioning. There is no
automatic fallback from a failed/missing encoder; the mode is fixed on input.

Application-call counters and nn.Module forward-hook observations are distinct.
Neither these counters nor the presence of this source proves a GPU run,
four-surrogate integration, scientific acceptance or effect improvement.

API reference (pinned diffusers v0.38.0):
https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import hashlib
import json


IDENTITY = "conditional_generation_two_clip_no_t5_v1"
CONDITIONING_MODE = "two_clip_no_t5"


@dataclass(frozen=True)
class ConditionalTwoClipConfig:
    conditioning_mode: str = CONDITIONING_MODE
    num_inference_steps: int = 100
    begin_index: int = 75
    guidance_scale: float = 4.5
    seed: int = 20260919
    image_size: int = 512
    prompt: str = "top-down aerial view of clean asphalt road surface"
    negative_prompt: str = "vehicle, car, obstacle"
    max_sequence_length: int = 256

    def __post_init__(self):
        if type(self.conditioning_mode) is not str or self.conditioning_mode != CONDITIONING_MODE:
            raise ValueError("this separate kernel is exclusively two_clip_no_t5")
        # This first diagnostic has one declared schedule, not a search space.
        for key, expected in (("num_inference_steps", 100), ("begin_index", 75),
                              ("seed", 20260919), ("image_size", 512),
                              ("max_sequence_length", 256)):
            if type(getattr(self, key)) is not int or getattr(self, key) != expected:
                raise ValueError("undeclared diagnostic parameter: " + key)
        if type(self.guidance_scale) not in (int, float) or self.guidance_scale != 4.5:
            raise ValueError("diagnostic CFG must be exactly 4.5")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("a nonempty road condition is required")
        if not isinstance(self.negative_prompt, str) or not self.negative_prompt.strip():
            raise ValueError("a nonempty negative condition is required")

    @property
    def active_steps(self):
        return self.num_inference_steps - self.begin_index


def declared_rounds(updates: int) -> int:
    """Only the later 25/500 diagnostics; connectivity is a bounded prefix."""
    if type(updates) is not int or updates not in (25, 500):
        raise ValueError("updates must be exactly 25 or 500")
    return updates // 25


def require_two_clip_component_presence(pipe):
    """Structural refusal only; no type/weight/runtime admission or fallback."""
    names = ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2",
             "transformer", "vae", "scheduler")
    missing = [name for name in names if getattr(pipe, name, None) is None]
    if missing:
        raise ValueError("two-CLIP conditional SD3 components required: " + ", ".join(missing))
    for name in ("text_encoder_3", "tokenizer_3"):
        if not hasattr(pipe, name) or getattr(pipe, name) is not None:
            raise ValueError("two_clip_no_t5 requires explicit None for " + name)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; LOCAL source preparation is not GPU validation") from error
    return torch


def _attack_leaf(tensor):
    """The sole generation/attack graph boundary; caller must enable grad."""
    if _torch().is_inference_mode_enabled():
        raise RuntimeError("inference_mode cannot provide the attack gradient boundary")
    return tensor.detach().requires_grad_(True)


def embedding_descriptor(value):
    """Full dtype/shape-bound SHA of an ACTUALLY RETURNED embedding; no body export.

    Byte equality is exact. Hashing copies detached contiguous bytes to CPU,
    without casting or changing the scientific tensor. Token IDs are not
    intercepted by this module and are explicitly not claimed as collected.
    """
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.numel() == 0:
        raise ValueError("nonempty actual floating embedding required")
    tensor = value.detach().cpu().contiguous()
    fields = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    header = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {**fields, "tensor_bytes": len(raw),
            "typed_sha256": hashlib.sha256(header + raw).hexdigest(),
            "hash_rule": "canonical_json_dtype_shape_header_plus_exact_contiguous_tensor_bytes"}


def _cfg_merge(prediction, guidance_scale):
    if prediction.shape[0] != 2:
        raise ValueError("one-image CFG requires [negative, positive] batch2")
    negative, positive = prediction.chunk(2)
    return negative + guidance_scale * (positive - negative)


@dataclass
class TwoClipCallCounts:
    # Attempts are incremented before invocation; completion only after return.
    encode_condition_attempted: int = 0
    encode_condition_completed: int = 0
    vae_encode_attempted: int = 0
    vae_encode_completed: int = 0
    vae_decode_attempted: int = 0
    vae_decode_completed: int = 0
    transformer_attempted: int = 0
    transformer_completed: int = 0
    scheduler_step_attempted: int = 0
    scheduler_step_completed: int = 0
    noise_draws_completed: int = 0
    round_starts_completed: int = 0
    # These are observations from actual completed module.forward calls.
    module_forward_completed: dict = field(default_factory=lambda: {
        "text_encoder": 0, "text_encoder_2": 0,
        "transformer": 0, "vae_encoder": 0, "vae_decoder": 0})


class SD35ConditionalTwoClipKernel:
    """Owns a freshly admitted TWO-CLIP pipeline; no VAE-only helper or T5 fallback.

    Typical caller flow:
      encode_condition(device)
      latent, timesteps = start_round(current_patch, round_index)
      for timestep in timesteps:
          z = generated_leaf(latent, timestep)
          patch = decode_leaf(z)       # differentiable to z
          # Four frozen surrogates + projection + two VJPs are CALLER work.
          latent = caller_attack_update(z, ...).detach()
      current_patch = decode_leaf(latent).detach()

    Do not wrap this whole flow in torch.no_grad/inference_mode. Only frozen
    generation/encoding is no_grad; decoding a leaf retains its input gradient.
    Source-only tests are not proof that caller_attack_update has been integrated.
    """
    def __init__(self, pipe, config: ConditionalTwoClipConfig = ConditionalTwoClipConfig()):
        if type(config) is not ConditionalTwoClipConfig:
            raise TypeError("the frozen ConditionalTwoClipConfig type is required")
        require_two_clip_component_presence(pipe)
        torch = _torch()
        import diffusers
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
        from transformers import CLIPTextModelWithProjection
        if diffusers.__version__.split("+")[0] != "0.38.0":
            raise RuntimeError("new diagnostic binds diffusers 0.38.0")
        if type(pipe) is not StableDiffusion3Pipeline:
            raise TypeError("an official, unwrapped StableDiffusion3Pipeline is required")
        if type(pipe.transformer) is not SD3Transformer2DModel:
            raise TypeError("a real SD3Transformer2DModel is required")
        if type(pipe.vae) is not AutoencoderKL:
            raise TypeError("a real AutoencoderKL is required")
        if pipe.transformer.config.in_channels != 16 or pipe.vae.config.latent_channels != 16:
            raise ValueError("declared SD3.5 Transformer/VAE must use16 latent channels")
        if type(pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise TypeError("the declared FlowMatchEulerDiscreteScheduler is required")
        for name, expected_type in (("text_encoder", CLIPTextModelWithProjection),
                                    ("text_encoder_2", CLIPTextModelWithProjection)):
            if type(getattr(pipe, name)) is not expected_type:
                raise TypeError("a real declared text encoder architecture is required: " + name)
        scheduler_config = dict(pipe.scheduler.config)
        for key in ("use_dynamic_shifting", "stochastic_sampling", "invert_sigmas"):
            if scheduler_config.get(key, False):
                raise ValueError("undeclared scheduler mode: " + key)
        self.pipe = pipe
        self.config = config
        self.counts = TwoClipCallCounts()
        self._scheduler_type = FlowMatchEulerDiscreteScheduler
        self._scheduler_config = scheduler_config
        self._scheduler = None
        self._condition = None
        self._condition_metadata = None
        self._next_step = 0
        self._round_index = -1
        self._timesteps = None
        self._latent_binding = None
        self._closed = False
        self._handles = []
        for name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
            module = getattr(pipe, name)
            module.requires_grad_(False)
            module.eval()
        modules = {"text_encoder": pipe.text_encoder, "text_encoder_2": pipe.text_encoder_2,
                   "transformer": pipe.transformer,
                   "vae_encoder": pipe.vae.encoder, "vae_decoder": pipe.vae.decoder}
        for name, module in modules.items():
            self._handles.append(module.register_forward_hook(self._observe(name)))

    def _observe(self, name):
        def hook(module, args, output):
            self.counts.module_forward_completed[name] += 1
        return hook

    def _open(self):
        if self._closed:
            raise RuntimeError("kernel observation scope is closed")

    def encode_condition(self, device):
        """Run exactly the two loaded CLIPs; official zero-T5 block is verified.

        No provided/cached embeddings or token-ID interception are used. A
        missing/failed CLIP aborts normally rather than falling back to zeros.
        """
        self._open()
        require_two_clip_component_presence(self.pipe)
        if self._condition is not None:
            raise RuntimeError("condition already encoded; no silent repeated encode")
        torch = _torch()
        before = dict(self.counts.module_forward_completed)
        self.counts.encode_condition_attempted += 1
        with torch.no_grad():
            values = self.pipe.encode_prompt(
                prompt=self.config.prompt, prompt_2=self.config.prompt,
                prompt_3=None,
                negative_prompt=self.config.negative_prompt,
                negative_prompt_2=self.config.negative_prompt,
                negative_prompt_3=None,
                do_classifier_free_guidance=True, num_images_per_prompt=1,
                device=device, max_sequence_length=self.config.max_sequence_length,
                clip_skip=None)
        self.counts.encode_condition_completed += 1
        if not isinstance(values, tuple) or len(values) != 4:
            raise ValueError("official SD3 encode_prompt must return four tensors")
        for name in ("text_encoder", "text_encoder_2"):
            if self.counts.module_forward_completed[name] <= before[name]:
                raise RuntimeError("no actual forward observed for " + name)
        positive, negative, pooled, negative_pooled = values
        if any(not isinstance(x, torch.Tensor) or x.shape[0] != 1 or
               x.numel() == 0 or not bool(torch.isfinite(x).all()) for x in values):
            raise ValueError("finite one-image positive and negative embeddings required")
        # Official v0.38.0 concatenates the CLIP sequence with the T5
        # sequence along dim1. In this explicit mode that last256-token block
        # is created by its documented text_encoder_3-is-None zero branch.
        # Check actual returned values; do not manufacture an embedding here.
        expected_clip_length = int(self.pipe.tokenizer_max_length)
        expected_width = int(self.pipe.transformer.config.joint_attention_dim)
        expected_shape = (1, expected_clip_length + self.config.max_sequence_length, expected_width)
        if tuple(positive.shape) != expected_shape or tuple(negative.shape) != expected_shape:
            raise ValueError("official two-CLIP plus zero-T5 sequence shape mismatch")
        positive_tail = positive[:, -self.config.max_sequence_length:, :]
        negative_tail = negative[:, -self.config.max_sequence_length:, :]
        if bool((positive_tail != 0).any()) or bool((negative_tail != 0).any()):
            raise ValueError("explicit no-T5 mode did not return the exact all-zero T5 sequence")
        if not bool((positive[:, :expected_clip_length, :] != 0).any()) or not bool((negative[:, :expected_clip_length, :] != 0).any()):
            raise ValueError("actual CLIP conditional sequence must not be an all-zero replacement")
        self._condition = (torch.cat((negative, positive)).detach(),
                           torch.cat((negative_pooled, pooled)).detach())
        self._condition_metadata = {
            "actual_embedding_descriptors": {
                "positive": embedding_descriptor(positive), "negative": embedding_descriptor(negative),
                "positive_pooled": embedding_descriptor(pooled),
                "negative_pooled": embedding_descriptor(negative_pooled),
                "cfg_negative_then_positive": embedding_descriptor(self._condition[0]),
                "cfg_negative_then_positive_pooled": embedding_descriptor(self._condition[1])},
            "identity": IDENTITY,
            "conditioning_mode": CONDITIONING_MODE,
            "T5_enabled": False,
            "T5_execution": "not_executed",
            "T5_adaptation": "official_zero_T5_sequence_adaptation",
            "full_T5_conditioning_or_effect_equivalence_claim": False,
            "explicit_disabled_components": {"text_encoder_3": None, "tokenizer_3": None},
            "actual_zero_T5_sequence_check": {
                "sequence_length": self.config.max_sequence_length,
                "channel_count": expected_width, "clip_sequence_length": expected_clip_length,
                "positive_tail_all_zero": True, "negative_tail_all_zero": True,
                "zero_tensor_constructed_by_this_kernel": False},
            "actual_tokenizer_classes": {
                k: type(getattr(self.pipe, k)).__module__ + "." + type(getattr(self.pipe, k)).__qualname__
                for k in ("tokenizer", "tokenizer_2")},
            "actual_encoder_classes": {
                k: type(getattr(self.pipe, k)).__module__ + "." + type(getattr(self.pipe, k)).__qualname__
                for k in ("text_encoder", "text_encoder_2")},
            "token_ids_collected": False,
            "token_id_scope": "official encode_prompt internal tokenization; no token interception or token-ID export"}
        return {"application_call": "pipe.encode_prompt", "cached_embedding_input": False,
                **self._condition_metadata,
                "positive_shape": list(positive.shape), "negative_shape": list(negative.shape),
                "pooled_shape": list(pooled.shape),
                "actual_module_forward_deltas": {k: self.counts.module_forward_completed[k] - before[k]
                    for k in ("text_encoder", "text_encoder_2")}}

    def _patch(self, patch):
        torch = _torch()
        if (not isinstance(patch, torch.Tensor) or tuple(patch.shape) != (1, 3, 512, 512)
                or not patch.is_floating_point() or not bool(torch.isfinite(patch).all())
                or bool((patch < 0).any()) or bool((patch > 1).any())):
            raise ValueError("caller must supply an admitted RGB512 patch in [0,1]; no resize here")

    def start_round(self, patch, round_index: int):
        """Re-encode posterior MODE, reset scheduler, then mix its actual sigma.

        A connectivity check may stop after a declared prefix. It must not start
        another round or claim25/500 completion from that partial prefix.
        """
        self._open()
        torch = _torch()
        if self._condition is None:
            raise RuntimeError("encode the real road condition before a round")
        if type(round_index) is not int or round_index != self._round_index + 1:
            raise ValueError("outer indices must be exact sequential integers from0")
        if self._round_index >= 0 and self._next_step != self.config.active_steps:
            raise RuntimeError("prior round is incomplete; no silent reset")
        self._patch(patch)
        self._scheduler = self._scheduler_type.from_config(self._scheduler_config)
        self._scheduler.set_timesteps(self.config.num_inference_steps, device=patch.device)
        self._scheduler.set_begin_index(self.config.begin_index)
        timesteps = self._scheduler.timesteps[self.config.begin_index:]
        if len(timesteps) != 25 or len(self._scheduler.sigmas) != 101:
            raise RuntimeError("actual scheduler does not supply exactly100+terminal sigma /25active steps")
        self.counts.vae_encode_attempted += 1
        before = self.counts.module_forward_completed["vae_encoder"]
        with torch.no_grad():
            source = patch.detach().to(dtype=self.pipe.vae.dtype) * 2 - 1
            posterior = self.pipe.vae.encode(source, return_dict=True).latent_dist
            latent = posterior.mode()
            scale = float(self.pipe.vae.config.scaling_factor)
            shift = float(self.pipe.vae.config.shift_factor)
            if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shift):
                raise ValueError("finite bound VAE scaling/shift required")
            latent = ((latent - shift) * scale).to(dtype=self.pipe.transformer.dtype)
        self.counts.vae_encode_completed += 1
        if tuple(latent.shape) != (1, 16, 64, 64) or not bool(torch.isfinite(latent).all()):
            raise ValueError("the declared RGB512 VAE must produce finite1x16x64x64 latents")
        if self.counts.module_forward_completed["vae_encoder"] <= before:
            raise RuntimeError("no actual VAE encoder.forward was observed")
        sigma = self._scheduler.sigmas[self.config.begin_index].to(device=latent.device, dtype=latent.dtype)
        if not bool(torch.isfinite(sigma)) or not 0 < float(sigma) < 1:
            raise ValueError("actual starting sigma must be finite in (0,1)")
        generator = torch.Generator(device=latent.device).manual_seed(self.config.seed + 1000 * round_index)
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        self.counts.noise_draws_completed += 1
        latent = ((1 - sigma) * latent + sigma * noise).detach()
        if not bool(torch.isfinite(latent).all()):
            raise FloatingPointError("nonfinite round restart latent")
        self._latent_binding = (tuple(latent.shape), latent.device, latent.dtype)
        self._round_index = round_index
        self._next_step = 0
        self._timesteps = timesteps.detach().clone()
        self.counts.round_starts_completed += 1
        return latent, self._timesteps.clone()

    def generated_leaf(self, latent, timestep):
        """One real CFG Transformer + scheduler step, then detach/new leaf."""
        self._open()
        torch = _torch()
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("attack caller must enable grad outside frozen generation")
        if self._scheduler is None or self._next_step >= 25:
            raise RuntimeError("no pending inner step")
        expected = self._timesteps[self._next_step]
        if (not isinstance(timestep, torch.Tensor) or timestep.ndim != 0 or
                timestep.dtype != expected.dtype or timestep.device != expected.device or
                not torch.equal(timestep, expected)):
            raise ValueError("timestep is not the exact next scheduled value")
        if (not isinstance(latent, torch.Tensor) or
                (tuple(latent.shape), latent.device, latent.dtype) != self._latent_binding or
                not latent.is_floating_point() or not bool(torch.isfinite(latent).all())):
            raise ValueError("finite latent with this round's exact shape/device/dtype required")
        condition, pooled = self._condition
        with torch.no_grad():
            model_input = torch.cat((latent.detach(), latent.detach()), dim=0)
            condition = condition.to(device=latent.device, dtype=self.pipe.transformer.dtype)
            pooled = pooled.to(device=latent.device, dtype=self.pipe.transformer.dtype)
            before = self.counts.module_forward_completed["transformer"]
            self.counts.transformer_attempted += 1
            prediction = self.pipe.transformer(
                hidden_states=model_input, timestep=timestep.expand(model_input.shape[0]),
                encoder_hidden_states=condition, pooled_projections=pooled,
                joint_attention_kwargs=None, return_dict=False)[0]
            self.counts.transformer_completed += 1
            if self.counts.module_forward_completed["transformer"] != before + 1:
                raise RuntimeError("exactly one actual Transformer.forward required per step")
            if prediction.shape != model_input.shape or not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("Transformer prediction shape/finite mismatch")
            guided = _cfg_merge(prediction, self.config.guidance_scale)
            self.counts.scheduler_step_attempted += 1
            updated = self._scheduler.step(guided, timestep, latent.detach(), return_dict=False)[0]
            self.counts.scheduler_step_completed += 1
            if updated.shape != latent.shape or not bool(torch.isfinite(updated).all()):
                raise FloatingPointError("scheduler output shape/finite mismatch")
        self._next_step += 1
        return _attack_leaf(updated)

    def decode_leaf(self, latent):
        """VAE parameters frozen; gradients through decoder to latent preserved."""
        self._open()
        torch = _torch()
        if latent.requires_grad and not torch.is_grad_enabled():
            raise RuntimeError("live latent decoder must run with gradients enabled")
        scale = float(self.pipe.vae.config.scaling_factor)
        shift = float(self.pipe.vae.config.shift_factor)
        self.counts.vae_decode_attempted += 1
        before = self.counts.module_forward_completed["vae_decoder"]
        decoded = self.pipe.vae.decode(latent.to(dtype=self.pipe.vae.dtype) / scale + shift,
                                       return_dict=False)[0]
        self.counts.vae_decode_completed += 1
        if self.counts.module_forward_completed["vae_decoder"] <= before:
            raise RuntimeError("no actual VAE decoder.forward observed")
        patch = (decoded / 2 + .5).clamp(0, 1)
        if tuple(patch.shape) != (1, 3, 512, 512) or not bool(torch.isfinite(patch).all()):
            raise FloatingPointError("decoded patch shape/finite mismatch")
        if latent.requires_grad and (not patch.requires_grad or patch.grad_fn is None):
            raise RuntimeError("decoder detached the attack gradient path")
        return patch

    def observation(self):
        self._open()
        return {"schema": "sd35-two-clip-no-t5-kernel-observation-v1", "identity": IDENTITY,
                "conditioning_mode": CONDITIONING_MODE,
                "T5_enabled": False, "T5_execution": "not_executed",
                "T5_adaptation": "official_zero_T5_sequence_adaptation",
                "full_T5_conditioning_or_effect_equivalence_claim": False,
                "scope": "kernel_calls_observed_in_this_process_only_not_science_acceptance",
                "application_counts_and_actual_module_hooks_are_distinct": True,
                "pipeline_class": type(self.pipe).__module__ + "." + type(self.pipe).__qualname__,
                "config": asdict(self.config), "calls": asdict(self.counts),
                "condition_metadata": self._condition_metadata,
                "last_round_index": self._round_index, "completed_inner_steps_in_round": self._next_step,
                "four_surrogate_integration_verified_by_this_kernel": False,
                "whole_snapshot_or_GPU_admission_verified_by_this_kernel": False}

    def close(self):
        """Remove only hooks installed by this kernel; no model/process cleanup."""
        if self._closed:
            raise RuntimeError("kernel already closed")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True
