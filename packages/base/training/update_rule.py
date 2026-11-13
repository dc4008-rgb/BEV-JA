"""Frozen C0 U1 final-injection clipping with one package-bound nominal learning rate."""
from __future__ import annotations
import math
import torch
if __package__:
    from .inputs import LEARNING_RATE, VARIANT_LABEL
else:
    from inputs import LEARNING_RATE, VARIANT_LABEL

CAP = 0.1767578125
VARIANTS = ('U1',)


def make_update(latent, joint_gradient, lr, variant):
    if variant not in VARIANTS:
        raise ValueError('this learning-rate candidate permits only U1')
    if type(lr) is not float or lr != LEARNING_RATE:
        raise ValueError('package-bound latent learning rate required: ' + str(LEARNING_RATE))
    if (latent.dtype not in (torch.bfloat16, torch.float32) or
            joint_gradient.dtype != torch.float32 or latent.shape != joint_gradient.shape or
            latent.device != joint_gradient.device or latent.numel() == 0):
        raise ValueError('invalid update tensor contract')
    if not torch.isfinite(latent).all() or not torch.isfinite(joint_gradient).all():
        raise ValueError('nonfinite update input')
    # Preserve v23 multiplication/cast order exactly. No post-clamp rescaling.
    cast = joint_gradient.detach().to(latent)
    proposed = lr * cast
    applied = proposed.clamp(-CAP, CAP)
    after = (latent.detach() - applied).detach()
    if not torch.isfinite(after).all():
        raise ValueError('nonfinite updated latent')
    nominal = proposed.detach().float()
    injected = applied.detach().float()
    realized = latent.detach().float() - after.float()
    clipped = proposed != applied
    if bool((injected.abs() > CAP).any()):
        raise ValueError('nominal injection cap exceeded')
    def rms(x):
        return float(x.double().square().mean().sqrt())
    info = {
        'mathematical_objective': 'C0', 'update_variant': variant,
        'rule': 'component_clip_after_original_bf16_multiply',
        'latent_lr': LEARNING_RATE, 'learning_rate_label': VARIANT_LABEL,
        'cap': CAP,
        'renormalized_after_clip': False,
        'proposed_rms': rms(nominal), 'proposed_linf': float(nominal.abs().max()),
        'applied_rms': rms(injected), 'applied_linf': float(injected.abs().max()),
        'realized_rms': rms(realized), 'realized_linf': float(realized.abs().max()),
        'clipped_count': int(clipped.sum()), 'numel': latent.numel(),
        'realized_zero_count': int((realized == 0).sum()),
        'realized_over_nominal_cap_count': int((realized.abs() > CAP).sum()),
        'cap_applies_to': 'injection_before_latent_subtraction; bf16 realized difference may exceed cap',
    }
    return after, {'proposed_step': proposed, 'applied_step': applied,
                   'realized_delta_float32': realized, 'update_statistics': info}


def validate_saved_update(pre, post, variant):
    """Exact CPU arithmetic check; no claim about CUDA loss/gradient replay."""
    after, evidence = make_update(pre['latent_variable'], pre['joint_gradient'],
                                  pre['latent_lr'], variant)
    for name in ('proposed_step', 'applied_step', 'realized_delta_float32'):
        if name not in pre or not torch.equal(evidence[name], pre[name]):
            raise ValueError('saved update tensor differs: ' + name)
    if not torch.equal(after, post['latents']):
        raise ValueError('saved updated latent differs')
    if (pre.get('update_statistics') != post.get('update_statistics') or
            pre['update_statistics'].get('update_variant') != variant or
            pre['update_statistics'].get('mathematical_objective') != 'C0' or
            pre['update_statistics'].get('cap') != CAP or
            type(pre['update_statistics'].get('latent_lr')) is not float or
            pre['update_statistics']['latent_lr'] != LEARNING_RATE or
            pre['update_statistics'].get('learning_rate_label') != VARIANT_LABEL or
            pre['update_statistics'].get('renormalized_after_clip') is not False):
        raise ValueError('saved update rule identity differs')
    return True
