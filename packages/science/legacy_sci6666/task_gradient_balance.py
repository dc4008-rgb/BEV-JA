"""Detached FP32 mixing of the two already model-weighted task gradients.

No loss is recomputed and no VJP is called here. The reference is exactly the
FP32 sum of the supplied current task gradients, with task coefficients [1,1];
it is not a fresh model-equal-weight gradient. Priorities are detached requested
probabilities fixed by the caller's cycle. Only the effective mixing proportion
can change at each step when the two current directions conflict.

``unit_direction`` mixes normalized active task gradients. ``raw_sum`` means
2*p_det*g_det + 2*p_depth*g_depth, so fixed [.5,.5] is the plain FP32 input sum.
With raw_sum conflict_guard enabled, the interval is applied to the equivalent
unit proportion p_det*||g_det||/(p_det*||g_det||+p_depth*||g_depth||). A changed
proportion preserves the original coefficient scale 2*sum(p_q*||g_q||). If no
clipping occurs, the original raw expression is retained. Thus guard and norm
matching remain independent switches; an enabled guard can alter raw_sum.

Single-active-task mixing uses its unit direction or its unique raw gradient,
independently of requested priority. Tasks with norm <= n_min are inactive.
A mixed direction with norm <= n_min is returned as exact zeros, never amplified.
Only exact zero reference norm forces zeros under norm_match; a small nonzero
reference shrinks a valid direction. Without norm_match, reference cancellation
does not discard a valid direction. First-order descent is analytically >=0
under the guard, not strictly >0. Actual FP32 dot products and conservative
roundoff bounds are logged rather than being clamped into positive values.

At a negative-cosine interval endpoint, FP32 unit-length/mixing rounding can
put the actual dot on the wrong side of zero, especially before norm matching
amplifies a near-cancelled mixture. Such a boundary is repaired only with FP32
representable alpha values inside the original analytic interval: at most 32
one-ULP moves toward the task whose actual dot is negative, then one interval
midpoint trial. A trial must have both actual FP32 dots nonnegative (or satisfy
the unchanged h <= n_min zero rule). No verified trial means exact zero with
an explicit numerical reason. This does not enlarge n_min, widen the interval,
change the reference, or claim that a finite BF16 update decreases both losses.

Plain-sum parity here is for two supplied FP32 task gradients. It does not assert
bitwise parity between historical single-VJP BF16 and new two-VJP execution.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch


def _finite_norm(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.linalg.vector_norm(value.reshape(-1))
    if not bool(torch.isfinite(result)):
        raise FloatingPointError(name + ' FP32 norm is nonfinite')
    return result


def _priorities(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if tuple(value.shape) != (2,) or not value.is_floating_point():
            raise ValueError('priorities must be a floating tensor with shape(2,)')
        result = value.detach().to(device=device, dtype=torch.float32)
    else:
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
                isinstance(v, bool) or not isinstance(v, Real) for v in value):
            raise ValueError('priorities must be two real probabilities')
        result = torch.tensor(value, device=device, dtype=torch.float32)
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()) or bool((result > 1).any()):
        raise ValueError('finite probabilities in [0,1] required')
    # Do not silently change requested probabilities. The allowance is only the
    # sum rounding of the existing FP32 softmax/EMA controller.
    if abs(float(result.sum()) - 1.0) > 8.0 * torch.finfo(torch.float32).eps:
        raise ValueError('requested task priorities must sum to one')
    return result


def combine_task_gradients(g_det, g_depth, priorities, *,
                           task_balance='unit_direction', conflict_guard=True,
                           norm_match=True, n_min=1e-8):
    """Return ``(detached FP32 gradient, JSON-safe details)`` without any VJP.

    ``alpha_requested`` is the pre-guard unit mixing proportion: p_det for unit
    balancing, or alpha_raw for raw_sum. ``task_priorities_requested`` separately
    preserves the caller's cycle request. All norms, cosines, direction mixes
    and scale ratios are computed in FP32 on the input device.
    """
    if task_balance not in ('unit_direction', 'raw_sum'):
        raise ValueError('task_balance must be unit_direction or raw_sum')
    if type(conflict_guard) is not bool or type(norm_match) is not bool:
        raise TypeError('conflict_guard and norm_match must be independent booleans')
    if isinstance(n_min, bool) or not isinstance(n_min, Real) or not math.isfinite(n_min) or n_min <= 0:
        raise ValueError('n_min must be finite and positive')
    for name, gradient in (('detection', g_det), ('depth', g_depth)):
        if not isinstance(gradient, torch.Tensor) or not gradient.is_floating_point() or not gradient.numel():
            raise TypeError(name + ' requires a nonempty floating gradient tensor')
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(name + ' task gradient is nonfinite')
    if g_det.shape != g_depth.shape or g_det.device != g_depth.device:
        raise ValueError('task gradients must share shape and device')
    with torch.no_grad():
        gd, gp = g_det.detach().float(), g_depth.detach().float()
        if not bool(torch.isfinite(gd).all() and torch.isfinite(gp).all()):
            raise FloatingPointError('task gradient conversion to FP32 overflowed')
        p = _priorities(priorities, gd.device)
        nd, np = _finite_norm(gd, 'detection'), _finite_norm(gp, 'depth')
        reference = gd + gp
        nr = _finite_norm(reference, 'reference task sum')
        active_det, active_depth = float(nd) > n_min, float(np) > n_min
        details = {
            'schema': 'periodic-foa-task-gradient-balance-v1',
            'task_balance': task_balance, 'conflict_guard': conflict_guard,
            'norm_match': norm_match, 'n_min': float(n_min),
            'input_dtypes': [str(g_det.dtype), str(g_depth.dtype)],
            'output_dtype': 'torch.float32', 'output_detached': True,
            'task_priorities_requested': p.cpu().tolist(),
            'task_priorities_requested_sum': float(p.sum()),
            'task_gradient_norms': {'det': float(nd), 'depth': float(np)},
            'active_tasks': {'det': active_det, 'depth': active_depth},
            'cosine': None, 'alpha_requested': None, 'alpha_effective': None,
            'alpha_raw': None, 'conflict_interval': None,
            'conflict_guard_clipped': False, 'raw_coefficient_total_scale': None,
            'reference_gradient_norm': float(nr), 'reference_task_coefficients': [1.0, 1.0],
            'reference_uses_current_model_weighted_task_gradients': True,
            'model_weights_recomputed': False, 'additional_vjp_calls': 0,
            'analytical_guard_applies_to_two_active_tasks': conflict_guard and active_det and active_depth,
            'analytical_unit_first_order_margins': None,
            'flat_numel': gd.numel(), 'norm_match_scale': None,
            'numerical_repair_applied': False, 'numerical_repair_attempts': 0,
            'numerical_repair_attempt_limit': 33, 'numerical_reason': None,
            'numerical_zero_fallback': False,
        }
        ud = gd / nd if active_det else None
        up = gp / np if active_depth else None
        if not active_det and not active_depth:
            h, reason = torch.zeros_like(gd), 'both_tasks_inactive'
        elif active_det != active_depth:
            single_det = active_det
            h = (ud if single_det else up) if task_balance == 'unit_direction' else (gd if single_det else gp)
            details['alpha_effective'] = 1.0 if single_det else 0.0
            reason = 'single_active_det' if single_det else 'single_active_depth'
        else:
            cosine = torch.dot(ud.reshape(-1), up.reshape(-1)).clamp(-1.0, 1.0)
            c = float(cosine)
            details['cosine'] = c
            if task_balance == 'raw_sum':
                scale_half = p[0] * nd + p[1] * np
                if not bool(torch.isfinite(scale_half)) or float(scale_half) <= 0.0:
                    raise FloatingPointError('active raw_sum contribution scale is invalid')
                alpha = p[0] * nd / scale_half
                details['alpha_raw'] = float(alpha)
                details['raw_coefficient_total_scale'] = float(2.0 * scale_half)
            else:
                alpha = p[0]
            details['alpha_requested'] = float(alpha)
            effective = alpha
            if c < 0.0:
                lower, upper = -cosine / (1.0 - cosine), 1.0 / (1.0 - cosine)
                details['conflict_interval'] = [float(lower), float(upper)]
                if conflict_guard:
                    effective = alpha.clamp(min=lower, max=upper)
            clipped = float(effective) != float(alpha)
            details['conflict_guard_clipped'] = clipped
            details['alpha_effective'] = float(effective)
            if task_balance == 'raw_sum' and not clipped:
                # Fixed [.5,.5] is exactly this two-FP32-input plain sum.
                h = (2.0 * p[0]) * gd + (2.0 * p[1]) * gp
                reason = 'raw_sum_unprojected'
            else:
                h = effective * ud + (1.0 - effective) * up
                if task_balance == 'raw_sum':
                    h = (2.0 * scale_half) * h
                    reason = 'raw_sum_conflict_projected_preserved_scale'
                else:
                    reason = 'unit_direction_conflict_projected' if clipped else 'unit_direction_unprojected'
            a = float(effective)
            details['analytical_unit_first_order_margins'] = {
                'det': a + (1.0 - a) * c, 'depth': 1.0 - a + a * c}
        nh = _finite_norm(h, 'mixed task direction')
        details['mixed_gradient_norm'] = float(nh)
        if not active_det and not active_depth:
            result = torch.zeros_like(gd)
        elif float(nh) <= n_min:
            result, reason = torch.zeros_like(gd), 'near_zero_mixed_direction'
        elif norm_match and float(nr) == 0.0:
            result, reason = torch.zeros_like(gd), 'exact_zero_reference_under_norm_match'
            details['norm_match_scale'] = 0.0
        elif norm_match:
            ratio = nr / nh
            if not bool(torch.isfinite(ratio)):
                raise FloatingPointError('FP32 norm-match scale is nonfinite')
            result = h * ratio
            details['norm_match_scale'] = float(ratio)
        else:
            result = h.clone()
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('combined FP32 task gradient is nonfinite')
        nu = _finite_norm(result, 'used task gradient')
        dots = {'det': float(torch.dot(gd.reshape(-1), result.reshape(-1))),
                'depth': float(torch.dot(gp.reshape(-1), result.reshape(-1)))}
        if not all(math.isfinite(v) for v in dots.values()):
            raise FloatingPointError('first-order dot product is nonfinite')
        if conflict_guard and active_det and active_depth and details['cosine'] < 0.0 and any(v < 0.0 for v in dots.values()):
            # Step order matters: test the actual post-scale output rather than
            # only the ideal unit Gram formula. Retain the requested priority,
            # interval and raw total coefficient scale; never change n_min.
            details['numerical_repair_applied'] = True
            details['alpha_before_numerical_repair'] = float(effective)
            details['mixed_norm_before_numerical_repair'] = float(nh)
            lower, upper = effective.new_tensor(details['conflict_interval'])
            midpoint = effective.new_tensor(.5).clamp(min=lower, max=upper)
            candidate_alpha, candidate_dots = effective, dots
            visited = {float(effective)}
            repaired = False
            for attempt in range(33):
                if attempt == 32:
                    trial_alpha = midpoint
                else:
                    # Increasing alpha improves the detection dot; decreasing
                    # alpha improves the depth dot. Each move stays in the
                    # originally requested mathematical conflict interval.
                    target = upper if candidate_dots['det'] < 0.0 else lower
                    trial_alpha = torch.nextafter(candidate_alpha, target)
                if float(trial_alpha) in visited:
                    if attempt < 32:
                        continue
                    break
                visited.add(float(trial_alpha))
                details['numerical_repair_attempts'] += 1
                if not float(lower) <= float(trial_alpha) <= float(upper):
                    raise FloatingPointError('numerical repair escaped the analytic interval')
                trial_h = trial_alpha * ud + (1.0 - trial_alpha) * up
                if task_balance == 'raw_sum':
                    trial_h = (2.0 * scale_half) * trial_h
                trial_nh = _finite_norm(trial_h, 'numerically repaired direction')
                if float(trial_nh) <= n_min:
                    trial_result = torch.zeros_like(gd)
                    trial_ratio = None
                    trial_reason = 'near_zero_mixed_direction'
                elif norm_match and float(nr) == 0.0:
                    trial_result, trial_ratio = torch.zeros_like(gd), 0.0
                    trial_reason = 'exact_zero_reference_under_norm_match'
                elif norm_match:
                    trial_scale = nr / trial_nh
                    if not bool(torch.isfinite(trial_scale)):
                        raise FloatingPointError('repaired FP32 norm-match scale is nonfinite')
                    trial_result, trial_ratio = trial_h * trial_scale, float(trial_scale)
                    trial_reason = 'fp32_conflict_boundary_repaired'
                else:
                    trial_result, trial_ratio = trial_h, None
                    trial_reason = 'fp32_conflict_boundary_repaired'
                trial_nu = _finite_norm(trial_result, 'numerically repaired output')
                trial_dots = {
                    'det': float(torch.dot(gd.reshape(-1), trial_result.reshape(-1))),
                    'depth': float(torch.dot(gp.reshape(-1), trial_result.reshape(-1)))}
                if not all(math.isfinite(v) for v in trial_dots.values()):
                    raise FloatingPointError('repaired first-order dot product is nonfinite')
                candidate_alpha, candidate_dots = trial_alpha, trial_dots
                if all(v >= 0.0 for v in trial_dots.values()):
                    effective, h, nh = trial_alpha, trial_h, trial_nh
                    result, nu, dots, reason = trial_result, trial_nu, trial_dots, trial_reason
                    details['norm_match_scale'] = trial_ratio
                    details['alpha_effective'] = float(effective)
                    details['mixed_gradient_norm'] = float(nh)
                    details['conflict_guard_clipped'] = float(effective) != details['alpha_requested']
                    details['numerical_reason'] = 'verified_representable_alpha_inside_original_interval'
                    repaired = True
                    break
            if not repaired:
                result, nu = torch.zeros_like(gd), gd.new_zeros(())
                dots = {'det': 0.0, 'depth': 0.0}
                reason = 'no_verified_direction_in_bounded_fp32_repair'
                details['numerical_reason'] = 'exact_zero_safe_fallback_no_interval_widening_or_n_min_change'
                details['numerical_zero_fallback'] = True
                details['norm_match_scale'] = None
            else:
                a, c = float(effective), details['cosine']
                details['analytical_unit_first_order_margins'] = {
                    'det': a + (1.0 - a) * c, 'depth': 1.0 - a + a * c}
        # FP32 dot/norm/mix rounding is not an exact strictly-positive theorem.
        # Cauchy bounds sum(abs(x_i*y_i)); gamma includes reduction and scalar
        # norm/mix operations. A too-large dimension is disclosed as unbounded.
        eps = torch.finfo(torch.float32).eps
        rounding_terms = gd.numel() + 64
        gamma = rounding_terms * eps / (1.0 - rounding_terms * eps) if rounding_terms * eps < 1.0 else None
        bounds = {name: (gamma * norm * float(nu) if gamma is not None else None)
                  for name, norm in (('det', float(nd)), ('depth', float(np)))}
        conditions = {name: (dots[name] >= -bounds[name] if bounds[name] is not None else None)
                      for name in dots}
        details.update(reason=reason, used_gradient_norm=float(nu),
            output_exact_zero=bool(torch.count_nonzero(result) == 0),
            first_order_dots=dots, first_order_roundoff_bounds=bounds,
            first_order_ge_zero_within_roundoff=conditions,
            analytical_guard_is_nonnegative_not_strictly_positive=True)
        if conflict_guard and active_det and active_depth and any(v is False for v in conditions.values()):
            raise FloatingPointError('guarded first-order descent violated the logged FP32 roundoff bound')
        return result.detach(), details
