"""CPU tests of task directions, independent switches and inactive boundaries."""
import json
import math
import unittest
from unittest.mock import patch

import torch

from task_gradient_balance import combine_task_gradients


class TaskGradientBalanceTests(unittest.TestCase):
    def mix(self, gd, gp, priorities=(.5, .5), **kwargs):
        return combine_task_gradients(torch.tensor(gd, dtype=torch.float32),
            torch.tensor(gp, dtype=torch.float32), priorities, **kwargs)

    def test_positive_and_orthogonal_directions_keep_requested_priority(self):
        for gp in ([2., 1.], [0., 5.]):
            result, log = self.mix([3., 0.], gp, (.25, .75), norm_match=False)
            expected = .25 * torch.tensor([1., 0.]) + .75 * torch.tensor(gp) / torch.linalg.vector_norm(torch.tensor(gp))
            self.assertTrue(torch.allclose(result, expected))
            self.assertEqual(log['alpha_requested'], .25)
            self.assertEqual(log['alpha_effective'], .25)
            self.assertFalse(log['conflict_guard_clipped'])
            self.assertGreaterEqual(log['first_order_dots']['det'], 0.0)
            self.assertGreaterEqual(log['first_order_dots']['depth'], 0.0)

    def test_negative_conflict_clips_at_both_analytic_boundaries(self):
        for priority, expected in ((.9, 1 / 1.8), (.1, .8 / 1.8)):
            result, log = self.mix([3., 0.], [-4., 3.], (priority, 1-priority), norm_match=False)
            self.assertAlmostEqual(log['cosine'], -.8, places=6)
            self.assertAlmostEqual(log['alpha_effective'], expected, places=6)
            self.assertTrue(log['conflict_guard_clipped'])
            self.assertTrue(all(log['first_order_ge_zero_within_roundoff'].values()))
            self.assertFalse(log['output_exact_zero'])
            self.assertTrue(torch.isfinite(result).all())

    def test_negative_conflict_interior_priority_does_not_change(self):
        _, log = self.mix([3., 0.], [-4., 3.], norm_match=False)
        self.assertEqual(log['alpha_effective'], .5)
        self.assertFalse(log['conflict_guard_clipped'])

    def test_opposite_cosine_minus_one_produces_exact_zero(self):
        for balance in ('unit_direction', 'raw_sum'):
            result, log = self.mix([3., 0.], [-5., 0.], (.9, .1), task_balance=balance)
            self.assertEqual(log['cosine'], -1.0)
            self.assertEqual(log['conflict_interval'], [.5, .5])
            self.assertTrue(torch.equal(result, torch.zeros_like(result)))
            self.assertEqual(log['reason'], 'near_zero_mixed_direction')
            self.assertEqual(log['norm_match_scale'], None)

    def test_near_opposite_near_zero_direction_is_never_amplified(self):
        result, log = self.mix([1., 1e-9], [-1., 0.], (.9, .1))
        self.assertTrue(torch.equal(result, torch.zeros(2)))
        self.assertLessEqual(log['mixed_gradient_norm'], 1e-8)
        self.assertEqual(log['reason'], 'near_zero_mixed_direction')
        self.assertIsNone(log['norm_match_scale'])

    @staticmethod
    def integration_boundary_gradients():
        # Captured from the actual finite two-task CPU integration fixture.
        # Unequal FP32 unit self-norms made the former ideal upper endpoint
        # produce a negative depth dot after reference norm matching.
        gd, gp = torch.zeros(1, 16, 4, 4), torch.zeros(1, 16, 4, 4)
        gd[0, 0] = torch.tensor([
            .00044900886132381856, .00044586620060727, .0004514623142313212, .0004583247937262058,
            .00044405445805750787, .00046746578300371766, .0004589151358231902, .0004543948161881417,
            .0004544096009340137, .00045342932571657, .000450017920229584, .00044916305341757834,
            .00045357359340414405, .0004538263601716608, .0004500536888372153, .00044931602315045893]).reshape(4, 4)
        gd[0, 1:3] = 7.903424261712644e-7
        gp[0, 0] = torch.tensor([
            -3.523764689816744e-6, -3.4991014672414167e-6, -3.543019147400628e-6, -3.5968753309134627e-6,
            -3.484883109194925e-6, -3.6686124076368287e-6, -3.601508296924294e-6, -3.56603300133429e-6,
            -3.5661491892824415e-6, -3.5584560009738198e-6, -3.5316838875587564e-6, -3.5249749998911284e-6,
            -3.559588321877527e-6, -3.5615721571957693e-6, -3.5319644666742533e-6, -3.5261755328974687e-6]).reshape(4, 4)
        return gd, gp

    def test_actual_integration_endpoint_roundoff_is_repaired_inside_interval(self):
        gd, gp = self.integration_boundary_gradients()
        result, log = combine_task_gradients(gd, gp, [.5270181561690589, .4729818438309412])
        self.assertTrue(log['numerical_repair_applied'])
        self.assertFalse(log['numerical_zero_fallback'])
        self.assertGreater(log['numerical_repair_attempts'], 0)
        self.assertLessEqual(log['numerical_repair_attempts'], 33)
        lower, upper = log['conflict_interval']
        self.assertLessEqual(lower, log['alpha_effective'])
        self.assertLessEqual(log['alpha_effective'], upper)
        self.assertEqual(log['n_min'], 1e-8)
        self.assertTrue(all(v >= 0. for v in log['first_order_dots'].values()))
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(log['used_gradient_norm'] / float((gd + gp).norm()), 1., places=6)
        self.assertEqual(log['task_priorities_requested'], torch.tensor([.5270181561690589, .4729818438309412]).tolist())
        json.dumps(log, allow_nan=False)

    def test_exact_antiparallel_fp32_residual_has_safe_zero_not_amplification(self):
        # The mathematical rays are opposite. FP32 normalization leaves a
        # residual above n_min, so increasing n_min would be a wrong repair.
        gd = torch.full((1, 16, 4, 4), .0003159)
        gp = torch.full_like(gd, -.000022879)
        result, log = combine_task_gradients(gd, gp, [.7, .3])
        self.assertGreater(log['mixed_norm_before_numerical_repair'], 1e-8)
        self.assertTrue(log['numerical_zero_fallback'])
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertEqual(log['n_min'], 1e-8)
        self.assertEqual(log['reason'], 'no_verified_direction_in_bounded_fp32_repair')
        self.assertEqual(log['first_order_dots'], {'det': 0., 'depth': 0.})
        self.assertLessEqual(log['numerical_repair_attempts'], 33)
        self.assertEqual(log['reference_gradient_norm'], float((gd + gp).norm()))

    def test_raw_sum_numerical_repair_keeps_original_total_scale(self):
        gd, gp = self.integration_boundary_gradients()
        result, log = combine_task_gradients(gd, gp, [.5270181561690589, .4729818438309412],
            task_balance='raw_sum', norm_match=False)
        self.assertTrue(log['numerical_repair_applied'])
        self.assertFalse(log['numerical_zero_fallback'])
        p = torch.tensor([.5270181561690589, .4729818438309412])
        original_scale = float(2 * (p[0] * gd.norm() + p[1] * gp.norm()))
        self.assertEqual(log['raw_coefficient_total_scale'], original_scale)
        alpha = log['alpha_effective']
        ud, up = gd / gd.norm(), gp / gp.norm()
        expected = original_scale * (alpha * ud + (1-alpha) * up)
        self.assertTrue(torch.allclose(result, expected, atol=1e-12, rtol=1e-5))
        self.assertTrue(all(v >= 0. for v in log['first_order_dots'].values()))

    def test_disabled_guard_does_not_silently_repair_the_reproducer(self):
        gd, gp = self.integration_boundary_gradients()
        _, log = combine_task_gradients(gd, gp, [.5270181561690589, .4729818438309412], conflict_guard=False)
        self.assertFalse(log['numerical_repair_applied'])
        self.assertEqual(log['numerical_repair_attempts'], 0)

    def test_both_inactive_and_threshold_equal_are_zero(self):
        result, log = self.mix([1e-8, 0.], [0., 1e-8])
        self.assertEqual(log['active_tasks'], {'det': False, 'depth': False})
        self.assertEqual(log['reason'], 'both_tasks_inactive')
        self.assertTrue(torch.equal(result, torch.zeros(2)))
        # The reference sum can exceed n_min, but does not activate either task.
        self.assertGreater(log['reference_gradient_norm'], 1e-8)

    def test_one_active_unit_direction_ignores_its_zero_requested_priority(self):
        result, log = self.mix([3., 4.], [0., 0.], (0., 1.), norm_match=False)
        self.assertTrue(torch.allclose(result, torch.tensor([.6, .8])))
        self.assertEqual(log['active_tasks'], {'det': True, 'depth': False})
        self.assertEqual(log['alpha_effective'], 1.0)
        self.assertEqual(log['reason'], 'single_active_det')

    def test_one_active_raw_sum_uses_unique_raw_gradient(self):
        result, log = self.mix([0., 0.], [3., 4.], (1., 0.), task_balance='raw_sum', norm_match=False)
        self.assertTrue(torch.equal(result, torch.tensor([3., 4.])))
        self.assertEqual(log['alpha_effective'], 0.0)
        self.assertEqual(log['reason'], 'single_active_depth')

    def test_norm_match_uses_current_supplied_task_sum(self):
        result, log = self.mix([2., 0.], [0., 5.], (.2, .8))
        reference_norm = math.sqrt(29.)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result)), reference_norm, places=6)
        self.assertAlmostEqual(log['reference_gradient_norm'], reference_norm, places=6)
        self.assertEqual(log['reference_task_coefficients'], [1., 1.])
        self.assertTrue(log['reference_uses_current_model_weighted_task_gradients'])
        self.assertFalse(log['model_weights_recomputed'])

    def test_reference_cancellation_zeroes_only_when_norm_match_is_on(self):
        off, log_off = self.mix([1., 0.], [-1., 0.], (.8, .2), conflict_guard=False, norm_match=False)
        on, log_on = self.mix([1., 0.], [-1., 0.], (.8, .2), conflict_guard=False, norm_match=True)
        self.assertFalse(log_off['output_exact_zero'])
        self.assertTrue(torch.allclose(off, torch.tensor([.6, 0.])))
        self.assertTrue(torch.equal(on, torch.zeros(2)))
        self.assertEqual(log_on['reason'], 'exact_zero_reference_under_norm_match')

    def test_nonzero_tiny_reference_shrinks_not_thresholds_direction(self):
        result, log = self.mix([1., 0.], [-1., 1e-9], (.8, .2), conflict_guard=False)
        self.assertGreater(log['reference_gradient_norm'], 0.)
        self.assertLess(log['reference_gradient_norm'], 1e-8)
        self.assertFalse(log['output_exact_zero'])
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result)) / 1e-9, 1., places=5)

    def test_raw_sum_fixed_equal_is_exact_supplied_fp32_sum(self):
        gd = torch.tensor([1.234567, -3.125, 2.], dtype=torch.float32)
        gp = torch.tensor([2.34357, .125, 4.], dtype=torch.float32)
        for norm_match in (False, True):
            result, _ = combine_task_gradients(gd, gp, [.5, .5],
                task_balance='raw_sum', conflict_guard=False, norm_match=norm_match)
            self.assertTrue(torch.equal(result, gd + gp))

    def test_raw_sum_adaptive_requested_coefficients_without_guard(self):
        result, log = self.mix([3., 0.], [-4., 3.], (.9, .1),
            task_balance='raw_sum', conflict_guard=False, norm_match=False)
        expected = 2 * torch.tensor(.9) * torch.tensor([3., 0.]) + 2 * torch.tensor(.1) * torch.tensor([-4., 3.])
        self.assertTrue(torch.equal(result, expected))
        self.assertAlmostEqual(log['alpha_raw'], 2.7 / 3.2, places=6)
        self.assertFalse(log['first_order_ge_zero_within_roundoff']['depth'])

    def test_raw_sum_guard_projects_effective_unit_proportion_preserving_scale(self):
        result, log = self.mix([3., 0.], [-4., 3.], task_balance='raw_sum', norm_match=False)
        self.assertAlmostEqual(log['alpha_requested'], 3 / 8, places=6)
        self.assertAlmostEqual(log['alpha_effective'], .8 / 1.8, places=6)
        self.assertEqual(log['raw_coefficient_total_scale'], 8.)
        ud, up = torch.tensor([1., 0.]), torch.tensor([-.8, .6])
        expected = 8 * (log['alpha_effective'] * ud + (1-log['alpha_effective']) * up)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))
        self.assertTrue(all(log['first_order_ge_zero_within_roundoff'].values()))

    def test_norm_match_is_independent_of_raw_sum_guard(self):
        off, _ = self.mix([3., 0.], [-4., 3.], task_balance='raw_sum', norm_match=False)
        on, log = self.mix([3., 0.], [-4., 3.], task_balance='raw_sum', norm_match=True)
        self.assertAlmostEqual(log['used_gradient_norm'], math.sqrt(10.), places=6)
        self.assertFalse(torch.equal(off, on))
        self.assertTrue(torch.allclose(off / torch.linalg.vector_norm(off), on / torch.linalg.vector_norm(on)))

    def test_bfloat16_input_priorities_detach_to_fp32_without_mutation_or_vjp(self):
        gd = torch.tensor([[3., 0.]], dtype=torch.bfloat16, requires_grad=True)
        gp = torch.tensor([[0., 4.]], dtype=torch.bfloat16, requires_grad=True)
        p = torch.tensor([.5, .5], requires_grad=True)
        before = (gd.detach().clone(), gp.detach().clone(), p.detach().clone())
        with patch('torch.autograd.grad', side_effect=AssertionError('no third VJP permitted')):
            result, log = combine_task_gradients(gd, gp, p)
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.shape, gd.shape)
        self.assertTrue(all(torch.equal(x.detach(), y) for x, y in zip((gd, gp, p), before)))
        self.assertEqual(log['additional_vjp_calls'], 0)
        json.dumps(log, allow_nan=False)

    def test_requested_priorities_fixed_but_alpha_can_change_each_call(self):
        p = [.9, .1]
        _, aligned = self.mix([1., 0.], [1., 0.], p)
        _, conflict = self.mix([1., 0.], [-.8, .6], p)
        self.assertEqual(aligned['task_priorities_requested'], conflict['task_priorities_requested'])
        self.assertEqual(p, [.9, .1])
        self.assertNotEqual(aligned['alpha_effective'], conflict['alpha_effective'])

    def test_invalid_nonfinite_shape_probability_or_switch_rejected(self):
        gd, gp = torch.ones(2), torch.zeros(2)
        for p in ([.5, .6], [-.1, 1.1], [float('nan'), .5], [True, False], [.5], {'det':.5,'depth':.5}):
            with self.assertRaises((ValueError, TypeError)): combine_task_gradients(gd, gp, p)
        for kwargs in ({'task_balance':'invented'}, {'conflict_guard':'off'}, {'norm_match':1}, {'n_min':0}, {'n_min':float('nan')}):
            with self.assertRaises((ValueError, TypeError)): combine_task_gradients(gd, gp, [.5,.5], **kwargs)
        with self.assertRaises(ValueError): combine_task_gradients(gd, torch.ones(3), [.5,.5])
        with self.assertRaises(FloatingPointError): combine_task_gradients(torch.tensor([float('inf'),0.]), gp, [.5,.5])


if __name__ == '__main__':
    unittest.main()
