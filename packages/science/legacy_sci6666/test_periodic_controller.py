"""CPU contract tests for periodic statistics, delayed updates and isolation."""
import json
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from periodic_controller import PeriodicConfig, PeriodicController


class PeriodicControllerTests(unittest.TestCase):
    def controller(self, **kwargs):
        value = PeriodicController(PeriodicConfig(**kwargs))
        value.reset_outer(0)
        return value

    def cycle(self, controller, start, det, depth):
        result = None
        for step in range(start, start + controller.config.period):
            result = controller.observe(step, det, depth)
        return result

    def assertPair(self, actual, expected):
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=13)

    def test_cycle_fixed_and_update_delayed(self):
        controller = self.controller()
        initial = controller.weights()
        for step in range(1, 5):
            self.assertIsNone(controller.observe(step, [2.0, 0.0], [0.0, 1.0]))
            self.assertEqual(controller.weights(), initial)
        cycle = controller.observe(5, [2.0, 0.0], [0.0, 1.0])
        self.assertEqual(cycle['current'], initial)
        self.assertTrue(cycle['applied_next'])
        self.assertEqual(cycle['member_steps'], [1, 2, 3, 4, 5])
        self.assertEqual(controller.weights(), cycle['next'])
        fixed = controller.weights()
        for step in range(6, 10):
            controller.observe(step, [0.0, 2.0], [1.0, 0.0])
            self.assertEqual(controller.weights(), fixed)

    def test_last_cycle_logged_without_refresh(self):
        controller = self.controller()
        cycles = []
        for step in range(1, 26):
            before = controller.weights()
            result = controller.observe(step, [2.0, 0.0], [0.0, 1.0])
            if result is not None:
                cycles.append(result)
        self.assertEqual(len(cycles), 5)
        final = cycles[-1]
        self.assertFalse(final['applied_next'])
        self.assertEqual(controller.weights(), before)
        self.assertEqual(final['current'], before)
        self.assertNotEqual(final['next'], before)
        self.assertEqual(final['cumulative_cycles'], 5)
        self.assertEqual(final['cumulative_refreshes'], 4)

    def test_20_outers_100_cycles_80_refreshes(self):
        controller = self.controller()
        for outer in range(20):
            controller.reset_outer(outer)
            for step in range(1, 26):
                result = controller.observe(step, [1.0, 2.0], [3.0, 4.0])
        self.assertEqual(result['cumulative_cycles'], 100)
        self.assertEqual(result['cumulative_refreshes'], 80)
        self.assertEqual(result['cycle_count_outer'], 5)
        self.assertEqual(result['refresh_count_outer'], 4)

    def test_previous_cycle_ratio_and_clamp(self):
        controller = self.controller(ema=False)
        first = self.cycle(controller, 1, [0.0, 4.0], [1.0, 2.0])
        self.assertEqual(first['R'], {'detector': [1.0, 1.0],
            'depth': [1.0, 1.0], 'task': [1.0, 1.0]})
        second = self.cycle(controller, 6, [8.0, 0.0], [2.0, 1.0])
        self.assertEqual(second['R']['detector'], [2.0, 0.5])
        self.assertPair(second['R']['depth'], [2.001 / 1.001, 1.001 / 2.001])
        self.assertPair(second['A']['depth'], [2.0 * 2.001 / 1.001,
                                               1.001 / 2.001])

    def test_arithmetic_task_mean_has_independent_ratio(self):
        controller = self.controller(ema=False)
        self.cycle(controller, 1, [0.0, 4.0], [2.0, 2.0])
        second = self.cycle(controller, 6, [8.0, 0.0], [1.0, 1.0])
        self.assertEqual(second['D']['task'], [4.0, 1.0])
        self.assertPair(second['R']['task'], [4.001 / 2.001, 1.001 / 2.001])
        self.assertNotEqual(second['R']['task'][0],
                            sum(second['R']['detector']) / 2.0)
        weighted_mean = sum(w * d for w, d in zip(
            second['current']['detector'], second['D']['detector'])) / 2.0
        self.assertNotAlmostEqual(weighted_mean, second['D']['task'][0])

    def test_cycle_D_uses_all_member_steps(self):
        controller = self.controller()
        for step in range(1, 6):
            result = controller.observe(step, [step, 0.0], [0.0, step * 2.0])
        self.assertEqual(result['D']['detector'], [3.0, 0.0])
        self.assertEqual(result['D']['depth'], [0.0, 6.0])
        self.assertEqual(result['D']['task'], [1.5, 3.0])

    def test_targets_weight_sums_and_exact_ema_formula(self):
        controller = self.controller()
        result = self.cycle(controller, 1, [2.0, 0.0], [0.0, 4.0])
        soft_det = 1.0 / (1.0 + math.exp(-2.0))
        soft_depth = 1.0 / (1.0 + math.exp(4.0))
        soft_task = 1.0 / (1.0 + math.exp(1.0))
        expected = {'detector': [0.5 + soft_det, 1.5 - soft_det],
                    'depth': [0.25 + 0.5 * soft_depth,
                              0.75 - 0.5 * soft_depth],
                    'task': [0.1 + 0.8 * soft_task,
                             0.9 - 0.8 * soft_task]}
        for group, total in [('detector', 2.0), ('depth', 1.0), ('task', 1.0)]:
            self.assertPair(result['target'][group], expected[group])
            self.assertPair(result['next'][group], [0.7 * old + 0.3 * target
                for old, target in zip(result['current'][group], expected[group])])
            self.assertAlmostEqual(sum(result['next'][group]), total, places=14)

    def test_off_branches_are_fixed_and_ema_off_applies_target(self):
        both_off = self.controller(model_adaptive=False, task_adaptive=False)
        initial = both_off.weights()
        for step in range(1, 26):
            cycle = both_off.observe(step, [0.0, 3.0], [20.0, 0.0])
            self.assertEqual(both_off.weights(), initial)
            if cycle:
                self.assertEqual(cycle['target'], initial)
                self.assertEqual(cycle['next'], initial)
        model_off = self.controller(model_adaptive=False, ema=False)
        cycle = self.cycle(model_off, 1, [0.0, 3.0], [20.0, 0.0])
        self.assertEqual(cycle['next']['detector'], [1.0, 1.0])
        self.assertEqual(cycle['next']['depth'], [0.5, 0.5])
        self.assertNotEqual(cycle['next']['task'], [0.5, 0.5])
        task_off = self.controller(task_adaptive=False, ema=False)
        cycle = self.cycle(task_off, 1, [0.0, 3.0], [20.0, 0.0])
        self.assertEqual(cycle['next']['task'], [0.5, 0.5])
        self.assertEqual(cycle['next'], cycle['target'])

    def test_reset_clears_history_buffer_and_ema(self):
        controller = self.controller()
        self.cycle(controller, 1, [0.0, 3.0], [20.0, 0.0])
        controller.observe(6, [100.0, 100.0], [100.0, 100.0])
        controller.reset_outer(1)
        self.assertEqual(controller.weights(), {'detector': [1.0, 1.0],
            'depth': [0.5, 0.5], 'task': [0.5, 0.5]})
        cycle = self.cycle(controller, 1, [1.0, 1.0], [1.0, 1.0])
        self.assertEqual(cycle['D']['detector'], [1.0, 1.0])
        self.assertEqual(cycle['R']['task'], [1.0, 1.0])
        self.assertEqual(cycle['cycle_count_outer'], 1)
        other = self.controller()
        self.assertEqual(other.weights(), controller.weights())
        self.assertEqual(cycle['cumulative_cycles'], 2)

    def test_period_one_has_24_applied_refreshes(self):
        controller = self.controller(period=1)
        for step in range(1, 26):
            result = controller.observe(step, [1.0, 0.0], [0.0, 1.0])
            self.assertEqual(result['member_steps'], [step])
            self.assertEqual(result['applied_next'], step < 25)
        self.assertEqual(result['cumulative_cycles'], 25)
        self.assertEqual(result['cumulative_refreshes'], 24)

    def test_depth_deficits_above_three_are_not_clipped(self):
        controller = self.controller()
        cycle = self.cycle(controller, 1, [0.0, 3.0], [10.0, 20.0])
        self.assertEqual(cycle['D']['depth'], [10.0, 20.0])
        self.assertEqual(cycle['D']['task'][1], 15.0)

    def test_invalid_steps_and_deficits_do_not_advance_state(self):
        controller = PeriodicController(PeriodicConfig())
        with self.assertRaises(ValueError):
            controller.observe(1, [1.0, 1.0], [1.0, 1.0])
        controller.reset_outer(0)
        for invalid in (0, 2, 26, True, 1.0):
            with self.assertRaises(ValueError):
                controller.observe(invalid, [1.0, 1.0], [1.0, 1.0])
        for invalid in ([math.nan, 0.0], [math.inf, 0.0], [-1.0, 0.0],
                        [True, 0.0], [1.0], [[1.0], [2.0]]):
            with self.assertRaises(ValueError):
                controller.observe(1, invalid, [0.0, 0.0])
            with self.assertRaises(ValueError):
                controller.observe(1, [0.0, 0.0], invalid)
        self.assertIsNone(controller.observe(1, [0.0, 0.0], [0.0, 0.0]))
        with self.assertRaises(ValueError):
            controller.observe(1, [0.0, 0.0], [0.0, 0.0])

    def test_invalid_configuration_and_frozen_config(self):
        from dataclasses import FrozenInstanceError
        for kwargs in ({'period': 2}, {'period': True}, {'model_mu': -0.1},
                       {'task_mu': 1.1}, {'ema_decay': math.nan},
                       {'model_temperature': 0.0}, {'task_temperature': math.inf},
                       {'ratio_delta': 0.0}, {'ratio_min': 2.1},
                       {'model_adaptive': 1}, {'ema': 'off'}):
            with self.assertRaises(ValueError):
                PeriodicConfig(**kwargs)
        config = PeriodicConfig()
        with self.assertRaises(FrozenInstanceError):
            config.period = 1

    def test_snapshots_cannot_mutate_internal_state(self):
        controller = self.controller()
        weights = controller.weights()
        weights['detector'][0] = 99.0
        self.assertEqual(controller.weights()['detector'], [1.0, 1.0])
        cycle = self.cycle(controller, 1, [0.0, 2.0], [1.0, 1.0])
        saved = controller.weights()
        cycle['next']['detector'][0] = 99.0
        cycle['D']['detector'][0] = 99.0
        self.assertEqual(controller.weights(), saved)
        cycle2 = self.cycle(controller, 6, [0.0, 2.0], [1.0, 1.0])
        self.assertEqual(cycle2['R']['detector'], [1.0, 1.0])
        json.dumps(cycle2, allow_nan=False)

    def test_no_python_rng_consumption(self):
        state = random.getstate()
        controller = self.controller()
        for step in range(1, 26):
            controller.observe(step, [0.0, 2.0], [1.0, 1.0])
        self.assertEqual(random.getstate(), state)

    def test_tensor_inputs_detach_and_no_tensor_rng_consumption(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable; pure-Python contract tests remain')
        state = torch.random.get_rng_state().clone()
        det = torch.tensor([2.0, 0.0], requires_grad=True)
        depth = [torch.tensor(0.0, requires_grad=True),
                 torch.tensor(6.0, requires_grad=True)]
        controller = self.controller()
        cycle = self.cycle(controller, 1, det, depth)
        self.assertEqual(cycle['D']['detector'], [2.0, 0.0])
        self.assertEqual(cycle['D']['depth'], [0.0, 6.0])
        self.assertIsNone(det.grad)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(all(type(x) is float for x in cycle['next']['detector']))


if __name__ == '__main__':
    unittest.main()
