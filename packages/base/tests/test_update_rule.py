"""Frozen U1/lr binding, exact finite-precision arithmetic and saved-state rejection."""
import copy
import json
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import inputs
from training.update_rule import CAP, make_update, validate_saved_update
LR = inputs.LEARNING_RATE


class UpdateTests(unittest.TestCase):
    def tensors(self, dtype):
        z = torch.tensor([.125, 2.75, -1.5, 0., .0625, 1.], dtype=dtype, requires_grad=True)
        g = torch.tensor([.5, -30., 19., 0., -1., 1.], dtype=torch.float32)
        return z, g

    def test_unique_u1_lr_matches_cast_multiply_clip_subtraction_exactly(self):
        for dtype in (torch.bfloat16, torch.float32):
            z, g = self.tensors(dtype)
            proposal = LR * g.to(z)
            applied = proposal.clamp(-CAP, CAP)
            after, e = make_update(z, g, LR, 'U1')
            self.assertTrue(torch.equal(e['proposed_step'], proposal))
            self.assertTrue(torch.equal(e['applied_step'], applied))
            self.assertTrue(torch.equal(after, z.detach() - applied))
            self.assertEqual(e['update_statistics']['latent_lr'], LR)
            self.assertEqual(e['update_statistics']['learning_rate_label'], inputs.VARIANT_LABEL)

    def test_clip_changes_only_tails_and_does_not_renormalize(self):
        for dtype in (torch.bfloat16, torch.float32):
            z, g = self.tensors(dtype)
            _, e = make_update(z, g, LR, 'U1')
            proposal = LR * g.to(z)
            mask = proposal.abs().float() > CAP
            self.assertGreaterEqual(int(mask.sum()), 2)
            self.assertTrue(torch.equal(e['applied_step'][~mask], proposal[~mask]))
            self.assertTrue(torch.equal(e['applied_step'][mask], proposal[mask].sign() * CAP))
            self.assertLess(e['update_statistics']['applied_rms'], e['update_statistics']['proposed_rms'])
            self.assertFalse(e['update_statistics']['renormalized_after_clip'])

    def test_inputs_rng_unchanged_output_detached(self):
        z, g = self.tensors(torch.bfloat16)
        a, b, rng = z.detach().clone(), g.clone(), torch.get_rng_state().clone()
        after, _ = make_update(z, g, LR, 'U1')
        self.assertTrue(torch.equal(z, a))
        self.assertTrue(torch.equal(g, b))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(after.requires_grad)
        self.assertIsNone(z.grad)

    def test_nominal_cap_does_not_bound_bf16_realized_delta(self):
        _, e = make_update(torch.tensor([8.], dtype=torch.bfloat16), torch.tensor([100.]), LR, 'U1')
        self.assertEqual(float(e['applied_step'].abs().max()), CAP)
        self.assertGreater(e['update_statistics']['realized_linf'], CAP)
        self.assertEqual(e['update_statistics']['realized_over_nominal_cap_count'], 1)

    def test_boundary_and_small_values_preserved(self):
        g = torch.tensor([CAP / LR, -CAP / LR, 0., .1])
        z = torch.zeros_like(g)
        after, e = make_update(z, g, LR, 'U1')
        self.assertTrue(torch.equal(after, z - LR * g))
        self.assertEqual(e['update_statistics']['clipped_count'], 0)

    def test_nonfinite_wrong_shape_dtype_or_variant_rejected(self):
        z, g = self.tensors(torch.bfloat16)
        cases = [(z, g, LR, 'U0'), (z, g, LR, 'C0'), (z, g[:1], LR, 'U1'),
                 (z.double(), g, LR, 'U1'), (z, g.double(), LR, 'U1'),
                 (z, torch.full_like(g, float('nan')), LR, 'U1'),
                 (torch.full_like(z, float('inf')), g, LR, 'U1')]
        for args in cases:
            with self.subTest(args=str(args[-2:])), self.assertRaises(ValueError):
                make_update(*args)

    def test_every_other_lr_and_bool_rejected(self):
        z, g = self.tensors(torch.bfloat16)
        for wrong in (True, 1, .0, .3, float('nan'), *[v for v in (.05, .1, .2) if v != LR]):
            with self.subTest(lr=wrong), self.assertRaises(ValueError):
                make_update(z, g, wrong, 'U1')

    def test_saved_tensor_or_lr_label_tampering_rejected(self):
        z, g = self.tensors(torch.bfloat16)
        after, e = make_update(z, g, LR, 'U1')
        pre = dict(latent_variable=z.detach(), joint_gradient=g, latent_lr=LR, **e)
        post = dict(latents=after, update_statistics=e['update_statistics'])
        self.assertTrue(validate_saved_update(pre, post, 'U1'))
        for key in ('proposed_step', 'applied_step', 'realized_delta_float32'):
            changed = copy.deepcopy(pre)
            changed[key][0] += 1
            with self.assertRaises(ValueError):
                validate_saved_update(changed, post, 'U1')
        for key, value in [('latent_lr', .3), ('learning_rate_label', 'wrong')]:
            a, b = copy.deepcopy(pre), copy.deepcopy(post)
            a['update_statistics'][key] = b['update_statistics'][key] = value
            with self.assertRaises(ValueError):
                validate_saved_update(a, b, 'U1')
        with self.assertRaises(ValueError):
            validate_saved_update(dict(pre, latent_lr=.3), post, 'U1')
        with self.assertRaises(ValueError):
            validate_saved_update(pre, dict(post, latents=after + 1), 'U1')

    def test_frozen_cap_source_unchanged_for_every_lr(self):
        result = json.loads((inputs.ROOT / 'reference_analysis/RESULT.json').read_text())
        plan = json.loads((inputs.ROOT / 'reference_analysis/ANALYSIS_PLAN.json').read_text())
        self.assertEqual(CAP, result['threshold']['bf16_floor_cap'])
        self.assertEqual(CAP, inputs.load_protocol()['clip_threshold'])
        self.assertEqual(plan['quantile'], .99)
        self.assertFalse(plan['validation_used_for_selection'])
        self.assertEqual(result['threshold']['source'], 'C3 train only')


if __name__ == '__main__':
    unittest.main()
