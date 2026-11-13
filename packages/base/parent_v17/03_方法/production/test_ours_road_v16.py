"""CPU regression checks only; no SD3.5 or real detector weights are loaded."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_公共数据集"))
sys.path.insert(0, str(ROOT / "03_方法/production"))
from carriers import SceneContext
from ours_road_v16 import (model_order, combine_detector_gradients, detector_progress,
                           audit_candidate_trace)
from ours_stateful import StatefulOursConfig
from road_patch_contract import RoadRGBView, save_patch, load_patch


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

runner = load_module("ours_v16_runner_test", ROOT / "05_运行/run_ours_stateful_validation.py")
evaluator = load_module("ours_v16_eval_test", ROOT / "05_运行/evaluate_ours_stateful_validation.py")


class OursRoadV16Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_detector_gradient_normalization_does_not_favor_loss_scale(self):
        first = combine_detector_gradients([torch.tensor([10., 0.]), torch.tensor([0., 1.])])
        rescaled = combine_detector_gradients([torch.tensor([.1, 0.]), torch.tensor([0., 100.])])
        self.assertTrue(torch.allclose(first.joint_gradient, torch.ones(2)))
        self.assertTrue(torch.equal(first.joint_gradient, rescaled.joint_gradient))
        self.assertEqual(first.model_weights.tolist(), [.5, .5])

    def test_one_zero_detector_is_recorded_and_reweighted(self):
        control = combine_detector_gradients([torch.zeros(2), torch.tensor([0., 2.])])
        self.assertEqual(control.zero_model_gradient_indices, (0,))
        self.assertEqual(control.model_weights.tolist(), [0., 1.])
        self.assertTrue(torch.isfinite(control.joint_gradient).all())

    def test_all_zero_conflicting_or_nonfinite_gradients_fail(self):
        for gradients in ([torch.zeros(2), torch.zeros(2)],
                          [torch.tensor([1., 0.]), torch.tensor([-1., 0.])],
                          [torch.tensor([float("nan"), 0.]), torch.ones(2)]):
            with self.assertRaises(ValueError):
                combine_detector_gradients(gradients)
        with self.assertRaises(ValueError):
            combine_detector_gradients([torch.ones(2)] * 4)

    def test_progress_has_exactly_two_real_entries(self):
        progress = detector_progress([torch.tensor(2.), torch.tensor(4.)],
                                     [torch.tensor(1.), torch.tensor(5.)])
        self.assertEqual(progress.tolist(), [.5, 0.])
        with self.assertRaises(ValueError):
            detector_progress([torch.tensor(1.)] * 4, [torch.tensor(1.)] * 4)

    def test_detector_only_never_calls_depth_builder(self):
        with patch.object(runner, "build_offline_detector", side_effect=lambda *args: args[0]) as detector, \
             patch.object(runner, "build_adapter", side_effect=AssertionError("depth builder called")):
            adapters = runner.build_selected_adapters(model_order("det-only"), torch.device("cpu"), Path("manifest"))
        self.assertEqual(tuple(adapters), ("fasterrcnn", "ssd300"))
        self.assertEqual(detector.call_count, 2)

    def test_dual_still_loads_exact_two_depth_models(self):
        with patch.object(runner, "build_offline_detector", side_effect=lambda *args: args[0]) as detector, \
             patch.object(runner, "build_adapter", side_effect=lambda *args: args[0]) as depth:
            adapters = runner.build_selected_adapters(model_order("dual"), torch.device("cpu"), Path("manifest"))
        self.assertEqual(tuple(adapters), model_order("dual"))
        self.assertEqual([x.args[0] for x in depth.call_args_list], ["monodepth2", "dpt_large"])
        self.assertEqual(detector.call_count, 2)

    def test_v16_contracts_and_fixed_denoising_budget(self):
        for variant in ("det-only", "dual"):
            payload, config = runner.load_candidate_contract(ROOT / "00_冻结协议/ours_road_v16_candidate" / (variant + ".json"), variant)
            self.assertEqual(config.active_steps_per_round, 25)
            self.assertEqual(config.adversarial_updates, 100)
            self.assertEqual(config.noise_samples, 4)
            self.assertFalse(payload["formal_test_allowed"])
            self.assertEqual(tuple(payload["training_models"]), model_order(variant))
        with self.assertRaises(ValueError):
            runner.load_candidate_contract(ROOT / "00_冻结协议/ours_road_v16_candidate/dual.json", "det-only")

    def test_inherited_contract_is_not_valid_v16(self):
        path = ROOT / "00_冻结协议/ours_road_v16_candidate/det-only.json"
        payload = json.loads(path.read_text()); payload["candidate_id"] = "normalized-v15"
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "protocol.json"; bad.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                runner.load_candidate_contract(bad)

    def test_cli_defaults_cannot_start_gpu_implicitly(self):
        with patch.object(sys, "argv", ["runner"]):
            args = runner.parse_args()
        self.assertEqual(args.mode, "preflight")
        self.assertEqual(args.task_variant, "det-only")
        self.assertIsNone(args.sd35_snapshot)
        self.assertEqual(args.smoke_active_steps, 1)

    @staticmethod
    def context():
        background = torch.full((1, 3, 320, 1024), .3)
        benign = background.clone(); benign[..., 160:230, 450:560] = .7
        mask = torch.zeros(1, 1, 320, 1024); mask[..., 160:230, 450:560] = 1
        return SceneContext(background, benign, torch.zeros(1, 3, 20, 20),
                            torch.ones(1, 1, 20, 20), mask, (450, 160, 560, 230),
                            ((.4, .74), (.6, .74), (.8, .99), (.2, .99)))

    def test_eot_is_common_across_models_and_preserves_gradient_and_car(self):
        context = self.context()
        raw = torch.full((1, 3, 512, 512), .5, requires_grad=True)
        calls = []
        def fake_loss(config, contract, model_id, adapter, ctx, image, support, cache, sample_id):
            self.assertTrue(torch.equal(image[..., 160:230, 450:560], ctx.benign[..., 160:230, 450:560]))
            calls.append((model_id, image.detach().clone()))
            total = (image * support).sum() * (1 if model_id == "fasterrcnn" else 2)
            return SimpleNamespace(total=total, scalars=lambda: {"synthetic": float(total.detach())})
        kwargs = dict(patch=raw, context=context, record=SimpleNamespace(sample_id="synthetic_train", split="train"),
                      run_config=SimpleNamespace(train_models=model_order("det-only")), method_contract={},
                      adapters={x: None for x in model_order("det-only")}, normalizers={x: 1 for x in model_order("det-only")},
                      benign_depth_cache={}, device=torch.device("cpu"), global_step=1, seed=20260919)
        with patch.object(runner, "model_loss", side_effect=fake_loss):
            losses, details = runner.evaluate_model_losses(**kwargs)
            repeated, repeated_details = runner.evaluate_model_losses(**kwargs)
        self.assertEqual([x[0] for x in calls[:4]], ["fasterrcnn", "ssd300"] * 2)
        self.assertTrue(torch.equal(calls[0][1], calls[1][1]))
        self.assertTrue(torch.equal(calls[2][1], calls[3][1]))
        factors = details["fasterrcnn"]["eot_brightness_factors"]
        self.assertEqual(factors, details["ssd300"]["eot_brightness_factors"])
        self.assertEqual(details, repeated_details)
        self.assertEqual(len(factors), 2)
        self.assertNotEqual(factors[0], factors[1])
        gradient, = torch.autograd.grad(sum(losses), raw)
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_actual_runner_and_evaluator_share_canonical_saved_pixels(self):
        raw = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.pt"
            save_patch(path, raw, {"role": "synthetic_test"}, raw_generator_rgb=raw)
            restored = evaluator.load_patch(path)
            train = runner.DecodedRoadCarrier(raw).render(self.context())
            evaluated = evaluator.RoadRGBView(restored).render(self.context())
            self.assertTrue(torch.equal(train.image, evaluated.image))
            self.assertEqual(tuple(restored.shape), (1, 3, 64, 128))
            self.assertTrue(torch.equal(torch.load(path, weights_only=True)["raw_generator_rgb"], raw))

    def test_output_guard_rejects_historical_or_root_paths(self):
        from ours_road_v16 import require_candidate_run_path
        for invalid in (ROOT.parent / "normalized-v15-ours-formal-20260918/output", ROOT / "runs"):
            with self.assertRaises(ValueError):
                require_candidate_run_path(invalid, ROOT)
        valid = ROOT / "runs/ours_det-only/smoke"
        self.assertEqual(require_candidate_run_path(valid, ROOT), valid.resolve())

    def test_snapshot_binding_rejects_changed_weight_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"; snapshot.mkdir()
            (snapshot / "model_index.json").write_text("{}")
            for name in ("scheduler", "transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
                (snapshot / name).mkdir()
                (snapshot / name / ("scheduler_config.json" if name == "scheduler" else "config.json")).write_text("{}")
                if name in ("transformer", "vae", "text_encoder", "text_encoder_2"):
                    (snapshot / name / "synthetic.safetensors").write_bytes(b"synthetic fixture; not real weights")
            self.assertEqual(runner.validate_snapshot(snapshot)["status"], "structure_present_file_hashes_unbound")
            entries = [{"path": str(p.relative_to(snapshot)), "sha256": runner.sha256(p), "bytes": p.stat().st_size}
                       for p in sorted(snapshot.rglob("*")) if p.is_file()]
            manifest = Path(directory) / "files.json"
            manifest.write_text(json.dumps({"model_id": "stabilityai/stable-diffusion-3.5-medium", "files": entries}))
            digest = runner.sha256(manifest)
            self.assertEqual(runner.validate_snapshot(snapshot, manifest, digest)["status"], "file_hashes_verified")
            (snapshot / "vae/synthetic.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError): runner.validate_snapshot(snapshot, manifest, digest)
            (snapshot / "extra.json").write_text("{}")
            with self.assertRaises(ValueError): runner.validate_snapshot(snapshot, manifest, digest)

    def test_evaluation_rejects_tampered_final_patch(self):
        from ours_road_v16 import verify_patch_bindings
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for name in ("zero_step_patch.pt", "final_patch.pt"):
                save_patch(run / name, torch.full((1, 3, 64, 128), .5))
            summary = {"patch_bindings": {name: runner.sha256(run / name) for name in ("zero_step_patch.pt", "final_patch.pt")}}
            verify_patch_bindings(run, summary)
            (run / "final_patch.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError): verify_patch_bindings(run, summary)

    def test_trace_rejects_fake_depth_slots_and_nontrain_data(self):
        config = StatefulOursConfig()
        row = dict(split="train", model_count=2, training_models=list(model_order("det-only")),
                   losses_pre={x: {} for x in model_order("det-only")}, losses_post={x: {} for x in model_order("det-only")},
                   gradient_norms=[1, 1], task_variant="det-only", task_projection_applied=False, task_cosine=None)
        rows = [dict(row, global_step=i + 1) for i in range(100)]
        audit_candidate_trace(rows, config, "det-only")
        rows[0] = dict(rows[0], model_count=4)
        with self.assertRaises(ValueError): audit_candidate_trace(rows, config, "det-only")
        rows[0] = dict(row, global_step=1, split="test")
        with self.assertRaises(ValueError): audit_candidate_trace(rows, config, "det-only")


if __name__ == "__main__":
    unittest.main()
