"""Synthetic CPU evidence/loop checks; no source image or weight loading."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import inputs
from training.run_training import evaluate_model_losses, parse_args, run_optimization, runtime_gate, safe_output, write_json
from training.state_evidence import (CallLedger, CountingAdapter, StateRecorder, canonical_json,
                                    fingerprint, gradient_stats, rng_snapshot, safe_snapshot, scheduler_snapshot)


class SyntheticScheduler:
    def __init__(self):
        self.config = {"shift": 1.0}
        self._step_index = None
        self.history = []

    def set_timesteps(self, count, device):
        self.timesteps = torch.arange(count, 0, -1, device=device)
        self.sigmas = torch.linspace(.5, 0, count + 1, device=device)
        self._step_index = 75


class SyntheticAdapter:
    def __init__(self, factor):
        self.factor = factor

    def predict(self, image):
        return (image.mean() + 4 * image.flatten()[0]) * self.factor


class SyntheticView:
    def __init__(self, rgb):
        self.rgb = rgb

    def render(self, context, **kwargs):
        return SimpleNamespace(image=self.rgb, support=torch.ones_like(self.rgb[:, :1]))


class TrainingTests(unittest.TestCase):
    def test_state_preserves_all_rng_and_original_dtype(self):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        before = rng_snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            writer = StateRecorder(tmp, "synthetic", "U1")
            tensor = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 2, 2)
            binding = writer.save("gradients", 1, {"latent": tensor, "rng": before})
            self.assertEqual(fingerprint(before), fingerprint(rng_snapshot()))
            saved = torch.load(Path(tmp) / binding["path"], weights_only=True)
            self.assertEqual(saved["state"]["latent"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(saved["state"]["latent"], tensor))
            self.assertEqual(saved["metadata"]["update_variant"], "U1")
            self.assertEqual(saved["metadata"]["latent_lr"], inputs.LEARNING_RATE)
            self.assertEqual(saved["metadata"]["learning_rate_label"], inputs.VARIANT_LABEL)
            self.assertEqual(saved["metadata"]["identity"], inputs.IDENTITY)
            manifest_sha = writer.finish("failed_retained", 0)
            manifest = json.loads((Path(tmp) / "state_evidence/MANIFEST.json").read_text())
            self.assertEqual(manifest_sha, inputs.sha(Path(tmp) / "state_evidence/MANIFEST.json"))
            entry = manifest["entries"][0].copy()
            digest = entry.pop("entry_sha256")
            self.assertEqual(digest, hashlib.sha256(canonical_json(entry)).hexdigest())
            with self.assertRaises(FileExistsError):
                writer.finish("complete", 100)

    def test_complete_scheduler_and_unknown_field_rejection(self):
        scheduler = SyntheticScheduler()
        scheduler.set_timesteps(100, "cpu")
        evidence = scheduler_snapshot(scheduler)
        self.assertEqual(evidence["runtime_field_names"], sorted(vars(scheduler)))
        self.assertEqual(evidence["runtime_fields"]["_step_index"], 75)
        scheduler.unsupported = object()
        with self.assertRaises(TypeError):
            scheduler_snapshot(scheduler)

    def test_installed_real_scheduler_when_available(self):
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
        except ImportError:
            self.skipTest("diffusers is absent locally; no runtime install or model load")
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(100, device="cpu")
        scheduler.set_begin_index(75)
        scheduler.step(torch.ones(1, 4, 2, 2), scheduler.timesteps[75],
                       torch.zeros(1, 4, 2, 2), return_dict=False)
        result = scheduler_snapshot(scheduler)
        self.assertEqual(result["runtime_field_names"], sorted(vars(scheduler)))
        self.assertTrue(result["complete_instance_dict_saved"])
        self.assertEqual(result["runtime_fields"]["_step_index"], 76)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = StateRecorder(tmp, "synthetic-scheduler", "U1")
            binding = recorder.save("before_denoise", 1, {"scheduler": result})
            loaded = torch.load(Path(tmp) / binding["path"], weights_only=True)
            self.assertEqual(fingerprint(loaded["state"]["scheduler"]), fingerprint(result))

    def test_true_zero_underflow_and_nonfinite_distinguished(self):
        zero = gradient_stats(torch.zeros(4))
        tiny = gradient_stats(torch.tensor([1e-30]))
        self.assertEqual(zero["cpu_recomputed_classification"], "exact_zero")
        self.assertEqual(tiny["nonzero_count"], 1)
        self.assertGreater(tiny["l2_float64"], 0)
        self.assertEqual(gradient_stats(None)["classification"], "disconnected")
        self.assertFalse(gradient_stats(torch.tensor([float("nan")]))["all_finite"])

    def test_budget_failure_retains_attempt_and_refuses_retry(self):
        ledger = CallLedger({"query": 1})
        def fail():
            raise RuntimeError("deliberate")
        with self.assertRaises(RuntimeError):
            ledger.call("query", fail)
        self.assertEqual(ledger.attempted["query"], 1)
        self.assertEqual(ledger.completed["query"], 0)
        with self.assertRaises(RuntimeError):
            ledger.call("query", lambda: None)
        with self.assertRaises(ValueError):
            CallLedger({"query": True})

    def test_cli_has_no_smoke_or_free_budget(self):
        args = parse_args(["--variant", "U1"])
        self.assertEqual(args.mode, "preflight")
        with contextlib.redirect_stderr(io.StringIO()):
            for tail in (["--variant", "U0"], ["--mode", "smoke"], ["--latent-lr", "0.5"],
                         ["--mode", "train", "--allow-unsealed-local-preflight"]):
                with self.assertRaises(SystemExit):
                    parse_args(["--variant", "U1", *tail])

    def test_runtime_rejects_wrong_scheduler_library_before_cuda(self):
        import torchvision
        protocol = dict(inputs.protocol_contract(), required_torch_base_version=torch.__version__.split("+")[0],
                        required_torchvision_base_version=torchvision.__version__.split("+")[0],
                        required_diffusers_base_version="0.38.0")
        args = SimpleNamespace(expected_hostname=socket.gethostname(), package_manifest_sha256="bound")
        fake_diffusers = SimpleNamespace(__version__="0.37.0", FlowMatchEulerDiscreteScheduler=SyntheticScheduler)
        with patch.dict(sys.modules, diffusers=fake_diffusers, transformers=SimpleNamespace(__version__="synthetic")), \
                patch("training.run_training.inputs.sha", return_value="bound"), \
                patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be queried")):
            with self.assertRaisesRegex(RuntimeError, "unsupported diffusers"):
                runtime_gate(args, protocol)

    def test_safe_output_rejects_external_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                safe_output(Path(tmp) / "outside")
            with patch("training.run_training.inputs.candidate_run_path", side_effect=lambda p: p):
                with self.assertRaises(FileExistsError):
                    safe_output(Path(tmp).resolve())
                link = Path(tmp).resolve() / "link"
                link.symlink_to(Path(tmp), target_is_directory=True)
                with self.assertRaises(ValueError):
                    safe_output(link / "output")

    def test_wrong_lr_or_variant_fails_before_loop_io_and_model_calls(self):
        for variant, rate in [("U0", inputs.LEARNING_RATE), ("U1", .3), ("U1", True)]:
            with self.subTest(variant=variant, rate=rate), self.assertRaises(ValueError):
                run_optimization(parent=None, config=SimpleNamespace(latent_lr=rate),
                    protocol=inputs.protocol_contract(), objective=variant, output=None, pipeline=None,
                    prompt_pack=None, initial_patch=None, schedule=None, context_for=None,
                    run_config=None, method_contract=None, adapters=None, normalizers=None,
                    calibration_sha=None, ledger=None, recorder=None, loss_function=None,
                    device=None, dtype=None)

    def test_state_refuses_inconsistent_lr_before_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = StateRecorder(directory, "synthetic", "U1")
            for state in ({"latent_lr": .3}, {"config": {"latent_lr": .3}},
                          {"update_statistics": {"latent_lr": .3}}, {"learning_rate_label": "wrong"}):
                with self.subTest(state=state), self.assertRaises(ValueError):
                    recorder.save("pre_update", 1, state)
            self.assertEqual(recorder.entries, [])

    def synthetic_loop(self, objective, output, *, recorder_type=StateRecorder, fail_phase=None):
        parent = inputs.parent_runner()
        protocol = inputs.protocol_contract()
        config = parent.StatefulOursConfig(**protocol["stateful_sd35"])
        ledger = CallLedger(protocol["budget_per_run"])
        fake = SimpleNamespace(**{name: getattr(parent, name) for name in (
            "build_execution_plan", "combine_detector_gradients", "detector_progress", "tensor_list",
            "audit_candidate_trace", "texture_diagnostics", "save_patch", "save_training_checkpoint", "make_generator")})
        fake.canonicalize_rgb = lambda x: x
        fake.RoadRGBView = SyntheticView
        fake.patch_brightness = lambda x, g, bounds: (x * (.9 + .2 * torch.rand((), generator=g)),
                                                     float(.9 + .2 * torch.rand((), generator=g)))
        fake.encode_patch = lambda pipeline, image, dtype: image.to(dtype)
        fake.decode_patch = lambda pipeline, latent: latent.sigmoid()
        fake.save_rgb = lambda image, path: Path(path).write_bytes(b"synthetic-png-placeholder")
        def denoise(pipeline, latents, timestep, pack, guidance):
            pipeline.scheduler._step_index += 1
            return latents * .999
        fake.standard_denoise_step = denoise
        pipeline = SimpleNamespace(scheduler=SyntheticScheduler())
        run_config = SimpleNamespace(train_models=("fasterrcnn", "ssd300"))
        adapters = {name: CountingAdapter(SyntheticAdapter(index + 1), ledger)
                    for index, name in enumerate(run_config.train_models)}
        calls = []
        def loss_function(config, contract, name, adapter, context, image, support, cache, sample_id, *, update_variant, phase):
            calls.append((update_variant, phase, name))
            value = adapter.predict(image)
            if phase == fail_phase:
                raise RuntimeError("deliberate synthetic phase failure after actual forward")
            if name == "ssd300":
                value = value.square()
            return SimpleNamespace(total=value, scalars=lambda: {"value": float(value.detach())}, diagnostics={})
        # Calibration invocation counts exactly eight actual adapter callbacks.
        ledger.phase = "calibration"
        with torch.no_grad():
            for index in range(4):
                for name in run_config.train_models:
                    loss_function(run_config, None, name, adapters[name], None, torch.ones(1, 3, 4, 4),
                                  None, {}, str(index), update_variant=objective, phase="calibration")
        normalizers = {"fasterrcnn": 1., "ssd300": 2.}
        write_json(output / "calibration.json", {"normalizers": normalizers})
        recorder = recorder_type(output, "synthetic-" + objective, objective)
        schedule = [SimpleNamespace(sample_id=f"sample{index}", dataset="kitti" if index % 2 == 0 else "nuscenes", split="train")
                    for index in range(100)]
        with contextlib.redirect_stdout(io.StringIO()):
            final, trace, checkpoints = run_optimization(parent=fake, config=config, protocol=protocol, objective=objective,
                output=output, pipeline=pipeline, prompt_pack=(torch.ones(1),) * 4,
                initial_patch=torch.full((1, 3, 4, 4), .5), schedule=schedule,
                context_for=lambda r: SimpleNamespace(), run_config=run_config, method_contract=None,
                adapters=adapters, normalizers=normalizers, calibration_sha=inputs.sha(output / "calibration.json"),
                ledger=ledger, recorder=recorder, loss_function=loss_function, device=torch.device("cpu"), dtype=torch.float32)
        return final, trace, checkpoints, ledger, recorder, calls

    def test_synthetic_100_updates_full_state_budget_and_arithmetic_fixed_u1_lr(self):
        for objective in ("U1",):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                before = fingerprint(rng_snapshot())
                final, trace, checkpoints, ledger, recorder, calls = self.synthetic_loop(objective, output)
                self.assertEqual(before, fingerprint(rng_snapshot()))
                self.assertEqual(len(trace), 100)
                self.assertEqual(set(checkpoints), {f"checkpoint_{step:04d}.pt" for step in (25, 50, 75, 100)})
                self.assertEqual(ledger.completed, inputs.protocol_contract()["budget_per_run"])
                self.assertEqual(sum(phase == "calibration" for _, phase, _ in calls), 8)
                self.assertEqual(sum(phase == "train_pre" for _, phase, _ in calls), 400)
                self.assertEqual(sum(phase == "train_post" for _, phase, _ in calls), 400)
                self.assertEqual({item[0] for item in calls}, {objective})
                self.assertEqual(len(recorder.entries), 409)
                for row in trace:
                    self.assertEqual(set(row["state_bindings"]), {"before_denoise", "gradients", "pre_update", "post_update"})
                    def load(phase):
                        binding = row["state_bindings"][phase]
                        path = output / binding["path"]
                        self.assertEqual(inputs.sha(path), binding["sha256"])
                        return torch.load(path, weights_only=True)["state"]
                    pre, post = load("pre_update"), load("post_update")
                    self.assertEqual(pre["latent_lr"], inputs.LEARNING_RATE)
                    self.assertEqual(pre["learning_rate_label"], inputs.VARIANT_LABEL)
                    self.assertEqual(row["latent_lr"], inputs.LEARNING_RATE)
                    self.assertEqual(row["learning_rate_label"], inputs.VARIANT_LABEL)
                    self.assertEqual(pre["update_statistics"]["latent_lr"], inputs.LEARNING_RATE)
                    proposed = inputs.LEARNING_RATE * pre["joint_gradient_cast_for_update"]
                    applied = proposed.clamp(-.1767578125, .1767578125) if objective == "U1" else proposed
                    expected = pre["latent_variable"] - applied
                    self.assertTrue(torch.equal(applied, pre["applied_step"]))
                    self.assertTrue(torch.equal(expected, post["latents"]))
                if objective == "U1":
                    self.assertGreater(sum(row["update_statistics"]["clipped_count"] for row in trace), 0)
                recorder.finish("complete", 100)

    def test_serialization_does_not_change_optimization_trajectory(self):
        class NullRecorder:
            def __init__(self, *args):
                pass
            def save(self, phase, step, state):
                return {"path": phase + str(step), "sha256": "not-real-test-only"}
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            captured = self.synthetic_loop("U1", Path(a))[0]
            uncaptured = self.synthetic_loop("U1", Path(b), recorder_type=NullRecorder)[0]
            self.assertTrue(torch.equal(captured, uncaptured))

    def test_post_forward_failure_preserves_applied_update_and_partial_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "deliberate synthetic phase failure"):
                self.synthetic_loop("U1", output, fail_phase="train_post")
            entries = [json.loads(line) for line in (output / "state_evidence/ledger.jsonl").read_text().splitlines()]
            self.assertEqual(entries[-1]["phase"], "partial_post_update")
            self.assertEqual((output / "attack_trace.jsonl").read_text(), "")
            partial = torch.load(output / entries[-1]["path"], weights_only=True)["state"]
            pre = torch.load(output / partial["pre_update_state"]["path"], weights_only=True)["state"]
            self.assertTrue(torch.equal(partial["latents"], pre["latent_variable"] - (inputs.LEARNING_RATE * pre["joint_gradient_cast_for_update"]).clamp(-.1767578125,.1767578125)))


if __name__ == "__main__":
    unittest.main()
