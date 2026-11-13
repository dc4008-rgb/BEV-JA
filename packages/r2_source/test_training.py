"""CPU integration checks for the actual fixed100 loop; no pretrained models/data."""
import contextlib
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("meeting_training_test_target", ROOT / "training.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from evidence import CallLedger, StateRecorder, fingerprint
from adaptive_loss import foa_weights


@dataclass
class Step:
    global_step: int
    outer_round: int
    inner_step: int
    sample_id: str
    dataset: str
    noise_seed: int
    render_seed: int


class Scheduler:
    def __init__(self):
        self.config = {"test": "CPU synthetic Euler"}
        self.index = 0

    def set_timesteps(self, n, device):
        self.timesteps = torch.arange(n, 0, -1, device=device).float()
        self.sigmas = torch.linspace(1., .01, n, device=device)
        self.index = 0

    def step(self, prediction, timestep, latent, return_dict=False):
        self.index += 1
        return (latent - .001 * prediction,)


class Transformer:
    def __call__(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict=False):
        return ((.01 * hidden_states + .001 * encoder_hidden_states.reshape(-1, 1, 1, 1)).to(hidden_states.dtype),)


class Config:
    def __init__(self):
        self.__dict__.update(runner.CONFIG)
        self.adversarial_updates = 100

    def to_dict(self):
        return dict(self.__dict__)


class View:
    def __init__(self, patch):
        self.patch = patch

    def render(self, context):
        support = torch.zeros(1, 1, 16, 16)
        support[..., 7:9, 7:9] = 1
        return SimpleNamespace(support=support, image=context.background)


class Parent:
    RoadRGBView = View

    @staticmethod
    def build_execution_plan(config, schedule):
        return [Step(i+1, i//25+1, i%25+1, r.sample_id, r.dataset,
                     runner.SEED+1000*(i//25), runner.SEED+100000+i+1) for i,r in enumerate(schedule)]

    @staticmethod
    def encode_patch(pipeline, patch, dtype):
        return patch.detach().mean().to(dtype).expand(1, 16, 2, 2).clone()

    @staticmethod
    def decode_patch(pipeline, latent):
        return latent[:, :3].float().repeat_interleave(8, -1).repeat_interleave(8, -2)

    @staticmethod
    def save_patch(path, patch, metadata, **kwargs):
        torch.save({"rgb": patch.detach(), "metadata": metadata}, path)

    @staticmethod
    def save_rgb(patch, path):
        path.write_bytes(b"synthetic-test-image")

    @staticmethod
    def save_training_checkpoint(output, patch, step):
        name = f"checkpoint_{step:04d}.pt"
        Parent.save_patch(output/name, patch, {"updates": step})
        return name, runner.sha(output/name)


def fake_losses(parent, patch, context, views, adapters, ledger):
    # Two nonnegative objectives, finite gradients; same four-view evaluation budget.
    # Constants differ so adaptive weights must genuinely use per-step histories.
    raw = [(.2 + 1e-5 * (patch - .4).square().mean()),
           (.3 + 2e-5 * (patch + .2).square().mean())]
    for i in range(8):
        ledger.call("detector_forwards", lambda: None)
    return raw, {"synthetic": True, "view_count": len(views)}


class FullLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ["PYTHONHASHSEED"] = str(runner.SEED)
        runner.configure_determinism()

    def run_once(self, output):
        random.seed(runner.SEED)
        np.random.seed(runner.SEED)
        torch.manual_seed(runner.SEED)
        output.mkdir()
        config = Config()
        schedule = [SimpleNamespace(sample_id=f"train{i:03d}", split="train", dataset="synthetic") for i in range(100)]
        context = SimpleNamespace(background=torch.zeros(1, 3, 16, 16),
                    object_box=[7, 7, 8, 8], road_quad=[[7,7],[9,7],[9,9],[7,9]])
        ledger = CallLedger(runner.BUDGET)
        recorder = StateRecorder(output, runner.CANDIDATE_ID)
        pipeline = SimpleNamespace(scheduler=Scheduler(), transformer=Transformer())
        with contextlib.redirect_stdout(io.StringIO()):
            final, trace, checkpoints = runner.run_optimization(parent=Parent, config=config, output=output,
                pipeline=pipeline, prompt_pack=(torch.ones(1),torch.zeros(1),torch.ones(1),torch.zeros(1)),
                initial_patch=torch.ones(1,3,16,16)*.1, schedule=schedule, context_for=lambda r: context,
                adapters={}, ledger=ledger, recorder=recorder, device=torch.device("cpu"),
                dtype=torch.bfloat16, eot_config=json.loads((ROOT/"eot_config.json").read_text()),
                loss_evaluator=fake_losses)
        recorder.finish("complete", 100)
        return final, trace, checkpoints, recorder, ledger

    def test_complete_loop_history_update_and_bitwise_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.run_once(root/"one")
            second = self.run_once(root/"two")
            self.assertEqual(first[4].completed, runner.BUDGET)
            self.assertEqual(len(first[3].entries), 409)
            self.assertEqual(len(first[1]), 100)
            self.assertEqual(set(first[2]), {f"checkpoint_{s:04d}.pt" for s in (25,50,75,100)})
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertEqual([e["scientific_fingerprint"] for e in first[3].entries],
                             [e["scientific_fingerprint"] for e in second[3].entries])
            previous = None
            for entry in first[3].entries:
                if entry["phase"] != "pre_update":
                    continue
                state = torch.load(root/"one"/entry["path"], weights_only=True)["state"]
                # Check exact real-trajectory FOA recurrence, including boundaries26/51/76.
                self.assertEqual(fingerprint(state["previous_raw_losses"]), fingerprint(previous))
                expected_weights, _ = foa_weights(state["raw_losses"], previous)
                self.assertTrue(torch.equal(state["adaptive_weights"], expected_weights))
                previous = state["raw_losses"]
                z, g = state["latent_variable"], state["weighted_loss_gradient"]
                self.assertEqual(z.dtype, torch.bfloat16)
                expected = z - (.05 * (100000.0 * g.to(z)))
                self.assertTrue(torch.equal(expected, state["candidate_latent"]))
                self.assertEqual(state["update_statistics"]["normalization"], "none")
                self.assertEqual(state["update_statistics"]["clamp"], "none")
                self.assertFalse(state["update_statistics"]["independent_one_step"])
            self.assertEqual(first[1][0]["adaptive_weights"], [1.,1.])

    def test_denoise_matches_original_cfg_arithmetic(self):
        pipe = SimpleNamespace(scheduler=Scheduler(), transformer=Transformer())
        pipe.scheduler.set_timesteps(100, "cpu")
        z = torch.arange(64, dtype=torch.float32).reshape(1,16,2,2).to(torch.bfloat16)/20
        prompt = (torch.tensor([2.]),torch.tensor([-.5]),torch.ones(1),torch.zeros(1))
        actual, evidence = runner.denoise_with_evidence(pipe,z,torch.tensor(25.),prompt,4.5)
        raw = (.01*torch.cat([z,z]) + .001*torch.tensor([-.5,2.]).reshape(-1,1,1,1)).to(z.dtype)
        uncond,cond=raw.chunk(2)
        expected = z - .001*(uncond+4.5*(cond-uncond))
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(evidence["transformer_raw_prediction"],raw))
        self.assertEqual(pipe.scheduler.index,1)

    def test_failed_forward_retains_current_state(self):
        config = Config()
        schedule = [SimpleNamespace(sample_id="train", split="train", dataset="synthetic")] * 100
        context = SimpleNamespace(background=torch.zeros(1,3,16,16), object_box=[7,7,8,8],
                                  road_quad=[[7,7],[9,7],[9,9],[7,9]])
        def failing(*args):
            raise FloatingPointError("injected nonfinite loss")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            rec=StateRecorder(output,runner.CANDIDATE_ID)
            pipe=SimpleNamespace(scheduler=Scheduler(),transformer=Transformer())
            ledger=CallLedger(runner.BUDGET)
            with self.assertRaisesRegex(FloatingPointError,"injected"):
                runner.run_optimization(parent=Parent,config=config,output=output,pipeline=pipe,
                    prompt_pack=(torch.ones(1),)*4,initial_patch=torch.ones(1,3,16,16)*.1,
                    schedule=schedule, context_for=lambda r: context,adapters={},ledger=ledger,
                    recorder=rec,device=torch.device("cpu"),dtype=torch.bfloat16,
                    eot_config=json.loads((ROOT/"eot_config.json").read_text()),loss_evaluator=failing)
            self.assertEqual(rec.entries[-1]["phase"], "failed_forward")
            state=torch.load(output/rec.entries[-1]["path"],weights_only=True)["state"]
            self.assertEqual(set(("latent_variable","raw_patch","denoise","eot","rng")) - set(state),set())
            self.assertEqual(rec.updates_applied,0)
            self.assertEqual(ledger.attempted["autograd_calls"],0)

    def test_geometry_audit_fields_do_not_change_sampling(self):
        spec=importlib.util.spec_from_file_location("old_geometry_test",ROOT/"reference_geometry_eot.py")
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        from geometry_eot import sample_views
        config=json.loads((ROOT/"eot_config.json").read_text())
        for step in (1,25,26,50,51,75,76,100):
            args=(320,1024,[400,150,500,230],[410,240,490,280],runner.SEED+step*1000,config)
            original=old.sample_views(*args)
            new=sample_views(*args)
            for view in new:
                view.pop("sampling_rng_before")
                view.pop("sampling_rng_after")
            self.assertEqual(new,original)

    def test_non_train_is_rejected_before_denoise(self):
        config=Config()
        schedule=[SimpleNamespace(sample_id="invalid",split="validation",dataset="synthetic")]*100
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            rec=StateRecorder(output,runner.CANDIDATE_ID)
            pipe=SimpleNamespace(scheduler=Scheduler(),transformer=Transformer())
            ledger=CallLedger(runner.BUDGET)
            with self.assertRaisesRegex(RuntimeError,"non-train"):
                runner.run_optimization(parent=Parent,config=config,output=output,pipeline=pipe,
                    prompt_pack=(torch.ones(1),)*4,initial_patch=torch.zeros(1,3,16,16),schedule=schedule,
                    context_for=lambda r: None,adapters={},ledger=ledger,recorder=rec,device=torch.device("cpu"),
                    dtype=torch.bfloat16,eot_config=json.loads((ROOT/"eot_config.json").read_text()),loss_evaluator=fake_losses)
            self.assertEqual(ledger.attempted["transformer_forward_calls"],0)
            self.assertEqual(ledger.attempted["autograd_calls"],0)


if __name__ == "__main__":
    unittest.main()
