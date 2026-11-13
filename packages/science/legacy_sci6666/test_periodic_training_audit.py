"""Focused CPU audit failure checks, without a fabricated full scientific run."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent / "r2_joint_clip120_followup_20261004"))

import torch

from clip_update import clipped_scaled_update
from periodic_controller import PeriodicController
from sample_contract import CONTROLLER_CONFIG, planned_budget, scientific_config_for
from sample_evaluate import validate_summary
from task_gradient_balance import combine_task_gradients
import periodic_training_audit as audit


SAMPLE = {"sample_id": "sample_a", "case_id": "sample_a", "dataset": "kitti", "source_split": "train", "source_sha256": "1" * 64}


def replay_fixture(updates=25, sample=None):
    """Authentic controller log only; no synthetic model/run/manifest inventory."""
    sample = SAMPLE if sample is None else sample
    config = scientific_config_for("complete", updates)
    controller = audit._controller(config)
    rows, cycles = [], []
    for step in range(1, updates + 1):
        outer, inner = (step - 1) // 25, (step - 1) % 25 + 1
        if inner == 1:
            controller.reset_outer(outer)
        weights = controller.weights()
        det, depth = [0.25, 0.9], [1.4, 0.2]
        cycle = controller.observe(inner, det, depth)
        row = {"global_step": step, "outer_round": outer + 1, "inner_step": inner,
               "sample_id": sample["sample_id"], "case_id": sample["case_id"], "dataset": sample["dataset"], "split": sample["source_split"],
               "switches": config["switches"], "controller_config": config["controller_config"],
               "current_period_weights": weights, "task_loss_dtype": "torch.float32",
               "detection_model_weights": audit.quantized_pair(weights["detector"], "torch.float32"),
               "adaptive_weights": audit.quantized_pair(weights["detector"], "torch.float32"),
               "depth_model_weights": audit.quantized_pair(weights["depth"], "torch.float32"), "task_weights": weights["task"],
               "autograd_calls_this_step": 2, "weighted_total_loss_role": "detached_diagnostic_not_third_backward",
               "noise_seed": config["seed"] + 1000 * outer, "eot_seed": config["seed"] + 1000 * step,
               "detection_deficit_details": {"deficits": det},
               "depth_deficit_details": {"deficits": depth, "detached": True, "extra_clipping": False}, "cycle_summary": cycle}
        rows.append(row)
        if cycle is not None:
            cycles.append(cycle)
    return rows, cycles, config


def math_fixture(*, clipped=False):
    config = scientific_config_for("complete", 25)
    factor = .01 if clipped else .001
    gd, gp = torch.tensor([factor, 0.]), torch.tensor([-.7 * factor, .7 * factor])
    priorities = [.5, .5]
    gradient, details = combine_task_gradients(gd, gp, priorities)
    z = torch.tensor([1., 2.], dtype=torch.bfloat16)
    _, tensors, update = clipped_scaled_update(z, gradient, 100000., .05, max_norm=120.)
    weights = {"detector": [1., 1.], "depth": [.5, .5], "task": priorities}
    det, depth = torch.tensor([.25, .9]), torch.tensor([1.4, .2])
    state = {"detection_task_gradient": gd, "depth_task_gradient": gp, "joint_gradient": gradient,
             "latent_variable": z, "switches": config["switches"], "current_period_weights": weights,
             "requested_task_priorities": priorities, "detection_deficits": det, "depth_deficits": depth,
             "task_allocation": details, **tensors}
    row = {"global_step": 1, "current_period_weights": weights, "task_weights": priorities,
           "detection_deficit_details": {"deficits": det.tolist()}, "depth_deficit_details": {"deficits": depth.tolist()},
           "task_allocation": details, "update_statistics": update}
    return state, row, config


class PeriodicTrainingAuditTests(unittest.TestCase):
    def test_25_and_500_replay_counts_outer_reset_and_last_cycle(self):
        for updates, expected_cycles, expected_refreshes in ((25, 5, 4), (500, 100, 80)):
            rows, cycles, config = replay_fixture(updates)
            result = audit.replay_case_trace(rows, cycles, config, SAMPLE, updates)
            self.assertEqual(result["statistical_cycles"], expected_cycles)
            self.assertEqual(result["weight_refreshes_used"], expected_refreshes)
            self.assertFalse(cycles[-1]["applied_next"])

    def test_cycle_member_current_or_future_weight_tampering_rejected(self):
        for path in ("member_steps", "current", "next"):
            rows, cycles, config = replay_fixture()
            corrupted = copy.deepcopy(rows)
            cycle = corrupted[4]["cycle_summary"]
            if path == "member_steps":
                cycle[path][0] = 2
            else:
                cycle[path]["detector"][0] += .01
            with self.assertRaisesRegex(ValueError, "cycle summary"):
                audit.replay_case_trace(corrupted, cycles, config, SAMPLE, 25)

    def test_refresh_cannot_be_used_on_the_closing_cycle_step(self):
        rows, cycles, config = replay_fixture()
        rows[4]["current_period_weights"] = copy.deepcopy(cycles[0]["next"])
        with self.assertRaisesRegex(ValueError, "period weights"):
            audit.replay_case_trace(rows, cycles, config, SAMPLE, 25)

    def test_weight_change_inside_cycle_and_wrong_float32_cast_rejected(self):
        rows, cycles, config = replay_fixture()
        rows[2]["current_period_weights"]["depth"][0] += .01
        with self.assertRaisesRegex(ValueError, "period weights"):
            audit.replay_case_trace(rows, cycles, config, SAMPLE, 25)
        rows, cycles, config = replay_fixture()
        rows[5]["detection_model_weights"][0] += 1e-7
        with self.assertRaisesRegex(ValueError, "logged model/task"):
            audit.replay_case_trace(rows, cycles, config, SAMPLE, 25)

    def test_outer_and_sample_history_cannot_leak(self):
        rows, cycles, config = replay_fixture(500)
        rows[25]["current_period_weights"] = copy.deepcopy(rows[24]["current_period_weights"])
        with self.assertRaisesRegex(ValueError, "period weights"):
            audit.replay_case_trace(rows, cycles, config, SAMPLE, 500)
        other = dict(SAMPLE, sample_id="sample_b", case_id="sample_b")
        rows, cycles, config = replay_fixture(sample=other)
        with self.assertRaisesRegex(ValueError, "another sample"):
            audit.replay_case_trace(rows, cycles, config, SAMPLE, 25)

    def test_cycle_file_missing_or_extra_cycle_and_depth_clip_rejected(self):
        rows, cycles, config = replay_fixture()
        with self.assertRaisesRegex(ValueError, "cycle_trace"):
            audit.replay_case_trace(rows, cycles[:-1], config, SAMPLE, 25)
        rows[0]["depth_deficit_details"]["extra_clipping"] = True
        with self.assertRaisesRegex(ValueError, "un-clipped"):
            audit.replay_case_trace(rows, cycles, config, SAMPLE, 25)

    def test_native_summary_rejects_changed_actual_vjp_or_detector_budget(self):
        config = scientific_config_for("complete", 25)
        plan = {"_plan_sha256": "2" * 64, "source_bindings": {"source": "3" * 64},
                "scientific_config": config, "budget": {"updates_per_sample": 25}}
        counts = planned_budget(25)
        summary = {**SAMPLE, "identity": audit.IDENTITY, "status": "train_optimization_complete",
                   "plan_sha256": plan["_plan_sha256"], "source_bindings": plan["source_bindings"], "executed_updates": 25,
                   "checkpoint_selection": "fixed_final", "test_data_opened": False,
                   "validation_data_opened": False, "optimization_on_original_validation": False,
                   "scientific_config": config, "controller_config": CONTROLLER_CONFIG,
                   "task_autograd_calls": 50, "combined_scalar_autograd_calls": 0, "all_predeclared_scenes_retained": True,
                   "controller_rule": "periodic_gap_gated_foa_outer_reset", "training_depth_models": config["depth_models"],
                   "budget": {key: dict(counts) for key in ("planned", "attempted", "completed")},
                   "patch_bindings": {"zero_step_patch.pt": {}, "final_patch.pt": {}}, "checkpoint_bindings": {"checkpoint_0025.pt": {}}}
        validate_summary(plan, SAMPLE, summary)
        self.assertEqual(counts["detector_forwards"], 202)
        self.assertEqual(counts["autograd_calls"], 50)
        for field in ("autograd_calls", "detector_forwards"):
            bad = copy.deepcopy(summary)
            bad["budget"]["completed"][field] -= 1
            with self.assertRaisesRegex(ValueError, "call budget"):
                validate_summary(plan, SAMPLE, bad)

    def test_saved_task_gradients_and_bf16_clip_arithmetic_pass_for_both_branches(self):
        for clipped in (False, True):
            state, row, config = math_fixture(clipped=clipped)
            result = audit.audit_math_state(state, row, config)
            self.assertTrue(result["clip_element_arithmetic_exact_using_recorded_fp32_scale"])
            self.assertEqual(row["update_statistics"]["clipped"], clipped)

    def test_foreign_priority_changed_gradient_and_bf16_update_rejected(self):
        for field in ("requested_task_priorities", "joint_gradient", "candidate_latent"):
            state, row, config = math_fixture()
            if field == "requested_task_priorities":
                state[field] = [.8, .2]
            else:
                state[field] = state[field].clone()
                state[field][0] += 1.
            with self.assertRaises(ValueError):
                audit.audit_math_state(state, row, config)

    def test_probe_deficit_and_semantic_guard_branch_metadata_rejected(self):
        state, row, config = math_fixture()
        state["depth_deficits"][0] += .5
        with self.assertRaisesRegex(ValueError, "control deficits"):
            audit.audit_math_state(state, row, config)
        state, row, config = math_fixture()
        state["task_allocation"] = copy.deepcopy(state["task_allocation"])
        state["task_allocation"]["active_tasks"]["depth"] = False
        with self.assertRaisesRegex(ValueError, "scientific branch"):
            audit.audit_math_state(state, row, config)

    def test_norm_tolerance_does_not_accept_zero_branch_or_large_drift(self):
        audit.number_close(1.000001, 1., "norm")
        with self.assertRaises(ValueError):
            audit.number_close(1e-20, 0., "norm")
        with self.assertRaises(ValueError):
            audit.number_close(1.01, 1., "norm")
        with self.assertRaisesRegex(ValueError, "exact-zero"):
            audit.tensor_close(torch.tensor([1e-13]), torch.tensor([0.]), "zero-gradient")

    def test_state_owner_plan_source_metadata_and_file_sha_rejected(self):
        plan = {"_plan_sha256": "2" * 64, "source_bindings": {"source": "3" * 64}}
        metadata = {**SAMPLE, "identity": audit.IDENTITY, "plan_sha256": plan["_plan_sha256"],
                    "source_bindings": plan["source_bindings"], "test_data_opened": False}
        audit.validate_payload_metadata({"metadata": metadata}, plan, SAMPLE)
        for key in ("case_id", "plan_sha256", "source_sha256"):
            bad = copy.deepcopy(metadata)
            bad[key] = "foreign"
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                audit.validate_payload_metadata({"metadata": bad}, plan, SAMPLE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "summary.json"
            path.write_text(json.dumps(metadata))
            binding = {"bytes": path.stat().st_size, "sha256": audit.file_sha(path)}
            audit.checked_file(root, "summary.json", binding)
            path.write_text(json.dumps(dict(metadata, case_id="foreign")))
            with self.assertRaisesRegex(ValueError, "bytes/SHA"):
                audit.checked_file(root, "summary.json", binding)


if __name__ == "__main__":
    unittest.main()
