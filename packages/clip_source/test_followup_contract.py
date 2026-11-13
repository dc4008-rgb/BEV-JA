"""Synthetic follow-up provenance tests. No real plan, models, data, or server."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import followup_contract as c
from freeze_followup_plan import write_new


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class FollowupContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.reference = self.root / "original"
        self.old = self.root / "old_off"
        self.new = self.root / "new_off"
        self.original_id = "r2_joint_depth_long500_20261001-two-detector-foa-plus-fixed-depth"
        self.entries = [{"index": n, "phase": "synthetic_phase", "global_step": n % 501,
                         "updates_applied": n % 501, "state_fingerprint": f"{n:064x}",
                         "recording_preserved_rng": True, "stored_state": None}
                        for n in range(1, 2042)]
        self.science = dict(config={"seed": 20260919}, budget={"synthetic": 500},
            training_models=["fasterrcnn", "ssd300"], training_depth_models=["monodepth2", "dpt_large"],
            depth_beta=.1, accepted_r2_source_manifest_sha256="a" * 64,
            accepted_joint_v2_manifest_sha256="b" * 64, training_manifest_sha256="c" * 64,
            model_manifest_sha256="d" * 64)
        write(self.reference / "schedule.json", [{"synthetic": "train"}])
        self.science["schedule_sha256"] = c.sha(self.reference / "schedule.json")
        write(self.reference / "summary.json", self.science)
        write(self.reference / "state_evidence/MANIFEST.json", {"entries": self.entries})
        (self.reference / "attack_trace.jsonl").write_text("synthetic pinned original trace\n")
        self.ref_bindings = {"candidate_id": self.original_id, "files": {
            p: c.sha(self.reference / p) for p in
            ("summary.json", "attack_trace.jsonl", "state_evidence/MANIFEST.json", "schedule.json")}}
        self.bindings = {"prior_package_sha256": "1" * 64,
            "prior_identity": "r2_joint_clip_sweep_20261003",
            "prior_candidate_id": "r2_joint_clip_sweep_20261003-two-detector-foa-plus-fixed-depth",
            "prior_run_role": "fresh_joint500_clip_sweep_training", "baselines_sha256": "2" * 64,
            "reference": {"candidate_id": self.original_id, "summary_sha256": self.ref_bindings["files"]["summary.json"],
                          "state_evidence_manifest_sha256": self.ref_bindings["files"]["state_evidence/MANIFEST.json"]}}
        write(self.root / "old_grid.json", {"synthetic_old_grid": True})
        self.bindings["prior_grid_sha256"] = c.sha(self.root / "old_grid.json")
        for name, key in (("comparison.json", "seen_comparison_sha256"), ("points.json", "seen_points_sha256")):
            write(self.root / name, {"synthetic": name})
            self.bindings[key] = c.sha(self.root / name)
        self.runtime = {name: "synthetic_" + name for name in c.ENVIRONMENT_FIELDS}
        self.runtime.update(gpu_identity="Synthetic GPU, GPU-old, 580.82.09",
                            package_manifest_sha256=self.bindings["prior_package_sha256"],
                            hostname="old", total_updates=500, depth_weight=1.0)
        self._make_off(self.old, self.runtime)
        self.bindings["off_files"] = self._file_hashes(self.old)
        new_runtime = dict(self.runtime, gpu_identity="Synthetic GPU, GPU-new, 580.119.02", hostname="new")
        self._make_off(self.new, new_runtime)
        self.addCleanup(patch.stopall)
        patch.object(c, "followup_bindings", return_value=self.bindings).start()
        patch.object(c, "reference_bindings", return_value=self.ref_bindings).start()

    def _file_hashes(self, root):
        return {name: c.sha(root / name) for name in
                ("summary.json", "attack_trace.jsonl", "schedule.json", "state_evidence/MANIFEST.json",
                 "CLIPPING_COMPARISON_GATE.json", "runtime_provenance.json", "CLIP_PLAN.json")}

    def _make_off(self, root, runtime):
        root.mkdir()
        (root / "CLIP_PLAN.json").write_bytes((self.root / "old_grid.json").read_bytes())
        (root / "schedule.json").write_bytes((self.reference / "schedule.json").read_bytes())
        trace = [{"global_step": n, "split": "train", "clipping_diagnostics": {"clipped": False}}
                 for n in range(1, 501)]
        (root / "attack_trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace) + "\n")
        write(root / "runtime_provenance.json", runtime)
        policy = dict(c.POLICY_FIXED, group_id="off", max_norm=None,
                      grid_sha256=self.bindings["prior_grid_sha256"])
        ref = {"candidate_id": self.original_id,
               "summary_sha256": self.ref_bindings["files"]["summary.json"],
               "state_manifest_sha256": self.ref_bindings["files"]["state_evidence/MANIFEST.json"]}
        write(root / "CLIPPING_COMPARISON_GATE.json",
              dict(status="exact_full_trajectory", compared_states=2041, tolerance=0,
                   reference=ref, group_id="off", max_norm=None, grid_sha256=policy["grid_sha256"]))
        state_root = root / "state_evidence"
        state_root.mkdir()
        (state_root / "ledger.jsonl").write_text("\n".join(json.dumps(row) for row in self.entries) + "\n")
        write(state_root / "MANIFEST.json", dict(
            schema="r2-joint-clip500-lean-state-manifest-v1",
            candidate_id=self.bindings["prior_candidate_id"], status="complete", executed_updates=500,
            gradient_clipping=policy, reference_binding=ref, comparison_counts={"states_checked": 2041},
            entries=self.entries, ledger_sha256=c.sha(state_root / "ledger.jsonl")))
        summary = dict(self.science, status="train_optimization_complete",
            identity=self.bindings["prior_identity"], candidate_id=self.bindings["prior_candidate_id"],
            run_role=self.bindings["prior_run_role"], package_manifest_sha256=self.bindings["prior_package_sha256"],
            executed_updates=500, depth_weight=1.0, seed=20260919, latent_lr=.05, gradient_multiplier=100000.,
            update_clamp=None, normalizations=[], test_data_opened=False, validation_data_opened=False,
            fresh_start=True, resume_enabled=False, checkpoint_selection="fixed final500 only",
            gradient_clipping=policy, attack_trace_sha256=c.sha(root / "attack_trace.jsonl"),
            runtime_provenance_sha256=c.sha(root / "runtime_provenance.json"),
            state_evidence_manifest_sha256=c.sha(state_root / "MANIFEST.json"),
            clipping_comparison_gate_sha256=c.sha(root / "CLIPPING_COMPARISON_GATE.json"))
        write(root / "summary.json", summary)

    def plan(self, off=None):
        return c.build_followup_plan(reference_run=self.reference, historical_off_run=self.old,
            off_run=self.new if off is None else off,
            comparison_file=self.root / "comparison.json", points_file=self.root / "points.json")

    def policy(self, plan):
        return dict(c.POLICY_FIXED, group_id="cap120", max_norm=120.,
                    grid_sha256=c.json_fingerprint(plan))

    def test_new_driver_needs_full_reverified_off_then_is_bound(self):
        plan = self.plan()
        self.assertEqual(plan["execution_off"]["origin"], c.OFF_ORIGINS[1])
        self.assertEqual(plan["execution_off"]["scientific_environment"]["values"]["gpu_driver"], "580.119.02")
        binding = c.require_completed_off(self.new, self.policy(plan), "e" * 64,
                                         plan=plan, reference_run=self.reference)
        runtime = c.read_json(self.new / "runtime_provenance.json")
        self.assertEqual(c.require_matching_environment(runtime, binding)["status"], "exact_scientific_environment")
        with self.assertRaisesRegex(ValueError, "gpu_driver"):
            c.require_matching_environment(self.runtime, binding)

    def test_historical_off_explicit_reuse_still_checks_environment(self):
        plan = self.plan(off=self.old)
        self.assertEqual(plan["execution_off"]["origin"], c.OFF_ORIGINS[0])
        binding = c.require_completed_off(self.old, self.policy(plan), "e" * 64,
                                         plan=plan, reference_run=self.reference)
        changed_host = dict(self.runtime, hostname="relocated", gpu_identity="Synthetic GPU, GPU-moved, 580.82.09")
        self.assertEqual(c.require_matching_environment(changed_host, binding)["status"], "exact_scientific_environment")

    def test_false_original_predeclared_label_or_other_cap_refused(self):
        plan = self.plan()
        for changed in (dict(plan, status="thresholds_frozen_from_train"),
                        dict(plan, exploratory_posthoc=False),
                        dict(plan, groups=[{"id": "p1", "aliases": ["p1"], "max_norm": 120., "executed_updates": 500}]),
                        dict(plan, scientific_configuration=dict(plan["scientific_configuration"], depth_weight=True))):
            with self.assertRaises(ValueError):
                c.validate_followup_plan(changed)

    def test_nonmatching_state_cannot_be_blessed_by_exact_gate_claim(self):
        manifest = c.read_json(self.new / "state_evidence/MANIFEST.json")
        manifest["entries"][1000]["state_fingerprint"] = "f" * 64
        write(self.new / "state_evidence/MANIFEST.json", manifest)
        summary = c.read_json(self.new / "summary.json")
        summary["state_evidence_manifest_sha256"] = c.sha(self.new / "state_evidence/MANIFEST.json")
        write(self.new / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "scientific state differs"):
            self.plan()

    def test_four_state_gate_not_sufficient_for_execution_off(self):
        gate = c.read_json(self.new / "CLIPPING_COMPARISON_GATE.json")
        gate.update(compared_states=4, status="exact_initialization_and_first_forward")
        write(self.new / "CLIPPING_COMPARISON_GATE.json", gate)
        summary = c.read_json(self.new / "summary.json")
        summary["clipping_comparison_gate_sha256"] = c.sha(self.new / "CLIPPING_COMPARISON_GATE.json")
        write(self.new / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "exact gate"):
            self.plan()

    def test_historical_sha_tamper_and_seen_result_tamper_fail(self):
        with (self.old / "runtime_provenance.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.plan()

    def test_nontrain_off_trace_refused_even_if_hash_is_updated(self):
        path = self.new / "attack_trace.jsonl"
        path.write_text(path.read_text().replace('"train"', '"validation"', 1))
        summary = c.read_json(self.new / "summary.json")
        summary["attack_trace_sha256"] = c.sha(path)
        write(self.new / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "500 unclipped train"):
            self.plan()

    def test_plan_load_hash_group_and_no_overwrite(self):
        plan = self.plan()
        path = self.root / "followup.json"
        digest = write_new(path, plan)
        loaded, policy = c.load_plan(path, digest, "cap120", self.reference)
        self.assertEqual(loaded, plan)
        self.assertEqual(policy["max_norm"], 120.)
        for group, expected_sha in (("p1", digest), ("cap120", "f" * 64)):
            with self.assertRaises(ValueError):
                c.load_plan(path, expected_sha, group, self.reference)
        with self.assertRaises(FileExistsError):
            write_new(path, plan)

    def test_bound_new_off_cannot_be_replaced_after_plan_freeze(self):
        plan = self.plan()
        with (self.new / "runtime_provenance.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            c.require_completed_off(self.new, self.policy(plan), "e" * 64,
                                    plan=plan, reference_run=self.reference)

    def test_missing_runtime_field_refused(self):
        broken = dict(self.runtime)
        broken.pop("cuda")
        with self.assertRaisesRegex(ValueError, "fields missing"):
            c.scientific_environment(broken)


class RealLocalSourceIdentityTests(unittest.TestCase):
    def test_scientific_training_functions_and_update_kernel_unchanged(self):
        self.assertEqual(c.verify_scientific_kernels()["status"],
                         "unchanged_scientific_functions_and_update_sources")


if __name__ == "__main__":
    unittest.main()
