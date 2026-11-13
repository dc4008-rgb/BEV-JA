"""Bind the new protocol to actual retained local historical receipts."""
import json
from pathlib import Path
import unittest

import followup_contract as c


class HistoricalBindingsTests(unittest.TestCase):
    def test_original_joint500_reference_matches_readback(self):
        project = Path(c.__file__).resolve().parents[4]
        bindings = c.reference_bindings()
        receipt = project / bindings["source_receipt"]
        self.assertEqual(c.sha(receipt), bindings["source_receipt_sha256"])
        run = json.loads(receipt.read_text())["runs"]["ours_joint500"]
        self.assertEqual(bindings["files"]["summary.json"], run["evaluation"]["run_summary_sha256"])
        for name, key in (("attack_trace.jsonl", "attack_trace_sha256"),
                          ("schedule.json", "schedule_sha256"),
                          ("state_evidence/MANIFEST.json", "state_evidence_manifest_sha256")):
            self.assertEqual(bindings["files"][name], run["training"][key])

    def test_prior_six_group_and_off_hashes_match_actual_readback(self):
        project = Path(c.__file__).resolve().parents[4]
        path = project / "11_远程运维脚本/r2_joint_clip_sweep_20261003_live/RESULT_READBACK_20261004.json"
        bindings = c.followup_bindings()
        self.assertEqual(c.sha(path), bindings["local_readback_sha256"])
        receipt = json.loads(path.read_text())
        summary = receipt["groups"]["off"]["summary"]
        for name, key in (("grid", "prior_grid_sha256"), ("comparison", "seen_comparison_sha256"),
                          ("points", "seen_points_sha256"), ("baselines", "baselines_sha256")):
            self.assertEqual(receipt["documents"][name]["sha256"], bindings[key])
        self.assertEqual(summary["package_manifest_sha256"], bindings["prior_package_sha256"])
        self.assertEqual(receipt["groups"]["off"]["summary_sha256"], bindings["off_files"]["summary.json"])
        for name, key in (("attack_trace.jsonl", "attack_trace_sha256"), ("schedule.json", "schedule_sha256"),
                          ("runtime_provenance.json", "runtime_provenance_sha256"),
                          ("CLIPPING_COMPARISON_GATE.json", "clipping_comparison_gate_sha256"),
                          ("state_evidence/MANIFEST.json", "state_evidence_manifest_sha256")):
            self.assertEqual(summary[key], bindings["off_files"][name])


if __name__ == "__main__":
    unittest.main()
