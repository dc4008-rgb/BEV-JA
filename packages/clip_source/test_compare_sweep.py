"""Synthetic numerical contracts only; none of these values are experiment results."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_sweep import compare_files, main, summarize, write_result


GRID_SHA = "1" * 64


def fixture(deltas=(1., -1., 2., -2.), off_delta=100., two_comparisons=False):
    groups = [{"id": "off", "max_norm": None}] + [{"id": f"cap{i}", "max_norm": float(i)} for i in range(len(deltas))]
    plan = {"schema": "r2-joint-clip-train-grid-v1", "status": "thresholds_frozen_from_train", "groups": groups,
            "unique_run_count": len(groups), "threshold_rule": {"validation_or_test_used": False, "post_evaluation_refinement_allowed": False}}
    reference = {"id": "mono_abs_rel", "task": "depth", "model": "monodepth2", "metric": "abs_rel_clean_ref",
                 "baseline_method": "SAAM_Mono", "value": 10., "unit": "%", "direction": "larger_is_more_prediction_deviation"}
    references = [reference]
    if two_comparisons:
        references.append({"id": "ssd_tnr", "task": "detection", "model": "ssd300", "metric": "tnr_percent",
                           "baseline_method": "Thys", "value": 20., "unit": "%", "direction": "larger_is_more_AP50_drop"})
    baselines = {"schema": "r2-joint-clip-baselines-v1", "comparisons": references}
    rows = []
    for baseline in references:
        for group, delta in zip(groups, (off_delta,) + tuple(deltas)):
            ours = baseline["value"] + delta
            rows.append({"group_id": group["id"], "max_norm": group["max_norm"], "comparison_id": baseline["id"],
                         "ours_value": ours, "baseline_value": baseline["value"], "delta": ours - baseline["value"],
                         **{field: baseline[field] for field in ("unit", "metric", "model", "task")}, "audit_verified": True})
    points = {"schema": "r2-joint-clip-evaluated-points-v1", "grid_sha256": GRID_SHA, "rows": rows}
    return plan, points, baselines


def run_fixture(data):
    return summarize(*data, grid_sha256=GRID_SHA)


class CompareSweepTests(unittest.TestCase):
    def test_multiple_crossings_are_all_retained_without_interpolation(self):
        result = run_fixture(fixture())
        comparison = result["comparisons"][0]
        self.assertEqual(len(comparison["crossing_brackets"]), 3)
        self.assertEqual(len(comparison["all_intersections"]), 3)
        self.assertEqual(comparison["range_status"], "observed_multiple_sign_change_brackets")
        self.assertEqual([item["transition_as_threshold_increases"] for item in comparison["crossing_brackets"]],
                         ["higher_to_lower", "lower_to_higher", "higher_to_lower"])
        for item in comparison["crossing_brackets"]:
            self.assertIsNone(item["interpolated_critical_value"])
        self.assertFalse(result["policy"]["monotonicity_assumed"])
        self.assertFalse(result["policy"]["threshold_selection"])

    def test_all_positive_and_all_negative_have_no_finite_crossings(self):
        for deltas, expected in [((1., 2., 3.), "higher"), ((-1., -2., -3.), "lower")]:
            comparison = run_fixture(fixture(deltas, off_delta=-deltas[0]))["comparisons"][0]
            self.assertEqual(comparison["range_status"], f"all_finite_points_{expected}")
            self.assertEqual(comparison["crossing_brackets"], [])
            self.assertEqual(comparison["all_intersections"], [])

    def test_exact_ties_do_not_become_interpolated_brackets(self):
        comparison = run_fixture(fixture((-1., 0., 1.)))["comparisons"][0]
        self.assertEqual(comparison["range_status"], "higher_and_lower_points_separated_by_exact_grid_ties")
        self.assertEqual(len(comparison["adjacent_sign_changes"]), 2)
        self.assertEqual(comparison["crossing_brackets"], [])
        self.assertEqual([item["max_norm"] for item in comparison["exact_ties"]], [1.])
        self.assertEqual(comparison["all_intersections"][0]["kind"], "exact_grid_tie")

    def test_all_equal_has_only_exact_sampled_ties(self):
        comparison = run_fixture(fixture((0., 0., 0.)))["comparisons"][0]
        self.assertEqual(comparison["range_status"], "all_finite_points_equal")
        self.assertEqual(len(comparison["exact_ties"]), 3)
        self.assertEqual(comparison["adjacent_sign_changes"], [])

    def test_off_is_independent_even_when_equal_or_opposite(self):
        for off_delta in (0., -100.):
            comparison = run_fixture(fixture((1., 2.), off_delta=off_delta))["comparisons"][0]
            self.assertIsNone(comparison["off_reference"]["max_norm"])
            self.assertEqual(comparison["off_reference"]["delta"], off_delta)
            self.assertEqual(comparison["all_intersections"], [])

    def test_nonmonotonic_positive_values_do_not_invent_crossing(self):
        comparison = run_fixture(fixture((1., 8., 2., 7.)))["comparisons"][0]
        self.assertEqual(comparison["range_status"], "all_finite_points_higher")
        self.assertEqual(comparison["crossing_brackets"], [])

    def test_unsorted_rows_and_groups_are_sorted_only_by_finite_cap(self):
        data = fixture()
        data[0]["groups"].reverse()
        data[1]["rows"].reverse()
        comparison = run_fixture(data)["comparisons"][0]
        self.assertEqual([point["max_norm"] for point in comparison["finite_points"]], [0., 1., 2., 3.])

    def test_two_comparisons_preserve_task_direction_and_full_coverage(self):
        result = run_fixture(fixture(two_comparisons=True))
        self.assertEqual(result["verified_point_count"], 10)
        self.assertEqual(result["comparison_count"], 2)
        self.assertEqual(result["comparisons"][0]["direction"], "larger_is_more_prediction_deviation")
        self.assertEqual(result["comparisons"][1]["direction"], "larger_is_more_AP50_drop")

    def test_missing_and_duplicate_points_rejected(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[0]))):
            data = fixture(two_comparisons=True)
            mutation(data[1]["rows"])
            with self.assertRaises(ValueError):
                run_fixture(data)

    def test_unknown_alias_group_and_comparison_rejected(self):
        for key, value in (("group_id", "alias_not_canonical"), ("comparison_id", "missing")):
            data = fixture()
            data[1]["rows"][0][key] = value
            with self.assertRaises(ValueError):
                run_fixture(data)

    def test_baseline_and_context_tampering_rejected(self):
        for field, value in (("baseline_value", 9.), ("delta", 1.), ("unit", "native"), ("task", "other"),
                             ("model", "other"), ("metric", "other"), ("max_norm", 1.), ("audit_verified", 1)):
            data = fixture()
            data[1]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_fixture(data)

    def test_nonfinite_and_boolean_values_rejected(self):
        for field in ("ours_value", "baseline_value", "delta"):
            for value in (float("nan"), float("inf"), -float("inf"), True):
                data = fixture()
                data[1]["rows"][1][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    run_fixture(data)
        for value in (float("nan"), float("inf"), -1., True):
            data = fixture()
            data[0]["groups"][1]["max_norm"] = value
            with self.assertRaises(ValueError):
                run_fixture(data)

    def test_bad_plan_and_baseline_contracts_rejected(self):
        for mutation in (
            lambda data: data[0].update(status="draft"),
            lambda data: data[0]["groups"].append(dict(data[0]["groups"][1])),
            lambda data: data[0]["groups"][1].update(max_norm=1.),
            lambda data: data[0]["groups"][0].update(id="not_off"),
            lambda data: data[0]["threshold_rule"].update(post_evaluation_refinement_allowed=True),
            lambda data: data[2]["comparisons"].append(dict(data[2]["comparisons"][0])),
            lambda data: data[2]["comparisons"][0].update(direction="signed_log_is_always_further"),
            lambda data: data[2]["comparisons"][0].update(value=float("nan")),
            lambda data: data[1].update(grid_sha256="2" * 64),
        ):
            data = fixture()
            mutation(data)
            with self.assertRaises(ValueError):
                run_fixture(data)

    def test_input_objects_remain_unchanged(self):
        data = fixture(two_comparisons=True)
        original = copy.deepcopy(data)
        run_fixture(data)
        self.assertEqual(data, original)

    def test_computed_subtraction_overflow_rejected(self):
        data = fixture((1.,))
        data[2]["comparisons"][0]["value"] = -1e308
        for row in data[1]["rows"]:
            row.update(baseline_value=-1e308, ours_value=1e308, delta=1.)
        with self.assertRaises(ValueError):
            run_fixture(data)

    def write_fixture(self, root):
        data = fixture()
        plan_path, points_path, baseline_path = [root / name for name in ("plan.json", "points.json", "baselines.json")]
        plan_path.write_text(json.dumps(data[0]), encoding="utf-8")
        data[1]["grid_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        points_path.write_text(json.dumps(data[1]), encoding="utf-8")
        baseline_path.write_text(json.dumps(data[2]), encoding="utf-8")
        return plan_path, points_path, baseline_path

    def test_cli_creates_bound_new_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan, points, baselines = self.write_fixture(root)
            output = root / "result.json"
            arguments = ["--plan", str(plan), "--points", str(points), "--baselines", str(baselines), "--output", str(output)]
            main(arguments)
            result = json.loads(output.read_text())
            self.assertEqual(result["input_bindings"]["plan"]["sha256"], hashlib.sha256(plan.read_bytes()).hexdigest())
            self.assertEqual(result["verified_point_count"], 5)
            with self.assertRaises(FileExistsError):
                main(arguments)

    def test_plan_byte_mutation_and_optional_baseline_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan, points, baselines = self.write_fixture(root)
            plan.write_text(plan.read_text() + "\n")
            with self.assertRaises(ValueError):
                compare_files(plan, points, baselines)
            plan, points, baselines = self.write_fixture(root)
            payload = json.loads(points.read_text())
            payload["baselines_sha256"] = "0" * 64
            points.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                compare_files(plan, points, baselines)

    def test_duplicate_json_keys_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for payload in ('{"schema":"a","schema":"b"}', '{"schema":NaN}'):
                plan, points, baselines = self.write_fixture(root)
                points.write_text(payload)
                with self.assertRaises(ValueError):
                    compare_files(plan, points, baselines)

    def test_output_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "existing.json"
            target.write_text("preserve")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                write_result(run_fixture(fixture()), link)
            self.assertEqual(target.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
