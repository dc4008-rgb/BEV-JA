"""Pure cap120 argv tests; no process, SSH, model or GPU execution."""
from __future__ import annotations
import argparse
import ast
import copy
from pathlib import Path
import unittest
import prepare_commands as commands


def synthetic_plan():
    return {'schema': 'r2-joint-clip120-followup-plan-v1',
        'status': 'fixed_user_requested_post_evaluation_followup',
        'selection_context': 'user_requested_120_after_six_group_development_results',
        'exploratory_posthoc': True,
        'groups': [{'id': 'cap120', 'max_norm': 120.0, 'executed_updates': 500, 'fresh_start': True}]}


def synthetic_config():
    return {'python': '/runtime/python', 'hostname': 'SYNTHETIC-ONLY',
        'child': '/inputs/child', 'r2_source': '/sources/meeting_full_repeat_20260924_r2',
        'v2_source': '/sources/joint_v2', 'reference_run': '/history/joint500/run',
        'off_run': '/new_off_recheck/run', 'output_root': '/outputs/cap120_followup'}


def parser_from_actual_source(path):
    module = ast.parse(path.read_text())
    funcs = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    function = funcs.get('parse_args', funcs.get('main'))
    statements = []
    for node in function.body:
        if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
               and x.func.attr == 'parse_args' for x in ast.walk(node)):
            break
        statements.append(node)
    namespace = {'argparse': argparse, 'Path': Path, 'ROOT': path.parent,
                 '__file__': str(path), '__doc__': 'CLI signature-only test'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['parser']


class FollowupCommandTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.config = synthetic_plan(), synthetic_config()
        self.root = Path(__file__).resolve().parent

    def build(self):
        return commands.build_commands(self.config, self.plan, plan_path='/control/CLIP_PLAN.json',
            plan_sha='1' * 64, package_sha='2' * 64, baselines_sha='3' * 64,
            control_root='/control/commands', package_root=self.root)

    def test_five_fixed_stages_and_posthoc_disclosure(self):
        matrix, inputs = self.build()
        self.assertEqual([r['id'] for r in matrix['commands']],
                         ['train_cap120', 'evaluate_cap120', 'audit_depth_cap120', 'audit_detection_cap120', 'collect_cap120'])
        self.assertEqual(matrix['total_training_updates'], 500)
        self.assertTrue(matrix['exploratory_posthoc'])
        self.assertFalse(matrix['preregistered_grid_point'])
        self.assertFalse(matrix['commands_executed'])
        self.assertEqual([g['id'] for g in inputs['groups']], ['cap120'])

    def test_every_generated_argv_parses_against_actual_sources(self):
        matrix, _ = self.build()
        for row in matrix['commands']:
            with self.subTest(command=row['id']):
                parser_from_actual_source(Path(row['argv'][1])).parse_args(row['argv'][2:])

    def test_exact_execution_off_and_new_output_paths(self):
        matrix, inputs = self.build()
        argv = matrix['commands'][0]['argv']
        self.assertEqual(argv[argv.index('--off-run') + 1], self.config['off_run'])
        self.assertEqual(argv[argv.index('--clip-group') + 1], 'cap120')
        self.assertNotIn('--updates', argv)
        output = inputs['groups'][0]
        for key, prefix in [('run', 'train'), ('evaluation', 'evaluate'), ('audit_depth', 'audit_depth'), ('audit_detection', 'audit_detection')]:
            row = next(r for r in matrix['commands'] if r['id'] == prefix + '_cap120')
            self.assertEqual(row['argv'][row['argv'].index('--output') + 1], output[key])
        last = matrix['commands'][-1]['argv']
        self.assertEqual(last[last.index('--output') + 1], self.config['output_root'] + '/CAP120_COMPARISON.json')

    def test_stage_dependencies_and_optional_flags(self):
        self.config.update(weights_root='/assets/weights', sd35_snapshot='/assets/sd35',
                           source_prefix='/old', source_root='/new')
        matrix, _ = self.build(); visited = set()
        for row in matrix['commands']:
            self.assertTrue(set(row['depends_on']) <= visited)
            visited.add(row['id'])
            parser_from_actual_source(Path(row['argv'][1])).parse_args(row['argv'][2:])
            self.assertEqual(row['execution_mode'], 'argv_without_shell')
        self.assertIn('--sd35-snapshot', matrix['commands'][0]['argv'])
        self.assertNotIn('--sd35-snapshot', matrix['commands'][1]['argv'])

    def test_missing_off_invalid_paths_or_old_grid_are_rejected(self):
        for key, value in [('off_run', None), ('off_run', 'relative'), ('hostname', 'bad host')]:
            bad = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                commands.validate_config(bad)
        for field, value in [('schema', 'r2-joint-clip-train-grid-v1'), ('exploratory_posthoc', False),
                             ('selection_context', 'predeclared_train_only')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                commands.validate_groups(dict(self.plan, **{field: value}))

    def test_real_followup_builder_and_evaluator_plan_contract_agree(self):
        import json
        import hashlib
        import clip_evaluate as evaluation
        from test_followup_contract import FollowupContractTests
        fixture = FollowupContractTests('test_new_driver_needs_full_reverified_off_then_is_bound')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        plan = fixture.plan()
        self.assertEqual(commands.validate_groups(plan)[0]['fresh_start'], True)
        run = fixture.root / 'synthetic_cap120'; run.mkdir()
        path = run / 'CLIP_PLAN.json'; path.write_text(json.dumps(plan))
        policy = {'group_id': 'cap120', 'max_norm': 120.0,
                  'grid_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'scope': 'global_total_latent_gradient', 'position': 'after_100000_before_lr', 'norm_dtype': 'float32'}
        self.assertEqual(evaluation.validate_clip_plan(run, policy), plan)
        self.plan = plan
        matrix, _ = self.build()
        self.assertEqual(len(matrix['commands']), 5)

    def test_no_extra_group_wrong_threshold_or_budget(self):
        for field, value in [('id', 'p50'), ('max_norm', 121.), ('max_norm', True), ('executed_updates', 100)]:
            plan = copy.deepcopy(self.plan); plan['groups'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                commands.validate_groups(plan)
        with self.assertRaises(ValueError):
            commands.validate_groups(dict(self.plan, groups=self.plan['groups'] * 2))


if __name__ == '__main__':
    unittest.main()
