"""No weights/images: protocol tamper, provenance and bounded child-process checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import inputs
import run_matrix


class InputContractTests(unittest.TestCase):
    def test_real_local_provenance_and_train_schedule(self):
        result = inputs.verify_inputs(require_sealed=False)
        self.assertEqual(result['parent_files_verified'],64)
        self.assertFalse(result['source_images_opened'])

    def test_protocol_rejects_budget_boolean_float_test_or_resume(self):
        original = inputs.protocol_contract()
        cases = []
        for key,value in [('test_access_allowed',True),('resume_enabled',True),('dual_allowed',True)]:
            altered = copy.deepcopy(original);altered[key]=value;cases.append(altered)
        for value in [True,100.0,400]:
            altered=copy.deepcopy(original);altered['train_schedule']['updates']=value;cases.append(altered)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json'
            for value in cases:
                p.write_text(json.dumps(value))
                with patch.object(inputs,'PROTOCOL_PATH',p), self.assertRaises(ValueError):
                    inputs.load_protocol()

    def test_paths_refuse_parent_escape_and_runs_root(self):
        for p in [inputs.PARENT/'runs/x',inputs.ROOT/'runs',inputs.ROOT/'runs/../../escape']:
            with self.assertRaises(ValueError):inputs.candidate_run_path(p)

    def test_both_contextual_matching_and_car_only_cannot_silently_change(self):
        original = inputs.protocol_contract()
        self.assertEqual(original['objectives'], ['C0', 'C3'])
        self.assertEqual(original['C3_target_class'], 3)
        self.assertEqual(set(original['C3_matching']), {'fasterrcnn', 'ssd300'})
        mutations = []
        for model in ('fasterrcnn', 'ssd300'):
            value = copy.deepcopy(original)
            value['C3_matching'][model]['iou_threshold'] = 0.1
            mutations.append(value)
        value = copy.deepcopy(original); value['C3_target_class'] = [3, 6, 8]; mutations.append(value)
        value = copy.deepcopy(original); value['objectives'] = ['C0', 'C2']; mutations.append(value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'protocol.json'
            for value in mutations:
                path.write_text(json.dumps(value))
                with patch.object(inputs, 'PROTOCOL_PATH', path), self.assertRaises(ValueError):
                    inputs.load_protocol()

    def test_entry_hash_duplicate_symlink_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);(b/'a').write_text('fixed')
            entry={'path':'a','sha256':inputs.sha(b/'a'),'bytes':5}
            self.assertEqual(inputs._verify_entries(b,[entry]),{'a'})
            with self.assertRaises(ValueError):inputs._verify_entries(b,[entry,entry])
            (b/'link').symlink_to(b/'a')
            with self.assertRaises(ValueError):inputs._verify_entries(b,[dict(entry,path='link')])
            with self.assertRaises(ValueError):inputs._verify_entries(b,[dict(entry,path='../a')])
            (b/'a').write_text('other')
            with self.assertRaises(ValueError):inputs._verify_entries(b,[entry])


class MatrixTests(unittest.TestCase):
    def test_all_training_before_validation_and_fixed_identity_bindings(self):
        args=SimpleNamespace(expected_hostname='host',package_manifest_sha256='a'*64,
                             sd35_snapshot=Path('/snapshot'),source_prefix=None,source_root=None,weights_root=None)
        stages=run_matrix.build_stages(args,inputs.ROOT/'runs/test')
        self.assertEqual([s['name'] for s in stages],inputs.protocol_contract()['stage_order'])
        for stage in stages[:4]:
            self.assertIn('--expected-hostname',stage['command'])
            self.assertIn('--package-manifest-sha256',stage['command'])
        self.assertNotIn('--run',stages[-1]['command'])

    def execute(self,code,timeout=5):
        with tempfile.TemporaryDirectory() as d:
            stage={'name':'synthetic_child','kind':'comparison','output':d,
                   'command':[sys.executable,'-B','-c',code]}
            return run_matrix.execute_stage(stage,timeout,dict(__import__('os').environ),Path(d))

    def test_failure_retained_without_retry(self):
        result=self.execute('raise SystemExit(7)')
        self.assertEqual(result['status'],'failed_retained')
        self.assertEqual(result['exit_code'],7)

    def test_timeout_stops_owned_process(self):
        result=self.execute('import time; time.sleep(3)',timeout=.05)
        self.assertEqual(result['status'],'timeout_retained')
        self.assertNotEqual(result['exit_code'],0)

    def test_success_distinct_from_effectiveness(self):
        result=self.execute('print("synthetic only")')
        self.assertEqual(result['status'],'process_completed')
        self.assertEqual(result['exit_code'],0)
        self.assertNotIn('attack_effectiveness',result)


class StageSummaryTests(unittest.TestCase):
    def fixture(self,name):
        protocol=inputs.protocol_contract()
        verified={'manifest_sha256':'a'*64,'protocol_sha256':'b'*64}
        kind='comparison' if name=='comparison' else name.split('_')[0]
        stage={'name':name,'kind':kind}
        summary={'task_variant':'det-only','test_data_opened':False,
                 'package_manifest_sha256':'a'*64,'protocol_sha256':'b'*64}
        if kind in ('train','validation'):
            objective=name.rsplit('_',1)[1]
            summary.update(attack_objective=objective,candidate_id=protocol['candidate_ids'][objective])
        if kind=='train':
            summary.update(identity=inputs.IDENTITY,status='train_optimization_complete',
                           executed_updates=100,training_models=['fasterrcnn','ssd300'],
                           seed=20260919,formal_test_result=False,validation_data_opened=False,
                           resume_enabled=False,checkpoint_selection='fixed final100 only',
                           budget={k:copy.deepcopy(protocol['budget_per_run']) for k in ('planned','attempted','completed')},
                           query_accounting=copy.deepcopy(protocol['budget_per_run']))
        elif kind=='validation':
            summary.update(status='development_validation_complete',updates=100,records=240,
                           evaluation_models=['fasterrcnn','ssd300'],evaluation_forward_total=320,
                           evaluation_model_scene_forwards={'fasterrcnn':160,'ssd300':160},
                           transfer_models_used=False,formal_test_allowed=False,
                           checkpoint_selection='fixed_final100',automatic_promotion=False)
        else:
            summary.update(identity=inputs.IDENTITY,status='complete_observations',
                           comparison='fixed_C0_C3_final100_validation_descriptive',
                           formal_test_allowed=False,candidate_promoted=False,additional_model_forwards=0,
                           total_existing_validation_forwards=640,total_existing_validation_rows=480,
                           common_control_inputs_exact=True,common_input_hash_rows_compared=240,
                           shared_control_predictions_exact=False)
        return stage,summary,protocol,verified

    def test_accepts_all_five_exact_stage_summaries_without_claiming_prediction_equality(self):
        for name in inputs.protocol_contract()['stage_order']:
            with self.subTest(name=name):
                args=self.fixture(name)
                self.assertEqual(run_matrix.validate_stage_summary(*args),args[1]['status'])

    def test_exit_zero_does_not_accept_failed_or_partial_scientific_summary(self):
        for name in inputs.protocol_contract()['stage_order']:
            for wrong_status in ('failed_retained','partial','process_completed',None):
                with self.subTest(name=name,status=wrong_status):
                    stage,summary,protocol,verified=self.fixture(name)
                    summary['status']=wrong_status
                    with self.assertRaises(ValueError):
                        run_matrix.validate_stage_summary(stage,summary,protocol,verified)

    def test_every_stage_requires_current_package_protocol_and_test_isolation(self):
        for name in inputs.protocol_contract()['stage_order']:
            for field,value in [('package_manifest_sha256','c'*64),('protocol_sha256','d'*64),
                                ('test_data_opened',True),('test_data_opened',0),('task_variant','dual')]:
                with self.subTest(name=name,field=field,value=value):
                    stage,summary,protocol,verified=self.fixture(name)
                    summary[field]=value
                    with self.assertRaises(ValueError):
                        run_matrix.validate_stage_summary(stage,summary,protocol,verified)

    def test_objective_and_candidate_cannot_be_swapped_or_revive_old_version(self):
        for name in ('train_C0','train_C3','validation_C0','validation_C3'):
            for field,value in [('attack_objective','c0'),('candidate_id','normalized-v17-ours'),
                                ('attack_objective','C3' if name.endswith('C0') else 'C0')]:
                with self.subTest(name=name,field=field):
                    stage,summary,protocol,verified=self.fixture(name)
                    summary[field]=value
                    with self.assertRaises(ValueError):
                        run_matrix.validate_stage_summary(stage,summary,protocol,verified)

    def test_train_rejects_partial_or_retyped_budgets_and_short_updates(self):
        for field,value in [('executed_updates',25),('executed_updates',100.0),('seed',True)]:
            stage,summary,protocol,verified=self.fixture('train_C0')
            summary[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                run_matrix.validate_stage_summary(stage,summary,protocol,verified)
        for section in ('planned','attempted','completed','query_accounting'):
            for value in (True,200.0,199,201):
                stage,summary,protocol,verified=self.fixture('train_C3')
                counters=summary[section] if section=='query_accounting' else summary['budget'][section]
                counters['autograd_calls']=value
                with self.subTest(section=section,value=value),self.assertRaises(ValueError):
                    run_matrix.validate_stage_summary(stage,summary,protocol,verified)

    def test_validation_and_comparison_refuse_incomplete_extra_queries_or_promotion(self):
        cases={'validation_C3':[('records',239),('evaluation_forward_total',240),
                               ('evaluation_model_scene_forwards',{'fasterrcnn':160,'ssd300':159}),
                               ('automatic_promotion',True),('checkpoint_selection','best')],
               'comparison':[('total_existing_validation_forwards',639),
                             ('total_existing_validation_rows',479),('additional_model_forwards',1),
                             ('additional_model_forwards',False),('candidate_promoted',True),
                             ('common_control_inputs_exact',False),('identity','normalized-v20')]}
        for name,changes in cases.items():
            for field,value in changes:
                stage,summary,protocol,verified=self.fixture(name)
                summary[field]=value
                with self.subTest(name=name,field=field),self.assertRaises(ValueError):
                    run_matrix.validate_stage_summary(stage,summary,protocol,verified)


if __name__=='__main__': unittest.main()
