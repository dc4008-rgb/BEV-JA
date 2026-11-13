"""Non-GPU contracts for fixed selection and honest heterogeneous case owners."""
import ast,copy,hashlib,json,sys
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import sample_contract as contract
import freeze_plan
import case_owners200 as owners
import selection_contract200 as selection
import sample_evaluate as evaluate
import sample_audit as audit
from source_bindings import verify_tree
WORKSPACE=next(p for p in ROOT.parents if (p/'.workspace_root').is_file())
MANIFEST=WORKSPACE/'复现/规范化流程后/01_公共数据集/runtime_v1/train_validation_manifest.csv'
SELECTION=ROOT.parent/'SAMPLE_SELECTION.json'
OLDPLANS=[ROOT/'original_plans'/f'PLAN_SHARD_{i}.json' for i in (0,1)]

def draft():
    return freeze_plan.build_plan(MANIFEST,'development-draft',selection=SELECTION,
        selection_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(),original_plans=OLDPLANS)

def eval_plan(shard=0):
    p=draft();p['samples']=contract.partition_samples(p['master_samples'])[str(shard)]
    p['budget']={'sample_count':50,'updates_per_sample':500,'total_updates':25000}
    p['_plan_sha256']='d'*64;p['admission_binding']={'document':{'package_sha256':'e'*64}}
    return p

class Contract200Tests(unittest.TestCase):
    def test_actual200_fixed_original40_and_four_balanced_native50_shards(self):
        p=draft();groups=contract.partition_samples(p['master_samples']);ids=[]
        self.assertEqual(p['master_samples_sha256'],'78fc617bcc0534c95288b6f1a91af5eb68707af49d2bcaa9b0a631788dc6a96a')
        for rows in groups.values():
            self.assertEqual(len(rows),50)
            for d in ('kitti','nuscenes'):
                self.assertEqual(sum(r['dataset']==d for r in rows),25)
                self.assertEqual(sum(r['dataset']==d and r['selection_role']=='retained_validation40' for r in rows),5)
            ids += [r['sample_id'] for r in rows]
        self.assertEqual(len(set(ids)),200)
        self.assertFalse(set(ids)&set(contract.DIAGNOSTIC_IDS))

    def test_candidate_and_draft_cannot_become_executable_without_actual_admission(self):
        p=draft()
        with self.assertRaisesRegex(ValueError,'not executable'):contract.validate_plan(p)
        doc=copy.deepcopy(p['selection_binding']['document']);doc['status']='candidate_pending_source_image_qa'
        with self.assertRaisesRegex(ValueError,'QA must be frozen'):selection.validate_selection(doc)
        self.assertEqual(selection.validate_selection(doc,allow_candidate=True)['sample_count'],200)

    def test_old_owner_preserves_actual_original20_hash_new_owner_uses_new50_hash(self):
        p=eval_plan()
        for sample in p['samples']:
            owner=owners.resolve_case_owner(p,sample)
            if sample['selection_role']=='retained_validation40':
                self.assertIn(owner['_plan_sha256'],owners.OLD_PLAN_SHA.values());self.assertNotEqual(owner['_plan_sha256'],p['_plan_sha256'])
                self.assertEqual(len(owner['samples']),20)
            else:
                self.assertIs(owner,p);self.assertEqual(len(owner['samples']),50)

    def test_raw_case_binds_both_original_training_owner_and_new_evaluation_owner(self):
        p=eval_plan();sample=next(s for s in p['samples'] if s['selection_role']=='retained_validation40')
        owner=owners.resolve_case_owner(p,sample)
        patch={k:sample[k] for k in ('sample_id','case_id','source_sha256')};patch.update(sha256='a'*64,tensor_sha256='b'*64)
        case={k:sample[k] for k in ('sample_id','case_id','dataset','source_split','source_sha256')}
        case.update(plan_sha256=owner['_plan_sha256'],evaluation_plan_sha256=p['_plan_sha256'],training_origin=owners.case_origin_binding(p,sample),
            train_summary_sha256='c'*64,patches={'zero':patch,'final':patch})
        audit.check_case(case,sample,p)
        for key,bad in (('plan_sha256',p['_plan_sha256']),('evaluation_plan_sha256',owner['_plan_sha256'])):
            with self.subTest(key=key),self.assertRaises(ValueError):audit.check_case({**case,key:bad},sample,p)

    def test_retained_patch_metadata_cannot_be_relabelled_as_new_plan(self):
        p=eval_plan();sample=next(s for s in p['samples'] if s['selection_role']=='retained_validation40')
        owner=owners.resolve_case_owner(p,sample)
        metadata={k:sample[k] for k in ('sample_id','case_id','source_sha256')};metadata.update(plan_sha256=owner['_plan_sha256'],updates=500)
        evaluate.validate_patch_metadata({'metadata':metadata},owner,sample,500)
        with self.assertRaises(ValueError):evaluate.validate_patch_metadata({'metadata':metadata},p,sample,500)

    def test_original_plan_sha_source_and_200_membership_tampering_rejected(self):
        p=eval_plan();sample=p['samples'][0]
        bad=copy.deepcopy(p);bad['original_plan_bindings']['0']['sha256']='f'*64
        with self.assertRaises(ValueError):owners.resolve_case_owner(bad,sample)
        bad=copy.deepcopy(p['selection_binding']['document']);bad['samples'][0]['source_sha256']='f'*64
        with self.assertRaises(ValueError):selection.validate_selection(bad)
        bad=copy.deepcopy(p['selection_binding']['document']);bad['samples'][0]['source_split']='test'
        with self.assertRaises(ValueError):selection.validate_selection(bad)

    def test_computation_bytes_native_metrics_and_all_nonrouting_functions_unchanged(self):
        for name in ('sample_training.py','periodic_controller.py','task_gradient_balance.py','detection_control.py',
            'depth_hinge_objective.py','artifact_io.py','periodic_training_audit.py','batch_training.py','source_bindings.py'):
            self.assertEqual((ROOT/name).read_bytes(),(ROOT/'legacy_sci6666'/name).read_bytes(),name)
        allowed={'sample_evaluate.py':{'load_cases','main'},'sample_audit.py':{'check_case','audit_evaluation','merge_results'}}
        for name,changed in allowed.items():
            def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            old,new=functions(ROOT/'legacy_sci6666'/name),functions(ROOT/name)
            self.assertEqual(set(old),set(new))
            for key in old:
                if key not in changed:self.assertEqual(old[key],new[key],name+':'+key)
        self.assertEqual(verify_tree(ROOT/'legacy_sci6666',owners.LEGACY_SHA)['verified_files'],28)


    def test_actual_new_namespace_source_qa_copies_are_mandatory(self):
        p=draft();selection.validate_selection_support(p)
        for key in ('source_input_proof','manual_qa'):
            bad=copy.deepcopy(p);bad['selection_support_bindings'][key]['path']=p['selection_binding']['document'][
                'source_input_proof_binding' if key=='source_input_proof' else 'manual_qa_binding']['path']
            with self.assertRaisesRegex(ValueError,'new source/QA byte binding differs'):
                selection.validate_selection_support(bad)

    def test_native500_case_adapter_routes_the_unchanged_audit_to_actual_training_owner(self):
        import native_audit200 as native
        p=eval_plan()
        for role in ('retained_validation40','added_train160'):
            sample=next(s for s in p['samples'] if s['selection_role']==role)
            expected=owners.resolve_case_owner(p,sample)['_plan_sha256']
            def verify(context,row,run,**kwargs):
                self.assertEqual(context['plans'][str(row['original_shard_id'])]['_plan_sha256'],expected)
                self.assertEqual(row['plan_sha256'],expected);self.assertEqual(row['sample_id'],sample['sample_id'])
                return {'owner_checked':True}
            with patch.object(native,'validate_native_case',side_effect=verify):
                self.assertEqual(native.audit_case(p,sample,'/private/tmp/unused')['evaluation_plan_sha256'],p['_plan_sha256'])

    def test_native_audit_arithmetic_functions_are_original_b118_ast(self):
        current=ast.parse((ROOT/'native_audit200.py').read_text())
        original=ast.parse((WORKSPACE/'11_远程运维脚本/periodic_foa_parallel_recovery_20261014_prepare/worker_training_and_audit.py').read_text())
        names={'validate_native_case','native_remaining_math','replay_math_probe','configure_math_runtime'}
        for name in names:
            a=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)

if __name__=='__main__':unittest.main()
