"""New full200 execution/ownership contract; original80D numerical functions unchanged.

This module validates plans and receipt *bindings*. Actual root full-capture
acceptance, current process/sourceCFG/GPU/resource admission remain runtime roles.
It never manufactures a completed diagnostic, current facts, or admission.
"""
from pathlib import Path
import copy,hashlib,json,math
ROOT=Path(__file__).resolve().parent
IDENTITY='ours_complete_eot1_parallel200_v1'
SCHEMA='ours-complete-eot1-parallel200-plan-v1'
REFERENCE_SHA='cca50857551d8125dc83a0c77de34fca955471b840e78dfdfea39a5059e3dc33'
POLICY='all_original200_fresh_complete_EOT1_no_retained_case_reuse'
TEMPLATE_STATUS='formal50_template_requires_actual_current_diagnostic_and_admission'

def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return sha(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def reference():
 b=(ROOT/'MASTER200_REFERENCE_001.json').read_bytes();need(sha(b)==REFERENCE_SHA,'exact original master/source/config reference');return json.loads(b)
def configure_science(mode='C1'):
 need(mode in('B1','C1'),'only EOT1 serial diagnostic or four-stream current science')
 import sample_contract as sc
 sc.configure_benchmark(mode)
 return sc

def validate_master_plan_reference(plan):
    # Reuse accepted source-only selection; do not repeat source QA or silently
    # follow the old host's absolute evidence paths in a new package namespace.
    r=reference();need(plan['master_samples']==r['master_samples']and plan['master_samples_sha256']==r['master_samples_sha256']and canonical(plan['master_samples'])==r['master_samples_sha256'],'exact already-accepted original200 literal source/ROI/role population')
    b=(ROOT/'SAMPLE_SELECTION.json').read_bytes();selection=json.loads(b);binding=plan['selection_binding']
    need(sha(b)==binding['sha256']=='3c8432679cd4926ec8dea6e4e8500ca1a95786f43cfe23155d2f53a1c730680b'and selection==binding['document']and canonical(selection)==binding['document_sha256']and selection['samples']==r['master_samples'],'whole immutable original200 selection bytes/document and same200')
    roles={'source_input_proof':('SOURCE_INPUT_PROOF.json','source_input_proof_binding'),'manual_qa':('SOURCE_RGB_QA.json','manual_qa_binding')}
    need(set(plan['selection_support_bindings'])==set(roles),'both original sourceRGB proof and QA bindings')
    for key,(name,selkey)in roles.items():
        whole=(ROOT/name).read_bytes();doc=json.loads(whole);bound=plan['selection_support_bindings'][key]
        need(sha(whole)==bound['sha256']==selection[selkey]['sha256']and doc==bound['document']and canonical(doc)==bound['document_sha256'],'exact original source-only QA/proof copied bytes; historical paths are provenance only')
    need(plan['partition']==r['partition'],'exact original four fixed50 order')
    from case_owners200 import validate_original_plan_bindings
    validate_original_plan_bindings(plan)
    by_id={x['sample_id']:x for x in r['master_samples']};shards={key:[by_id[i]for i in ids]for key,ids in r['partition']['shard_members'].items()}
    need(set(shards)=={'0','1','2','3'}and all(len(x)==50 for x in shards.values())and len({s['sample_id']for x in shards.values()for s in x})==200,'exact disjoint200 four50')
    return r['master_samples'],shards

def _shared(plan,mode,expected_master_descriptor,allowed_samples_ordered):
 sc=configure_science(mode);r=reference();p=Path(expected_master_descriptor['path'])
 need(p.resolve()==(ROOT/'MASTER200_REFERENCE_001.json').resolve()and not p.is_symlink(),'exact own original200 reference member')
 b=p.read_bytes();need(len(b)==expected_master_descriptor['bytes']and sha(b)==expected_master_descriptor['sha256']==REFERENCE_SHA,'whole original200 reference descriptor')
 need(plan['schema']==SCHEMA and plan['identity']==IDENTITY and plan['ablation_variant']=='complete','new complete EOT1 identity and plan')
 need(plan['case_owner_policy']==POLICY and plan.get('original_plan_bindings')is None,'new fresh200 ownership, never legacy result owner')
 need(plan['master_samples']==r['master_samples']and plan['master_samples_sha256']==r['master_samples_sha256']and plan['partition']==r['partition'],'same original literal200 rows/ROI/source/order/fixedfour partitions')
 need(plan['execution_mode']==mode and type(allowed_samples_ordered)in(list,tuple) and [x['sample_id']for x in plan['samples']]==list(allowed_samples_ordered),'explicit frozen current execution mode/scope order')
 by_id={s['sample_id']:s for s in r['master_samples']}
 need(all(s==by_id.get(s['sample_id'])for s in plan['samples']),'source pixels/ROI/roles/dimensions unchanged')
 updates=plan['budget']['updates_per_sample'];expected=copy.deepcopy(r['original_scientific_config']);expected['eot']['random_views']=1;expected['outer_rounds']=updates//25
 need(plan['scientific_config']==expected==sc.scientific_config_for('complete',updates),'only oldcomplete random3→1; all FOA/EMA/task/controller/math controls preserved')
 need(plan['call_budget_per_sample']==sc.planned_budget(updates),'truthful2sharedviews dualVJP callback budget')
 return sc,r

def validate_training_plan(plan,kind,mode,expected_master_descriptor,allowed_samples_ordered):
 sc,r=_shared(plan,mode,expected_master_descriptor,allowed_samples_ordered)
 if kind=='diag25':
  need(plan['budget']==dict(sample_count=2,updates_per_sample=25,total_updates=50)and plan['samples']==r['fixed_diagnostic_samples'],'fixed oneTRAIN per domain from original200, both25')
  need(plan['status']=='frozen_diagnostic_plan'and plan['mode']=='diagnostic','actual executable diagnostic plan only')
 elif kind=='formal50':
  need(mode=='C1'and plan['status']=='frozen_formal_shard'and plan['mode']=='sample_specific_development_shard','formalC1 only after actual bindings')
  need(plan['shard_id']in range(4)and list(allowed_samples_ordered)==r['partition']['shard_members'][str(plan['shard_id'])]and plan['budget']==dict(sample_count=50,updates_per_sample=500,total_updates=25000),'exact frozen50x500 scope')
  for key in('diagnostic_binding','admission_binding'):
   doc=plan.get(key,{}).get('document',{})
   need(doc.get('execution_mode')=='C1','formalC1 requires actual C1 diagnostic and resource admission, never B1 serial proxy')
 else:raise ValueError('only fixeddiag25/formal50 training roles')
 sc.validate_plan(plan)
 return None

def validate_formal_template(plan,expected_master_descriptor):
 sc,r=_shared(plan,'C1',expected_master_descriptor,[x['sample_id']for x in plan['samples']])
 need(plan['status']==TEMPLATE_STATUS and plan['shard_id']in range(4)and plan['mode']=='sample_specific_development_shard','only nonexecutable formal50 template')
 need(plan['budget']==dict(sample_count=50,updates_per_sample=500,total_updates=25000)and [x['sample_id']for x in plan['samples']]==r['partition']['shard_members'][str(plan['shard_id'])],'exact owned original50')
 need(not any(k in plan for k in('diagnostic_binding','admission_binding'))and plan['gpu_diagnostic_complete']is False,'no invented future diagnostic/admission')
 master=copy.deepcopy(plan);master.update(samples=r['master_samples'],budget=dict(sample_count=200,updates_per_sample=500,total_updates=100000),status='draft_not_executable',mode='development_attack_samples',optimization_on_original_validation=True)
 sc.validate_plan(master,allow_draft=True)
 return None

def activate_formal_plan(template,diagnostic_binding,admission_binding,diagnostic_plan_sha256,expected_master_descriptor):
 # Only call after runtime/root has captured and independently accepted actual
 # current diagnostic+resource originals. This function adds no execution facts.
 validate_formal_template(template,expected_master_descriptor)
 p=copy.deepcopy(template);p.update(status='frozen_formal_shard',gpu_diagnostic_complete=True,diagnostic_binding=copy.deepcopy(diagnostic_binding),admission_binding=copy.deepcopy(admission_binding),diagnostic_plan_sha256=diagnostic_plan_sha256)
 validate_training_plan(p,'formal50','C1',expected_master_descriptor,[x['sample_id']for x in p['samples']])
 return p
