"""All original200 are fresh current EOT1 cases; old40 is selection provenance only."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
POLICY='all_original200_fresh_complete_EOT1_no_retained_case_reuse'
MASTER_SHA='78fc617bcc0534c95288b6f1a91af5eb68707af49d2bcaa9b0a631788dc6a96a'
REFERENCE_SHA='cca50857551d8125dc83a0c77de34fca955471b840e78dfdfea39a5059e3dc33'
def require(value,message):
 if not value:raise ValueError(message)
def reference():
 b=(ROOT/'MASTER200_REFERENCE_001.json').read_bytes();require(hashlib.sha256(b).hexdigest()==REFERENCE_SHA,'exact original200 reference');return json.loads(b)
def validate_original_plan_bindings(plan):
 require(plan.get('case_owner_policy')==POLICY and plan.get('original_plan_bindings')is None,'all200 current-owner policy; no legacy40 owner/reuse')
 r=reference();require(plan['master_samples']==r['master_samples']and plan['master_samples_sha256']==MASTER_SHA,'exact original200 rows');return {}
def resolve_case_owner(plan,sample):
 matches=[s for s in plan['samples']if s['sample_id']==sample['sample_id']];require(matches==[sample],'one exact current scope sample')
 require(plan.get('case_owner_policy')==POLICY,'fresh current EOT1 owner even for original validation40')
 require(sample in reference()['master_samples'],'original200 input unchanged')
 return plan
def case_origin_binding(plan,sample):
 owner=resolve_case_owner(plan,sample)
 return dict(selection_role=sample['selection_role'],training_plan_sha256=owner['_plan_sha256'],source_package_sha256=hashlib.sha256((ROOT/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),all_original200_fresh_current_EOT1=True,prior_training_or_retained40_outputs_reused=False,diagnostic_only_not_counted_in_formal200=plan.get('mode')=='diagnostic')
