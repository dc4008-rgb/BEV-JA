"""Exact original-R2 vs joint-depth0 regression, with one explicit identity mapping."""
from __future__ import annotations
import argparse,json,sys,importlib.util
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--r2-source',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.r2_source))
 import torch
 from compare import sha,tree_difference,expected_order,bound_file
 left=json.loads((a.original/'summary.json').read_text());right=json.loads((a.candidate/'summary.json').read_text())
 assert left['identity']=='meeting_full_repeat_20260924_r2'
 assert right['identity']=='r2_joint_depth_20261001' and right['depth_weight']==0 and right['depth_disabled_path']
 assert left['executed_updates']==right['executed_updates']==100
 for k in ['config','training_models','seed','latent_lr','gradient_multiplier','update_clamp','normalizations','loss_temperature','adaptive_history','history_resets_at_outer','budget','query_accounting','training_manifest_sha256','model_manifest_sha256','schedule_sha256','test_data_opened','validation_data_opened']:
  assert tree_difference(left[k],right[k]) is None,'summary mismatch '+k
 lm=json.loads((a.original/'state_evidence/MANIFEST.json').read_text());rm=json.loads((a.candidate/'state_evidence/MANIFEST.json').read_text())
 assert lm['status']==rm['status']=='complete' and lm['executed_updates']==rm['executed_updates']==100
 for manifest in [lm,rm]:assert [(e['phase'],e['global_step'])for e in manifest['entries']]==expected_order()
 counts={'tensors_checked':0,'scalars_checked':0};results=[];first=None
 for le,re in zip(lm['entries'],rm['entries']):
  lp=bound_file(a.original,le['path'],le['sha256']);rp=bound_file(a.candidate,re['path'],re['sha256'])
  l=torch.load(lp,map_location='cpu',weights_only=False);r=torch.load(rp,map_location='cpu',weights_only=False)
  assert l['metadata']['candidate_id']==left['candidate_id'] and r['metadata']['candidate_id']==right['candidate_id']
  # Only this declared wrapper identity may differ; all scientific state stays exact.
  r['metadata']['candidate_id']=l['metadata']['candidate_id']
  diff=tree_difference(l,r,stats=counts)
  results.append({'phase':le['phase'],'step':le['global_step'],'left_sha256':le['sha256'],'right_sha256':re['sha256'],'equal':diff is None})
  if diff:first={'phase':le['phase'],'step':le['global_step'],'difference':diff};break
 assert a.output.parent.is_dir() and not a.output.exists()
 if first:
  a.output.write_text(json.dumps({'status':'failed_retained','first_difference':first,'checks':counts,'states':results},indent=2));raise RuntimeError('scientific state mismatch')
 patch_checks=[]
 for name in ['zero_step_patch.pt','initial_vae_reconstruction.pt','checkpoint_0100.pt','final_patch.pt']:
  l=torch.load(a.original/name,map_location='cpu',weights_only=True);r=torch.load(a.candidate/name,map_location='cpu',weights_only=True)
  if 'metadata' in r and 'candidate_id' in r['metadata']:
   assert r['metadata']['candidate_id']==right['candidate_id'];r['metadata']['candidate_id']=left['candidate_id']
  diff=tree_difference(l,r,stats=counts);assert diff is None,(name,diff)
  patch_checks.append({'name':name,'equal':True,'left_sha256':sha(a.original/name),'right_sha256':sha(a.candidate/name)})
 result={'status':'exact_depth_zero_regression_passed','states_checked':len(results),'tolerance':0,'checks':counts,'identity_exception':'metadata.candidate_id only; expected original and joint-depth0 bound by summaries','states':results,'patches':patch_checks,'original_summary_sha256':sha(a.original/'summary.json'),'candidate_summary_sha256':sha(a.candidate/'summary.json')}
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['states','patches']}))
if __name__=='__main__':main()
