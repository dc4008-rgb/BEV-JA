"""Independent v2 frozen-state gradient diagnostic; no production training changes.

This process retains one graph for separate detector/depth/joint VJPs. PyTorch
2.8 AOTAutograd therefore needs donated_buffer=False BEFORE graph construction.
The scoped override is restored on success/failure and never affects the sealed
training runner. Detector forward and gradient replay still require exact bits.
"""
from __future__ import annotations
import os,sys,json,importlib.util,time,argparse,socket
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def run_retained_graph_diagnostic(callback, *, backend_config=None):
 """Scope the AOTAutograd buffer policy to this standalone diagnostic only."""
 if backend_config is None:
  import torch._functorch.config as backend_config
 previous=getattr(backend_config,'donated_buffer',None)
 if type(previous) is not bool:
  raise RuntimeError('diagnostic v2 requires the frozen PyTorch donated_buffer config')
 diagnostic_backend={
  'revision':'probe_joint_v2',
  'scope':'standalone fixed-state gradient diagnostic only; not production training',
  'torch_functorch_donated_buffer_before':previous,
  'torch_functorch_donated_buffer_active':False,
  'reason':'five diagnostic VJPs retain one graph; donated buffers cannot support retain_graph=True',
  'override_applied_before_model_and_graph_construction':True,
  'restore_policy':'config.patch context exit, including exceptions',
  'production_training_modified':False,
  'detector_gradient_acceptance':'original dtype/shape/bytes against frozen fresh R2; tolerance zero',
 }
 with backend_config.patch(donated_buffer=False):
  if backend_config.donated_buffer is not False:
   raise RuntimeError('diagnostic donated_buffer override was not applied')
  return callback(diagnostic_backend)


def _run_diagnostic(diagnostic_backend):
 p=argparse.ArgumentParser();p.add_argument('--r2-source',type=Path,required=True);p.add_argument('--child',type=Path,required=True);p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-hostname',required=True);p.add_argument('--step',type=int,choices=[1,2],default=1);a=p.parse_args()
 assert socket.gethostname()==a.expected_hostname
 assert not a.output.exists()
 sys.path.insert(0,str(a.r2_source));sys.path.insert(0,str(Path(__file__).resolve().parent))
 spec=importlib.util.spec_from_file_location('_probe_original_r2',a.r2_source/'training.py');r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
 r2.verify_tree(a.r2_source,'92af9063dbe07898eb562b0d324e9d1bd746467ce1edea408f9b5f8940f82c85');r2.verify_tree(a.child,r2.CHILD_MANIFEST_SHA)
 r2.configure_determinism()
 import torch
 from evidence import CallLedger,fingerprint
 from adaptive_loss import foa_weights,weighted_loss
 from scale_update import scaled_update
 from components import tensor_statistics
 from compare import tree_difference
 inputs,parent=r2.load_parent(a.child)
 from depth_objective import DepthObjective
 pr=a.child/'parent_v17';manifest=pr/'02_模型/model_manifest.json'
 records=parent.training_records(pr/'01_公共数据集/runtime_v1/train_validation_manifest.csv',None,None)
 schedule=parent.seeded_round_robin(records,100,r2.SEED)
 record=schedule[a.step-1];assert record.split=='train'
 state_manifest=json.loads((a.reference_run/'state_evidence/MANIFEST.json').read_text())
 entry=next(e for e in state_manifest['entries'] if e['phase']=='pre_backward' and e['global_step']==a.step)
 src=a.reference_run/entry['path'];assert r2.sha(src)==entry['sha256']
 state=torch.load(src,map_location='cpu',weights_only=False)['state']
 update_entry=next(e for e in state_manifest['entries'] if e['phase']=='pre_update' and e['global_step']==a.step)
 update_src=a.reference_run/update_entry['path'];assert r2.sha(update_src)==update_entry['sha256']
 update_state=torch.load(update_src,map_location='cpu',weights_only=False)['state']
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 try:
  device=torch.device('cuda:0');dtype=torch.bfloat16
  car,mask=parent.load_car(pr/'01_公共数据集/car_00.jpg',pr/'01_公共数据集/car_00_mask.jpg',device)
  context=parent.build_context(record,car,mask,device,set())
  adapters=parent.build_selected_adapters(('fasterrcnn','ssd300','monodepth2','dpt_large'),device,manifest,None)
  for ad in adapters.values():
   for name in ['model','encoder','decoder']:
    model=getattr(ad,name,None)
    if isinstance(model,torch.nn.Module):model.eval().requires_grad_(False)
  pipeline=parent.build_pipeline(a.snapshot,device,dtype)
  z=state['latent_variable'].to(device).detach().requires_grad_(True)
  ledger=CallLedger({'vae_decode_calls':1,'detector_forwards':8,'depth_clean_forwards':8,'depth_adv_forwards':8,'diagnostic_backward_calls':5})
  patch=ledger.call('vae_decode_calls',parent.decode_patch,pipeline,z)
  raw,det_evidence=r2.compute_losses(parent,patch,context,state['eot'],adapters,ledger)
  previous=state['previous_raw_losses'].to(device) if state['previous_raw_losses'] is not None else None
  weights,details=foa_weights(torch.stack(raw),previous);det=weighted_loss(raw,weights)
  for key,value in [('raw_patch',patch),('raw_losses',torch.stack(raw)),('adaptive_weights',weights),('weighted_total_loss',det),('candidate_outputs',det_evidence)]:
   diff=tree_difference(state[key],value)
   assert diff is None,(key,diff)
  objective=DepthObjective(parent,{k:adapters[k]for k in ('monodepth2','dpt_large')},beta=.1)
  per_model_losses={}
  depth,dep_evidence=objective.call(patch,context,state['eot'],ledger,model_losses_out=per_model_losses)
  gd,=ledger.call('diagnostic_backward_calls',torch.autograd.grad,det,z,retain_graph=True)
  assert tree_difference(update_state['weighted_loss_gradient'],gd) is None,'detector gradient differs from fresh original reference'
  per_model_gradients={}
  for name,loss in per_model_losses.items():
   g,=ledger.call('diagnostic_backward_calls',torch.autograd.grad,loss,z,retain_graph=True)
   assert torch.isfinite(g).all() and torch.count_nonzero(g)>0,name+' invalid or zero latent gradient'
   per_model_gradients[name]=g
  gp,=ledger.call('diagnostic_backward_calls',torch.autograd.grad,depth,z,retain_graph=True)
  gj,=ledger.call('diagnostic_backward_calls',torch.autograd.grad,det+depth,z)
  for name,g in [('detection',gd),('depth',gp),('joint',gj)]:
   assert torch.isfinite(g).all(),name+' nonfinite gradient'
   assert torch.count_nonzero(g)>0,name+' zero gradient'
  zd,td,sd=scaled_update(z,gd,100000.,.05)
  zj,tj,sj=scaled_update(z,gj,100000.,.05)
  def rms(t):return float(t.detach().double().square().mean().sqrt().cpu())
  cos=float(torch.nn.functional.cosine_similarity(gd.detach().double().flatten(),gp.detach().double().flatten(),dim=0).cpu())
  payload={'latent':z.detach().cpu(),'detection_gradient':gd.detach().cpu(),'depth_gradient':gp.detach().cpu(),'joint_gradient':gj.detach().cpu(),'depth_model_gradients':{k:v.detach().cpu()for k,v in per_model_gradients.items()},'det_next_latent':zd.cpu(),'joint_next_latent':zj.cpu(),'raw_detector_losses':torch.stack(raw).detach().cpu(),'detector_weights':weights.cpu()}
  torch.save(payload,a.output/'latent_and_gradients.pt')
  ledger.require_complete()
  result={'status':'gradient_probe_passed','role':'train-only fixed-state numerical diagnostic; not a new training run or effect estimate','sample_id':record.sample_id,'split':record.split,'reference_state_sha256':entry['sha256'],'depth_weight':1.,'depth_beta':.1,'raw_detector_losses':[float(v.detach())for v in raw],'detector_weights':weights.cpu().tolist(),'detection_loss':float(det.detach()),'depth_loss':float(depth.detach()),'latent_rms':rms(z),'detection_gradient_rms':rms(gd),'depth_gradient_rms':rms(gp),'joint_gradient_rms':rms(gj),'depth_to_detection_gradient_rms':rms(gp)/rms(gd),'task_gradient_cosine':cos,'det_update':sd,'joint_update':sj,'depth_evidence':dep_evidence,'ledger':ledger.snapshot(),'payload_sha256':r2.sha(a.output/'latent_and_gradients.pt'),'elapsed_seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'test_data_opened':False,'validation_data_opened':False}
  result.update(step=a.step,dataset=record.dataset,detector_replay_exact=True,reference_pre_update_sha256=update_entry['sha256'],depth_model_gradient_rms={k:rms(v)for k,v in per_model_gradients.items()})
  result.update(diagnostic_backend=diagnostic_backend,diagnostic_script_sha256=r2.sha(Path(__file__)))
  r2.write_json(a.output/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
 except BaseException as e:
  r2.write_json(a.output/'FAILURE.json',{'error':repr(e),'elapsed_seconds':time.monotonic()-start,'test_data_opened':False,'diagnostic_backend':diagnostic_backend,'diagnostic_script_sha256':r2.sha(Path(__file__))});raise
def main():
 return run_retained_graph_diagnostic(_run_diagnostic)


if __name__=='__main__':main()
