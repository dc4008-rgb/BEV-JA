"""Audit the fresh b3-full-nuscenes100 TRAIN200 once, using the original per-case loop.

Four exact original50 groups, 200 newly audited cases and400 zero/final native
patch decodes. No historical accepted case, model or optimizer is replayed.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import time
import types

PREFIX_PIN=(17612,'df1b61212d0e2c6a2b3bd13bb5dba61b7370628f88b8f7481354250511abfe16')
PATCH_SHA='565066784eaac77f422c7943185221492263cfe590ba51b5642dc9b653c225a5'
RETENTION_SHA='3aec61b45b4a27a06edf09733bb6fefbd36434bcb7071189e7fb46e44a9cb17f'
SCHEMA='ours5-b3-full-nuscenes100-fresh200-independent-TRAIN-native-CPU-request-v1'
FIELDS={'schema','original_owner_request','training_collection','completed_train_parts','original_prefix_verifier','patch_contract_source','retention_source','output','admission'}
def need(ok,message):
    if not ok:raise ValueError(message)

def original(binding):
    need(set(binding)=={'path','bytes','sha256'} and (binding['bytes'],binding['sha256'])==PREFIX_PIN,'exact original prefix verifier source')
    p=Path(binding['path']);need(p.is_absolute() and not any(x.is_symlink()for x in(p,*p.parents)),'literal original verifier path')
    b=p.read_bytes();need((len(b),hashlib.sha256(b).hexdigest())==PREFIX_PIN,'exact original verifier whole')
    m=types.ModuleType('_unchanged_native_prefix_helpers');m.__file__=str(p);exec(compile(b,str(p),'exec'),m.__dict__)
    return m,b.decode()

def group_function(O,source):
    """Literal original loop, original retention helper and original bounds."""
    retained=source[source.index('    def retained(binding):'):source.index('\n    try:',source.index('    def retained(binding):'))]
    loop=source[source.index('        for row in result["rows"]:'):source.index('        need(len(cases) == count')]
    prefix='def audit_group(result, training_root, output, guard, retain, decoder, torch):\n    count=len(result["rows"])\n    need(0<count<=50 and not output.exists(),"one finite suffix group")\n    output.mkdir()\n    catalogue={};cases=[]\n    artifact_count=tensor_count=decoded_count=artifact_bytes=0\n'
    suffix='''        need(len(cases)==count and decoded_count==count*2,"all exact suffix patches decode")
        binding=save(output/"SUFFIX_GROUP_SERVER_RETENTION_CATALOGUE.json",dict(schema="ours5-ablation-training-native-suffix-group-retention-v1",files=list(catalogue.values()),cases=cases,artifact_file_count=artifact_count,tensor_file_count=tensor_count,artifact_file_bytes=artifact_bytes,decoded_zero_final_patch_count=decoded_count))
        return dict(sample_ids=[x["sample_id"]for x in cases],completed_cases=count,completed_updates=count*125,artifact_file_count=artifact_count,tensor_file_count=tensor_count,artifact_file_bytes=artifact_bytes,decoded_zero_final_patch_count=decoded_count,all_retained_file_count=len(catalogue),catalogue=binding)
    except BaseException:
        raise
'''
    text=prefix+retained+'\n    try:\n'+loop+suffix
    namespace=dict(O.__dict__);exec(compile(text,'<literal_original_native_per_case_loop>','exec'),namespace)
    return namespace['audit_group']

def verify(q,request_descriptor):
    need(set(q)==FIELDS and q['schema']==SCHEMA,'closed fresh200 native request')
    O,source=original(q['original_prefix_verifier']);read=O.read;plain=O.plain
    a=q['admission'];need(set(a)=={'expected_hostname','expected_boot_id','deadline_unix','wall_seconds','max_RSS_bytes','disk_reserve_bytes'},'closed native CPU admission')
    need(socket.gethostname()==a['expected_hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==a['expected_boot_id'],'actual assigned hostboot')
    for k in('deadline_unix','wall_seconds','max_RSS_bytes','disk_reserve_bytes'):
        need(type(a[k])in(int,float)and math.isfinite(a[k])and a[k]>0,'finite positive CPU bounds')
    remaining=min(a['wall_seconds'],a['deadline_unix']-time.time())
    need(0<remaining<=3600 and a['max_RSS_bytes']<=4*1024**3 and a['disk_reserve_bytes']>=1024**3,'original native CPU finite profile')
    output=plain(q['output']);need(not output.exists()and output.parent.is_dir(),'fresh exclusive native output')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('fresh native200 deadline')));signal.setitimer(signal.ITIMER_REAL,remaining)
    def guard():
        need(time.time()<a['deadline_unix'],'native deadline')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=a['max_RSS_bytes'],'native RSS')
        need(shutil.disk_usage(output.parent).free>=a['disk_reserve_bytes'],'native disk reserve')
    guard();owner=read(q['original_owner_request']);collection=read(q['training_collection']);complete=read(q['completed_train_parts'])
    need(owner['schema']=='ours5-b3-full-nuscenes100-fresh200-owned-request-v1'and owner['method_id']=='ours5_b3_full_nuscenes100','only newly authorized fresh b3-full-nuscenes100 task')
    pin=owner['gate_request']['actual_current_identity_pin'];need((pin['hostname'],pin['boot_id'])==(a['expected_hostname'],a['expected_boot_id'])and a['deadline_unix']<=owner['deadline_unix'],'original newowner identity/clock')
    need(Path(q['original_owner_request']['path'])==Path(owner['run_root'])/'inputs/RECEIVED_QUEUE_REQUEST.json'and Path(q['training_collection']['path'])==Path(owner['run_root'])/'inputs/TRAINING_COLLECTION_200.json','literal own task producer paths')
    need(complete['schema']=='ours5-b3-full-nuscenes100-actual-TRAIN200-completed-v1'and complete['method_id']==owner['method_id']and complete['training_collection']==q['training_collection'],'actual four completed parts evidence')
    completed=complete['actual_completed_parts'];need(len(completed)==2 and [x['part_id']for x in completed]==[0,1],'two real nuScenes50 parts')
    need(collection['schema']=='ours5-b3-nuscenes100-training-collection-v1'and collection['method_id']==owner['method_id']and collection['ablation_variant']=='custom'and collection['sample_count']==100 and collection['updates_per_sample']==125 and collection['effective_training_updates']==12500,'all200 fresh125 collection')
    need(collection['request_bindings']==[x['request']for x in completed]and collection['result_bindings']==[x['return']for x in completed],'all actual request/result producers')
    master=read(collection['master_reference']);need(collection['master_reference']==owner['training_common']['master_reference']and [r['sample']for r in collection['rows']]==[s for s in master['master_samples'] if s['dataset']=='nuscenes'],'same original ordered200')
    by_id={r['sample']['sample_id']:r for r in collection['rows']};need(len(by_id)==100,'unique cohort')
    groups=[]
    for part in range(2):
        pq=read(collection['request_bindings'][part]);result=read(collection['result_bindings'][part]);plan=read(owner['part_plans'][part]);end=read(completed[part]['owned_END'])
        need(pq['part_id']==part and pq['method_id']==owner['method_id']and pq['task_scope']==owner['task_scope']and result['request']==collection['request_bindings'][part]and result['effective_case_count']==50 and result['effective_updates']==6250 and result['new_case_count']==50 and result['reused_case_count']==0,'each part all50 fresh')
        need(len(result['rows'])==50 and [r['sample']for r in result['rows']]==plan['samples']and all(by_id[r['sample']['sample_id']]==r for r in result['rows']),'exact original50 membership')
        need(end['actual_wait_return']==0 and end['session_group_and_children_after_wait']['actual_members']==[]and end['session_group_and_children_after_wait']['root_actual_direct_children']==[],'actual successful full owned TRAIN wait')
        training_root=plain(pq['output_directory']);need(not output.is_relative_to(training_root)and not training_root.is_relative_to(output),'separate native output')
        groups.append((part,training_root,result))
    retain=O.original_functions(q['retention_source'],RETENTION_SHA,('retain_opaque',),{'plain':plain,'need':need,'hashlib':hashlib,'stat':O.stat,'os':os})['retain_opaque']
    import torch
    need(not torch.cuda.is_initialized(),'no CUDA/model initialization by native audit')
    decoder=O.original_functions(q['patch_contract_source'],PATCH_SHA,('_tensor_digest','_validate_rgb','load_patch'),{'torch':torch,'Path':Path,'hashlib':hashlib,'json':json},('CANONICAL_HW','PATCH_SCHEMA','RENDERER_CONTRACT'))
    audit_group=group_function(O,source);output.mkdir();results=[]
    for part,training_root,result in groups:
        guard();results.append(dict(part_id=part,**audit_group(result,training_root,output/('part_'+str(part)),guard,retain,decoder,torch)))
    need(sum(x['completed_cases']for x in results)==100 and sum(x['decoded_zero_final_patch_count']for x in results)==200,'all200 once native custody accepted')
    returned=dict(schema='ours5-b3-full-nuscenes100-fresh200-independent-TRAIN-native-CPU-accepted-v1',accepted=True,method_id='ours5_b3_full_nuscenes100',actual_request=request_descriptor,training_collection=q['training_collection'],completed_cases=100,completed_updates=12500,reused_native_cases=0,newly_audited_native_cases=100,decoded_zero_final_patch_count=200,groups=results,model_forwards=0,optimizer_updates=0)
    d=O.save(output/'TRAIN100_NATIVE_CPU_ACCEPTED.json',returned);signal.setitimer(signal.ITIMER_REAL,0);return d

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    path=Path(a.request);b=path.read_bytes();need(0<len(b)<=2*1024**2 and hashlib.sha256(b).hexdigest()==a.request_sha256,'actual bounded native request SHA')
    if a.execute:print(json.dumps(verify(json.loads(b),dict(path=str(path),bytes=len(b),sha256=a.request_sha256)),sort_keys=True))
