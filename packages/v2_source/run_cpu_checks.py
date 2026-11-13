import os,sys,json,subprocess,argparse,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--r2-source',required=True);p.add_argument('--production',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent
env=dict(os.environ,R2_JOINT_R2_SOURCE=a.r2_source,R2_JOINT_DEPTH_PRODUCTION=a.production,PYTHONDONTWRITEBYTECODE='1')
rows=[]
for name,extra in [('test_depth_spatial.py',[]),('test_depth_objective.py',[]),('test_joint_training.py',['--r2-source',a.r2_source])]:
 start=time.monotonic();r=subprocess.run([sys.executable,str(root/name),*extra,'-v'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
 (a.output/(name+'.stdout')).write_bytes(r.stdout);(a.output/(name+'.stderr')).write_bytes(r.stderr)
 rows.append({'test_file':name,'returncode':r.returncode,'seconds':time.monotonic()-start})
 print(r.stderr.decode(),flush=True)
 if r.returncode:break
result={'status':'passed'if len(rows)==3 and all(x['returncode']==0 for x in rows)else'failed','checks':rows,'scope':'synthetic CPU fixtures; RNG restoration also covers visible CUDA generators, not pretrained-model success'}
(a.output/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['status']=='passed'
