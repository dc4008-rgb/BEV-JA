"""Bounded sequential five-process repeatability batch; fail at first divergence."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os
from pathlib import Path
import shutil, signal, socket, subprocess, sys, time, traceback
sys.dont_write_bytecode = True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def verify_code(code, expected):
    manifest=code/'PACKAGE_MANIFEST.json'
    if sha(manifest)!=expected: raise ValueError('package manifest identity changed')
    entries=json.loads(manifest.read_text())['files']
    paths=[]
    for e in entries:
        rel=Path(e['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe package entry')
        p=code/rel
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:
            raise ValueError('input changed: '+str(rel))
        paths.append(str(rel))
    actual={str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()}
    if len(paths)!=len(set(paths)) or actual != set(paths)|{'PACKAGE_MANIFEST.json'}:
        raise ValueError('package file set changed')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--child',type=Path,required=True)
    ap.add_argument('--expected-hostname',required=True)
    ap.add_argument('--expected-gpu-uuid',required=True)
    ap.add_argument('--expected-driver',required=True)
    ap.add_argument('--package-manifest-sha256',required=True)
    ap.add_argument('--sd35-snapshot',type=Path,required=True)
    args=ap.parse_args()
    base=args.base.resolve(); code=Path(__file__).resolve().parent
    if socket.gethostname()!=args.expected_hostname: raise RuntimeError('wrong hostname')
    if base==code or not base.is_dir(): raise ValueError('base must exist outside code')
    verify_code(code,args.package_manifest_sha256)
    if (base/'BATCH_STATUS.json').exists() or (base/'runs').exists():
        raise FileExistsError('already started: no retry/resume/overwrite')
    if shutil.disk_usage(base).free<10*1024**3: raise RuntimeError('start free space below10GiB')
    (base/'runs').mkdir(); (base/'comparisons').mkdir(); (base/'logs').mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='20260919',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',DIFFUSERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    status={'status':'running','hostname':socket.gethostname(),'expected_gpu_uuid':args.expected_gpu_uuid,
        'driver':args.expected_driver,'package_manifest_sha256':args.package_manifest_sha256,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repeat_count':5,'seed':20260919,'stages':[],'automatic_retry':False,'resume':False,
        'all_training_before_evaluation':True,'deadline_seconds':21600}
    start=time.monotonic(); active=None
    def save(): write(base/'BATCH_STATUS.json',status)
    def stop(signum, frame): raise RuntimeError('batch received signal '+str(signum))
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def gpu_gate():
        q=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True,timeout=20).strip()
        fields=[s.strip() for s in q.split(',')]
        if fields!=[args.expected_gpu_uuid,args.expected_driver]: raise RuntimeError('GPU identity/driver changed: '+q)
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True,timeout=20).strip()
        if used: raise RuntimeError('GPU is occupied: '+used)
    def execute(label, command, limit, gpu=False):
        nonlocal active
        if shutil.disk_usage(base).free<2*1024**3: raise RuntimeError('stage free space below2GiB')
        verify_code(code,args.package_manifest_sha256)
        if gpu: gpu_gate()
        remaining=21600-(time.monotonic()-start)
        if remaining<limit: raise RuntimeError('remaining batch budget cannot cover stage')
        rec={'name':label,'command':command,'timeout_seconds':limit,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        status['stages'].append(rec); status['active_stage']=label; save()
        t=time.monotonic()
        with (base/'logs'/f'{label}.stdout').open('x') as out,(base/'logs'/f'{label}.stderr').open('x') as err:
            active=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            rec['pid']=active.pid; save()
            try: rc=active.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                rec['timed_out']=True
                signal_group(active.pid,signal.SIGTERM)
                try: active.wait(timeout=15)
                except subprocess.TimeoutExpired: signal_group(active.pid,signal.SIGKILL);active.wait(timeout=15)
                rc=124
            rec.update(exit_code=rc,elapsed_seconds=time.monotonic()-t,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            active=None;save()
        if rc: raise RuntimeError(f'{label} failed with exit{rc}; evidence retained')
    train=[]; evaluations=[]
    save()
    try:
        for i in range(1,6):
            root=base/'runs'/f'repeat{i:02d}';root.mkdir()
            path=root/'train';train.append(path)
            command=[sys.executable,'-B',str(code/'training.py'),'--child',str(args.child),'--output',str(path),
                     '--expected-hostname',args.expected_hostname,'--package-manifest-sha256',args.package_manifest_sha256,
                     '--sd35-snapshot',str(args.sd35_snapshot)]
            execute(f'repeat{i:02d}_train',command,1800,gpu=True)
            if i>=2:
                execute(f'train_compare_{i:02d}',[sys.executable,'-B',str(code/'compare.py'),'--runs',*[str(p) for p in train],
                    '--output',str(base/'comparisons'/f'train_through_{i:02d}.json')],600)
        for i,path in enumerate(train,1):
            evaluation=path.parent/'validation';evaluations.append(evaluation)
            command=[sys.executable,'-B',str(code/'evaluate.py'),'--candidate-root',str(args.child),'--run-output',str(path),
                '--output',str(evaluation),'--expected-hostname',args.expected_hostname,
                '--package-manifest-sha256',args.package_manifest_sha256,'--weights-root','/root/.cache/torch/hub/checkpoints']
            execute(f'repeat{i:02d}_validation',command,1800,gpu=True)
            if i>=2:
                execute(f'eval_compare_{i:02d}',[sys.executable,'-B',str(code/'compare.py'),'--runs',*[str(p) for p in train[:i]],
                    '--evaluations',*[str(p) for p in evaluations],'--output',str(base/'comparisons'/f'eval_through_{i:02d}.json')],600)
        final=json.loads((base/'comparisons'/'eval_through_05.json').read_text())
        if final['status']!='five_run_acceptance_pass': raise RuntimeError('five-run acceptance not passed')
        status.update(status='five_run_acceptance_pass',active_stage=None)
    except BaseException as error:
        status.update(status='failed_retained',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        if active is not None and active.poll() is None:
            signal_group(active.pid,signal.SIGTERM)
            try: active.wait(timeout=15)
            except subprocess.TimeoutExpired: signal_group(active.pid,signal.SIGKILL);active.wait(timeout=15)
        raise
    finally:
        status.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                      free_bytes=shutil.disk_usage(base).free)
        save()


if __name__=='__main__': main()
