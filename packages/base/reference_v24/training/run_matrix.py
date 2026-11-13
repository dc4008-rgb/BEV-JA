#!/usr/bin/env python3
"""Bound two final100 train runs, then held-out validation; default is preflight."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from inputs import ROOT, IDENTITY, candidate_run_path, load_protocol, verify_inputs, sha


def write_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def build_stages(args, session):
    python = sys.executable
    bind = ['--expected-hostname',args.expected_hostname,
            '--package-manifest-sha256',args.package_manifest_sha256]
    mapping = []
    for name in ('source_prefix','source_root','weights_root'):
        value = getattr(args,name,None)
        if value is not None:
            mapping += ['--'+name.replace('_','-'),str(value)]
    stages = []
    for objective in ('U0','U1'):
        output = session/('train_'+objective)
        cmd = [python,'-B',str(ROOT/'training/run_training.py'),'--variant',objective,
               '--mode','train','--output',str(output),'--sd35-snapshot',str(args.sd35_snapshot),
               *bind,*mapping]
        stages.append({'name':'train_'+objective,'kind':'train','output':str(output),'command':cmd})
    for objective in ('U0','U1'):
        output = session/('validation_'+objective)
        cmd = [python,'-B',str(ROOT/'training/evaluate_validation.py'),'--run',
               '--run-output',str(session/('train_'+objective)),'--output',str(output),*bind,*mapping]
        stages.append({'name':'validation_'+objective,'kind':'validation','output':str(output),'command':cmd})
    output = session/'comparison'
    cmd = [python,'-B',str(ROOT/'training/compare_results.py'),
           '--u0-evaluation',str(session/'validation_U0'),'--u1-evaluation',str(session/'validation_U1'),
           '--output',str(output)]
    stages.append({'name':'comparison','kind':'comparison','output':str(output),'command':cmd})
    return stages


def gpu_idle():
    value = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                     '--format=csv,noheader'],text=True).strip()
    if value:
        raise RuntimeError('GPU already has compute processes; no shared execution')


def _require_exact(actual, expected, label):
    # JSON equality preserves int/float/bool distinctions inside nested budgets.
    if json.dumps(actual,sort_keys=True,allow_nan=False) != json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('stage summary differs: ' + label)


def validate_stage_summary(stage, summary, protocol, verified):
    """Exit zero is only a process result; require the declared stage evidence."""
    if not isinstance(summary, dict):
        raise ValueError('stage summary must be an object')
    name, kind = stage['name'], stage['kind']
    if name not in protocol['stage_order']:
        raise ValueError('undeclared stage summary')
    expected_kind = ('comparison' if name == 'comparison' else name.split('_')[0])
    if kind != expected_kind:
        raise ValueError('stage kind differs from fixed stage identity')
    for field, expected in {
        'test_data_opened':False, 'task_variant':protocol['task_variant'],
        'package_manifest_sha256':verified['manifest_sha256'],
        'protocol_sha256':verified['protocol_sha256'],
    }.items():
        _require_exact(summary.get(field), expected, field)
    if kind in ('train','validation'):
        objective = name.rsplit('_',1)[1]
        for field, expected in {
            'update_variant':objective,
            'candidate_id':protocol['candidate_ids'][objective],
        }.items():
            _require_exact(summary.get(field), expected, field)
    if kind == 'train':
        fields = {'identity':IDENTITY, 'status':'train_optimization_complete',
                  'executed_updates':protocol['train_schedule']['updates'],
                  'training_models':protocol['training_models'],
                  'seed':protocol['stateful_sd35']['seed'],
                  'formal_test_result':False, 'validation_data_opened':False,
                  'resume_enabled':False, 'checkpoint_selection':'fixed final100 only',
                  'query_accounting':protocol['budget_per_run']}
        for field, expected in fields.items():
            _require_exact(summary.get(field),expected,field)
        budget = summary.get('budget')
        if not isinstance(budget,dict):
            raise ValueError('training budget evidence missing')
        for field in ('planned','attempted','completed'):
            _require_exact(budget.get(field),protocol['budget_per_run'],'budget.'+field)
    elif kind == 'validation':
        evaluation = protocol['evaluation']
        forwards_each = evaluation['model_forwards_per_run'] // len(evaluation['models'])
        fields = {'status':'development_validation_complete',
                  'updates':protocol['train_schedule']['updates'],
                  'records':evaluation['rows_per_run'],
                  'evaluation_models':evaluation['models'],
                  'evaluation_forward_total':evaluation['model_forwards_per_run'],
                  'evaluation_model_scene_forwards':dict.fromkeys(evaluation['models'],forwards_each),
                  'transfer_models_used':False,'formal_test_allowed':False,
                  'checkpoint_selection':'fixed_final100','automatic_promotion':False}
        for field, expected in fields.items():
            _require_exact(summary.get(field),expected,field)
    else:
        fields = {'identity':IDENTITY,'status':'complete_observations',
                  'comparison':'fixed_U0_U1_final100_validation_descriptive',
                  'formal_test_allowed':False,'candidate_promoted':False,
                  'additional_model_forwards':0,
                  'total_existing_validation_forwards':2*protocol['evaluation']['model_forwards_per_run'],
                  'total_existing_validation_rows':2*protocol['evaluation']['rows_per_run'],
                  'common_control_inputs_exact':True,
                  'common_input_hash_rows_compared':protocol['evaluation']['rows_per_run']}
        for field, expected in fields.items():
            _require_exact(summary.get(field),expected,field)
    return summary['status']


def execute_stage(stage, timeout, env, log_dir):
    started = time.monotonic()
    with (log_dir/(stage['name']+'.stdout.log')).open('w') as out, (log_dir/(stage['name']+'.stderr.log')).open('w') as err:
        process = subprocess.Popen(stage['command'],stdout=out,stderr=err,env=env,cwd=ROOT)
        try:
            code = process.wait(timeout=timeout)
            status = 'process_completed' if code == 0 else 'failed_retained'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            code, status = process.returncode, 'timeout_retained'
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            raise
    return {'name':stage['name'],'kind':stage['kind'],'output':stage['output'],
            'pid':process.pid,'status':status,'exit_code':code,
            'elapsed_seconds':time.monotonic()-started,'timeout_seconds':timeout}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--session',default='psi_clip_v1')
    parser.add_argument('--expected-hostname')
    parser.add_argument('--package-manifest-sha256')
    parser.add_argument('--sd35-snapshot',type=Path)
    parser.add_argument('--source-prefix',type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--weights-root',type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.session or Path(args.session).name != args.session or args.session in ('.','..'):
        raise ValueError('session must be a single nonempty name')
    verified, protocol = verify_inputs(require_sealed=True), load_protocol()
    if not args.run:
        print(json.dumps({'status':'sealed_preflight_pass','inputs':verified,
                         'stage_order':protocol['stage_order'],'real_gpu_executed':False,
                         'source_images_opened':False},ensure_ascii=False,indent=2))
        return
    if not args.expected_hostname or socket.gethostname() != args.expected_hostname:
        raise ValueError('explicit live hostname must match')
    if args.package_manifest_sha256 != verified['manifest_sha256']:
        raise ValueError('explicit input manifest SHA must match')
    if args.sd35_snapshot is None or not args.sd35_snapshot.is_dir():
        raise ValueError('complete existing local SD snapshot required')
    if (args.source_prefix is None) != (args.source_root is None):
        raise ValueError('source mapping requires both prefix and root')
    import torch, torchvision
    if torch.__version__.split('+')[0] != protocol['required_torch_base_version'] or torchvision.__version__.split('+')[0] != protocol['required_torchvision_base_version']:
        raise ValueError('runtime torch/torchvision base version differs')
    gpu_idle()
    if shutil.disk_usage(ROOT).free < protocol['minimum_start_free_disk_bytes']:
        raise RuntimeError('insufficient free space before run')
    session = candidate_run_path(ROOT/'runs'/args.session)
    if session.exists():
        raise FileExistsError('no retry/resume/overwrite of a named session')
    session.mkdir(parents=True)
    logs = session/'logs';logs.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',DIFFUSERS_OFFLINE='1',
               OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    status = {'identity':IDENTITY,'status':'running','hostname':socket.gethostname(),
              'started_utc':datetime.now(timezone.utc).isoformat(),'input_verification':verified,
              'training_before_validation':True,'stages':[],
              'test_images_opened':False,'formal_test_allowed':False,
              'candidate_promoted':False,'automatic_retry':False}
    target = session/'MATRIX_STATUS.json';write_json(target,status)
    stages = build_stages(args,session)
    write_json(session/'STAGE_PLAN.json',stages)
    try:
        for stage in stages:
            gpu_idle()
            if shutil.disk_usage(ROOT).free < protocol['minimum_stage_free_disk_bytes']:
                raise RuntimeError('insufficient space before stage '+stage['name'])
            status['active_stage'] = stage['name'];write_json(target,status)
            result = execute_stage(stage,protocol['stage_timeouts_seconds'][stage['kind']],env,logs)
            status['stages'].append(result);write_json(target,status)
            if result['exit_code'] != 0 or result['status'] != 'process_completed':
                raise RuntimeError('stage failed or timed out: '+stage['name'])
            output = Path(stage['output'])/'summary.json'
            if not output.is_file():
                raise RuntimeError('successful process did not create summary: '+stage['name'])
            summary = json.loads(output.read_text())
            result['scientific_status'] = validate_stage_summary(stage,summary,protocol,verified)
            result['summary_sha256'] = sha(output)
            write_json(target,status)
        status.update(status='complete_observations',active_stage=None,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      attack_improvement_established_by_process_completion=False)
        status['final_input_verification'] = verify_inputs(require_sealed=True)
        write_json(target,status)
    except BaseException as error:
        status.update(status='failed_retained',failure_type=type(error).__name__,failure=str(error),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(target,status)
        raise
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
