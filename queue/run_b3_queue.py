#!/usr/bin/env python3
"""Sequential Ours5 switch-only ablation queue driver (thin, no gate/lease layers).

For each queued ablation it runs the reviewed stage entrypoints unchanged:
  TRAIN part0..3  (train_ours5_ablation_part_v1.py, 50 inputs x 125 updates each)
  NATIVE100_CPU   (verify_train_native200_CPU_v1.py)
  EVAL100         (evaluate_ablation_full8.py, 8 targets x 200 x 4 states)
  CPU_VERIFY      (verify_ablation_full8_cpu.py)
Requests are built the same way as run_no_norm_match_whole200_owned_REMOTE_v1.py.
Before each ablation the free-space / GPU gate is checked; if it fails, or when the
queue is exhausted, the instance is shut down (plain `shutdown`, never released).
A failed stage stops the queue, retains everything and leaves the instance up.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2
PY = '/root/miniconda3/bin/python3.12'
INPUTS = Path('/root/ours5_ablation_first2_20261103_001/inputs')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(MIB), b''):
            h.update(b)
    return h.hexdigest()


def D(p):
    p = Path(p).absolute()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def write(p, value):
    p = Path(p)
    b = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    with p.open('xb') as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    return D(p)


def read(p):
    return json.loads(Path(p).read_bytes())


LOG = None
DIAG = False


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + msg
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, 'a') as f:
            f.write(line + '\n')


def identity():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=30).strip().splitlines()
    need(len(out) == 1, 'exactly one GPU expected')
    uuid, driver, used = [x.strip() for x in out[0].split(',')]
    return dict(hostname=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                gpu_uuid=uuid, driver=driver, gpu_used_MiB=int(used))


def free_bytes(path):
    v = os.statvfs(path)
    return v.f_bavail * v.f_frsize


def session_children(sid):
    """Processes still alive in the child's session after wait (truthful END field)."""
    members = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            fields = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
            if int(fields[3]) == sid:  # field 6 = session id
                members.append(int(pid))
        except Exception:
            continue
    return members


def run_child(ctx, label, entry, request, wall, env_extra=None):
    req = write(ctx.parent / 'inputs' / (label + '_REQUEST.json'), request)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', TORCHINDUCTOR_COMPILE_THREADS='1',
               PYTHONHASHSEED='20260919', PYTHONDONTWRITEBYTECODE='1')
    if env_extra:
        env.update(env_extra)
    argv = [PY, '-B', str(entry), '--request', req['path'], '--request-sha256', req['sha256'], '--execute']
    log(f'[{label}] start: {entry.name} request={req["sha256"][:12]}')
    started = time.time()
    with (ctx / (label + '.stdout')).open('wb') as so, (ctx / (label + '.stderr')).open('wb') as se:
        proc = subprocess.Popen(argv, cwd=str(entry.parent), env=env, stdout=so, stderr=se,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        sid = proc.pid
        try:
            if DIAG and label == 'TRAIN_0':
                pd = Path(request['output_directory'])
                while proc.poll() is None:
                    time.sleep(20)
                    if list(pd.glob('samples/*/CASE_RETURNED.json')):
                        log('[DIAGNOSTIC] first CASE_RETURNED observed; terminating child')
                        proc.terminate()
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise SystemExit('DIAGNOSTIC_FIRST_CASE_OK ' + str(pd))
            code = proc.wait(timeout=wall + 600)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
            log(f'[{label}] killed after wall overrun')
    time.sleep(1)
    members = session_children(sid)
    elapsed = time.time() - started
    end = write(ctx / (label + '_END.json'), dict(stage=label, request=req, actual_wait_return=code,
                elapsed_seconds=elapsed, started_unix=started, entrypoint=D(entry),
                session_group_and_children_after_wait=dict(actual_members=members, root_actual_direct_children=[])))
    log(f'[{label}] exit code {code} after {elapsed/60:.1f} min; leftover session members={members}')
    need(code == 0 and not members, f'{label} failed (exit {code}, leftover {members}); see {ctx / (label + ".stderr")}')
    return req, end


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_ablation(job, tpl, ident, args):
    slug, method = job['slug'], job['method_id']
    ns = job['namespace']
    slugd = slug.replace('_', '-')
    base = Path('/root') / ns
    data_ns = Path('/root/autodl-tmp') / ns
    sci = base / job['science']
    eva = base / job['evaluation']
    need(sci.is_dir() and eva.is_dir(), 'published packages required: ' + str(sci))
    label = slug + '_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + ('_diag' if DIAG else '')
    root, data = base / label, data_ns / label
    ctx = root / 'control'
    for d in (root / 'inputs', ctx, data):
        d.mkdir(parents=True)
    global LOG
    LOG = ctx / 'RUNNER.log'
    log(f'=== ablation {method} switches={json.dumps(job["switch_changes"])} run_root={root} data={data}')
    now = time.time()
    queue_deadline = now + 86400
    scope = dict(schema=f'ours5-{slugd}-authorized-scope-v1', outer_rounds=5, inner_steps=25, sample_count_per_method=100,
                 fresh_initialization=True, started_unix=now, absolute_deadline_unix=now + 129600,
                 actual_user_authority=dict(source='operator instruction 2026-11-05: continue remaining ablations after A1/A2 if disk allows, else shut down; thresholds data>=10GB sys>=2.2GB'),
                 assignments={method: dict(node_id=job['node_id'], hostname=ident['hostname'], boot_id=ident['boot_id'], part_ids=[0, 1])})
    scope_d = write(root / 'inputs/ACTUAL_TASK_SCOPE.json', scope)
    tc = dict(tpl['training_common'], schema=f'ours5-{slugd}-original50-training-request-v1',
              identity=f'ours5-{slugd}-existing-switch-ablation-v1',
              source_files={k: D(sci / n) for k, n in {'probe': 'train_ours5_ablation_part_v1.py', 'formal_contract': 'ablation_contract_v1.py',
                                                       'connectivity_library': 'budget_gpu_component_v1.py', 'kernel': 'sd35_conditional_two_clip_v1.py',
                                                       'derived_engine': 'derived_budget_training_engine_v2.py'}.items()})
    for k in ('protocol', 'master_reference'):
        need(D(tc[k]['path']) == tc[k], 'frozen input differs: ' + k)
    for p in tpl['part_plans']:
        need(D(p['path']) == p, 'frozen part plan differs: ' + p['path'])
    owner = dict(schema=f'ours5-{slugd}-fresh200-owned-request-v1', method_id=method, node_id=job['node_id'], run_root=str(root), data_root=str(data),
                 deadline_unix=queue_deadline, gate_request=dict(actual_current_identity_pin=dict(hostname=ident['hostname'], boot_id=ident['boot_id'])),
                 training_common=tc, part_plans=tpl['part_plans'], task_scope=scope_d, pidfd_bridge=tpl['pidfd_bridge'], python_path=PY,
                 science=D(sci / 'PACKAGE_MANIFEST.json'), evaluation=D(eva / 'PACKAGE_MANIFEST.json'), identity=ident,
                 runner='ablation_thin_runner_v1 thin sequential driver', switch_changes=job['switch_changes'])
    rd = write(root / 'inputs/RECEIVED_QUEUE_REQUEST.json', owner)
    sys.path.insert(0, str(sci))
    science = load_module('_ablation_contract_' + slug, sci / 'ablation_contract_v1.py')
    sys.path.pop(0)
    storage_profile = dict(namespace=str(data), maximum_allocated_bytes=args.data_cap_gib * GIB, reserve_bytes=GIB, maximum_entries=100000)
    completed = []
    for part in job.get('parts', [0, 1]):
        now = time.time()
        deadline = min(now + 25200, queue_deadline - 600)
        need(deadline - now >= 18000, 'remaining TRAIN window')
        pd = data / f'part_{part}'
        pd.mkdir()
        breq = dict(tc, method_id=method, node_id=job['node_id'], part_id=part, expected_hostname=ident['hostname'], expected_boot_id=ident['boot_id'],
                    deadline_unix=deadline, reference_plan=tpl['part_plans'][part], task_scope=scope_d, output_base=str(pd),
                    output_directory=str(pd / 'formal200_part_training'), storage_profile=storage_profile)
        bd = write(root / 'inputs' / f'TRAIN_{part}_BASE.json', breq)
        gi = identity()
        ready = dict(schema='cg01-budget200-root-ready-proof-v1', identity='bevja_cg01_formal200_budget135_v1',
                     actual_identity=dict(hostname=ident['hostname'], boot_id=ident['boot_id']), deadline_unix=deadline,
                     requested_stage=breq['stage'], part_id=part, budget_outer_rounds=5, task_scope=scope_d, node_id=job['node_id'],
                     ablation_method_id=method, base_request_without_admission=bd, protocol=tc['protocol'], master_reference=tc['master_reference'],
                     original_input_plan=breq['reference_plan'],
                     selected_GPU_query=dict(UUID=gi['gpu_uuid'], driver_version=gi['driver'], memory_used_MiB=gi['gpu_used_MiB'],
                                             selection_mode='single_visible_gpu_nvidia_smi_query',
                                             observed_CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                             observed_NVIDIA_VISIBLE_DEVICES=os.environ.get('NVIDIA_VISIBLE_DEVICES')),
                     storage_admission=storage_profile, actual_initial_strict_gate=dict(runner='thin_v1', data_free_bytes=free_bytes(data), sys_free_bytes=free_bytes('/')))
        ready_d = write(ctx / f'TRAIN_{part}_ACTUAL_READY.json', ready)
        actual = dict(breq, operator_admission=ready_d)
        science.validate_task_scope(actual)
        science.load_inputs(actual)
        req_d, end_d = run_child(ctx, f'TRAIN_{part}', sci / 'train_ours5_ablation_part_v1.py', actual, 25200)
        result_d = D(pd / 'formal200_part_training/ABLATION_PART_TRAINING_RETURNED.json')
        value = read(result_d['path'])
        need(value['request'] == req_d, 'part return provenance')
        science.validate_result(actual, value)
        completed.append(dict(part_id=part, request=req_d, **{'return': result_d}, owned_END=end_d))
        write(ctx / f'TRAIN_{part}_ACCEPTED_PREFIX.json', dict(schema='ours5-ablation-owner-true-part-prefix-v1', method_id=method,
              completed_parts=completed, effective_cases=50 * len(completed), actual_new_updates=6250 * len(completed)))
        log(f'TRAIN part {part} accepted ({50*len(completed)}/100 inputs)')
    collection = science.build_training_collection([x['request'] for x in completed], [x['return'] for x in completed])
    coll_d = write(root / 'inputs/TRAINING_COLLECTION_200.json', collection)
    comp_d = write(ctx / 'ACTUAL_TRAIN200_COMPLETED.json', dict(schema=f'ours5-{slugd}-actual-TRAIN200-completed-v1', method_id=method,
                   training_collection=coll_d, actual_completed_parts=completed))
    now = time.time()
    nd = min(now + 3600, queue_deadline - 600)
    nq = dict(schema=f'ours5-{slugd}-fresh200-independent-TRAIN-native-CPU-request-v1', original_owner_request=rd, training_collection=coll_d,
              completed_train_parts=comp_d, original_prefix_verifier=D(sci / 'original_completed_prefix_native.py'),
              patch_contract_source=tpl['native_helpers']['patch_contract_source'], retention_source=tpl['native_helpers']['retention_source'],
              output=str(data / 'TRAIN100_NATIVE_CPU'),
              admission=dict(expected_hostname=ident['hostname'], expected_boot_id=ident['boot_id'], deadline_unix=nd, wall_seconds=nd - now, max_RSS_bytes=4 * GIB, disk_reserve_bytes=GIB))
    run_child(ctx, 'NATIVE100_CPU', sci / 'verify_train_native200_CPU_v1.py', nq, 3600)
    nv = read(data / 'TRAIN100_NATIVE_CPU/TRAIN100_NATIVE_CPU_ACCEPTED.json')
    need(nv['accepted'] is True and nv['method_id'] == method and nv['completed_cases'] == 100 and nv['completed_updates'] == 12500 and nv['decoded_zero_final_patch_count'] == 200, 'native200 accepted')
    log('NATIVE200 accepted')
    now = time.time()
    ed = min(now + 14400, queue_deadline - 600)
    deps = tpl['evaluation_dependencies']
    for d in deps.values():
        need(D(d['path']) == d, 'dependency source differs: ' + d['path'])
    eq = dict(tpl['evaluation_common'], method_id=method, training_collection=coll_d,
              dependency_initialization=dict(owner_request=rd, control_root=str(ctx / 'EVAL100_DEPENDENCY_BOOTSTRAP'),
                                             expected_font_source=deps['font_source'], expected_fc_list=deps['fc_list']),
              output=str(data / 'evaluation_100'),
              schema='ours5-b3-nuscenes100-evaluation-request-v1',
              admission=dict(expected_hostname=ident['hostname'], expected_boot_id=ident['boot_id'], expected_GPU_UUID=ident['gpu_uuid'],
                             deadline_unix=ed, wall_seconds=14400, max_RSS_bytes=8 * GIB, disk_reserve_bytes=GIB))
    eqd, _ = run_child(ctx, 'EVAL100', eva / 'evaluate_ablation_full8.py', eq, 14400)
    evd = D(data / 'evaluation_100/ABLATION_FULL200_ALL8_EVALUATION_RETURNED.json')
    ev = read(evd['path'])
    need(ev['sample_count'] == 100 and ev['model_forward_count'] == 3200 and ev['method_id'] == method, 'EVAL6400 return')
    log('EVAL100 return verified: 3200 forwards')
    sys.path.insert(0, str(eva))
    bridge = load_module('_bridge_' + slug, eva / 'ablation_evaluation_bridge.py')
    sys.path.pop(0)
    cq = bridge.make_cpu_request(eqd, evd, data / 'CPU_VERIFICATION', time.time() + 3600, 8 * GIB, GIB)
    run_child(ctx, 'CPU_VERIFY', eva / 'verify_ablation_full8_cpu.py', cq, 3600)
    cvd = D(data / 'CPU_VERIFICATION/ABLATION_FULL200_CPU_VERIFIED.json')
    cv = read(cvd['path'])
    need(cv['accepted'] is True and cv['method_id'] == method and cv['sample_count'] == 100, 'CPU verified')
    summary = dict(schema='ours5-ablation-thin-runner-returned-v1', status='TRAIN200_NATIVE_FULL8_CPU_COMPLETE', method_id=method, label=job.get('label'),
                   switch_changes=job['switch_changes'], node_id=job['node_id'], identity=ident, owner_request=rd, training_collection=coll_d,
                   native_acceptance=D(data / 'TRAIN100_NATIVE_CPU/TRAIN100_NATIVE_CPU_ACCEPTED.json'), evaluation_return=evd, evaluation_CPU_return=cvd,
                   detection_summary=D(data / 'evaluation_100/detection_summary.json'), depth_summary=D(data / 'evaluation_100/depth_summary.json'),
                   sample_count=100, updates_per_sample=125, executed_updates=12500, model_evaluation_forwards=3200, run_root=str(root), data_root=str(data))
    out = write(ctx / 'QUEUE_RETURNED.json', summary)
    log('ABLATION COMPLETE -> ' + out['path'])
    return out


def gate_ok(args, state_log):
    ident = identity()
    df, sf = free_bytes('/root/autodl-tmp'), free_bytes('/')
    busy = [l for l in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
            if ('train_ours5' in l or 'evaluate_ablation' in l or 'verify_ablation' in l or 'verify_train_native' in l) and 'ps -eo' not in l]
    ok = df >= args.min_data_free_gb * 10 ** 9 and sf >= args.min_sys_free_gb * 10 ** 9 and ident['gpu_used_MiB'] < 1024 and not busy
    msg = f'gate: data_free={df/1e9:.1f}GB (need {args.min_data_free_gb}) sys_free={sf/1e9:.2f}GB (need {args.min_sys_free_gb}) gpu_used={ident["gpu_used_MiB"]}MiB busy={len(busy)} -> {"OK" if ok else "STOP"}'
    with open(state_log, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')
    print(msg, flush=True)
    return ok, msg


def shutdown(reason, state_log):
    with open(state_log, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + 'SHUTDOWN: ' + reason + '\n')
    print('SHUTDOWN: ' + reason, flush=True)
    subprocess.run(['sync'])
    subprocess.Popen(['bash', '-c', 'sleep 5; shutdown -h now'], start_new_session=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queue', required=True, type=Path, help='JSON list of jobs (slug, method_id, namespace, node_id, science, evaluation, switch_changes, label)')
    ap.add_argument('--template', required=True, type=Path, help='TEMPLATE_INPUTS.json extracted from the accepted no_norm_match request')
    ap.add_argument('--state-dir', required=True, type=Path)
    ap.add_argument('--min-data-free-gb', type=float, default=11.9, help='must cover data-cap + 1GiB reserve required by storage_guard at part 0')
    ap.add_argument('--min-sys-free-gb', type=float, default=2.2)
    ap.add_argument('--data-cap-gib', type=int, default=10, help='storage_guard maximum_allocated_bytes per run (GiB); +1GiB reserve must be free')
    ap.add_argument('--no-shutdown', action='store_true')
    ap.add_argument('--diagnostic-first-case', action='store_true', help='skip the GPU gate, run TRAIN_0 until the first CASE_RETURNED.json, then stop (no shutdown)')
    args = ap.parse_args()
    global DIAG
    DIAG = args.diagnostic_first_case
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_log = args.state_dir / 'QUEUE_STATE.log'
    tpl = read(args.template)
    science_root = tpl['training_common']['science_package']['root']
    if science_root not in sys.path:
        sys.path.insert(0, science_root)
    ident = identity()
    done_file = args.state_dir / 'DONE_SLUGS.json'
    done = read(done_file) if done_file.exists() else []
    with open(state_log, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + f'queue start identity={json.dumps(ident)} done={done}\n')
    results = []
    while True:
        jobs = read(args.queue)  # re-read every iteration so jobs can be appended while running
        pending = [j for j in jobs if j['slug'] not in done]
        if not pending:
            break
        job = pending[0]
        ok, msg = gate_ok(args, state_log)
        if DIAG:
            ok = True
        if not ok:
            results.append(dict(job=job, status='NOT_STARTED_GATE', gate=msg))
            write(args.state_dir / ('QUEUE_RESULTS_%s.json' % time.strftime('%Y%m%d_%H%M%S')), results)
            if not args.no_shutdown:
                shutdown('gate failed before job %s: %s' % (job['slug'], msg), state_log)
            return
        try:
            out = run_ablation(job, tpl, ident, args)
            results.append(dict(job=job, status='COMPLETE', queue_returned=out))
            done.append(job['slug'])
            done_file.write_text(json.dumps(done) + '\n')
        except SystemExit as e:
            with open(state_log, 'a') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + str(e) + '\n')
            print(str(e), flush=True)
            return
        except BaseException as e:
            tb = traceback.format_exc()
            results.append(dict(job=job, status='FAILED', error=str(e), traceback=tb))
            with open(state_log, 'a') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + f'FAILED job {job["slug"]}: {e}\n')
            write(args.state_dir / ('QUEUE_RESULTS_%s.json' % time.strftime('%Y%m%d_%H%M%S')), results)
            print(tb, flush=True)
            return  # keep instance up for inspection; nothing deleted
    write(args.state_dir / ('QUEUE_RESULTS_%s.json' % time.strftime('%Y%m%d_%H%M%S')), results)
    if not args.no_shutdown:
        shutdown('queue complete (%d jobs)' % len(jobs), state_log)


if __name__ == '__main__':
    main()
