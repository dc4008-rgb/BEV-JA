#!/usr/bin/env python3
"""Pre-flight check for one B3 node: GPU, disk, template inputs, packages."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

GIB = 1024 ** 3


def desc_ok(d):
    p = Path(d['path'])
    if not p.is_file():
        return False, f'missing {p}'
    b = p.read_bytes()
    if len(b) != d['bytes'] or hashlib.sha256(b).hexdigest() != d['sha256']:
        return False, f'bytes/sha mismatch {p}'
    return True, str(p)


def walk(obj, out, key=''):
    if isinstance(obj, dict):
        if {'path', 'bytes', 'sha256'} <= set(obj):
            ok, msg = desc_ok(obj); out.append((key, ok, msg)); return
        for k, v in obj.items():
            walk(v, out, f'{key}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, out, f'{key}/{i}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--node-id', required=True)
    ap.add_argument('--template', default='/root/b3_queue_20261107/TEMPLATE_INPUTS_B3_SERVER.json')
    ap.add_argument('--manifest', default='/root/b3_queue_20261107/B3_JOB_MANIFEST_3NODE.json')
    ap.add_argument('--packages-root', default='/root')
    ap.add_argument('--out', default='/root/b3_queue_20261107/NODE_READY.json')
    args = ap.parse_args()
    checks = []
    t = json.loads(Path(args.template).read_text())
    jobs = [j for j in json.loads(Path(args.manifest).read_text()) if j['node_id'] == args.node_id]
    if not jobs:
        raise SystemExit(f'no jobs for {args.node_id}')
    out = []
    walk(t, out)
    for name, ok, msg in out:
        checks.append(dict(name=name, ok=ok, detail=msg))
    for j in jobs:
        pkg = Path(args.packages_root) / j['namespace']
        sci = pkg / j['science']; eva = pkg / j['evaluation']
        for p in (sci / 'ablation_contract_v1.py', sci / 'b3_overrides.py',
                  sci / 'PACKAGE_MANIFEST.json', eva / 'eight_target_library.py'):
            checks.append(dict(name=f'package/{j["slug"]}/{p.name}', ok=p.is_file(), detail=str(p)))
        c = (sci / 'ablation_contract_v1.py')
        if c.is_file():
            checks.append(dict(name=f'node_id/{j["slug"]}', ok=(args.node_id in c.read_text()), detail=args.node_id))
    roots = [('science_root', t['training_common'].get('science_package', {}).get('root')),
             ('weights_root', t['training_common'].get('weights_root')),
             ('eval_weights_root', t['evaluation_common'].get('weights_root'))]
    for k, v in t['training_common'].get('helper_roots', {}).items():
        roots.append((f'helper/{k}', v))
    for name, path in roots:
        if path:
            checks.append(dict(name=name, ok=Path(path).is_dir(), detail=str(path)))
    # GPU
    try:
        s = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                                     '--format=csv,noheader'], text=True).strip().splitlines()[0]
        name, total, used = [x.strip() for x in s.split(',')]
        total_mib = int(total.split()[0]); used_mib = int(used.split()[0])
        gpu_ok = total_mib >= 20000 and used_mib < 1024
        checks.append(dict(name='gpu', ok=gpu_ok, detail=f'{name} total={total_mib}MiB used={used_mib}MiB'))
    except Exception as e:
        checks.append(dict(name='gpu', ok=False, detail=f'nvidia-smi failed: {e}'))
    for label, path, need in (('data_disk', '/root/autodl-tmp', 20 * GIB), ('system_disk', '/', int(2.2 * GIB))):
        try:
            free = shutil.disk_usage(path).free
            checks.append(dict(name=label, ok=free >= need, detail=f'free={free / GIB:.2f}GiB need={need / GIB:.2f}GiB'))
        except Exception as e:
            checks.append(dict(name=label, ok=False, detail=str(e)))
    # torch / diffusers
    try:
        import torch
        checks.append(dict(name='torch', ok=torch.cuda.is_available(), detail=f'{torch.__version__} cuda={torch.cuda.is_available()}'))
    except Exception as e:
        checks.append(dict(name='torch', ok=False, detail=str(e)))
    try:
        import diffusers
        checks.append(dict(name='diffusers', ok=True, detail=diffusers.__version__))
    except Exception as e:
        checks.append(dict(name='diffusers', ok=False, detail=str(e)))
    # protocol pin
    proto = t['training_common']['protocol']
    checks.append(dict(name='protocol_path', ok=Path(proto['path']).is_file(), detail=proto['path']))
    ready = all(c['ok'] for c in checks)
    result = dict(node_id=args.node_id, ready=ready, checks=checks)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for c in checks:
        print(('OK  ' if c['ok'] else 'FAIL') + ' ' + c['name'] + ' :: ' + c['detail'])
    print('NODE_READY', args.node_id, ready)
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()
