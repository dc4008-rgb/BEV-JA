"""Read only authenticated v23 TRAIN states; no model execution or trajectory replay."""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import sys

import torch

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / '.workspace_root').is_file())
SOURCE = HERE.parent / 'Contextual双模型损失结果_20260921/downloaded'
DELIVERY_SHA = 'eec0313b41cd33128e6ee0bc8ce45cd4094366766645351b85519139fde16b17'
IDENTITY = 'normalized-v23-contextual-ours-20260921'
RUNS = SOURCE / IDENTITY / 'runs/contextual_ours_v1'
MODELS = ('fasterrcnn', 'ssd300')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(x):
    x = x.detach().double().flatten()
    require(x.numel() > 0 and bool(torch.isfinite(x).all()), 'invalid numeric input')
    a = x.abs()
    q = torch.quantile(a, torch.tensor([.5, .9, .95, .99, .999], dtype=torch.float64))
    return dict(numel=x.numel(), nonzero_count=int(torch.count_nonzero(x)),
                rms=float(x.square().mean().sqrt()), abs_max=float(a.max()),
                abs_p50=float(q[0]), abs_p90=float(q[1]), abs_p95=float(q[2]),
                abs_p99=float(q[3]), abs_p999=float(q[4]))


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    n = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / n) if n > 0 else None


def main():
    torch.set_num_threads(2)
    out = HERE / 'analysis'
    require(not out.exists(), 'refuse overwriting analysis; use a new report for any revision')
    plan = json.loads((HERE / 'ANALYSIS_PLAN.json').read_text())
    require(plan['quantile'] == .99 and plan['fixed_objective_for_next_comparison'] == 'C3', 'plan differs')
    require(sha(SOURCE / 'DELIVERY_MANIFEST.json') == DELIVERY_SHA, 'delivery manifest differs')
    delivery = json.loads((SOURCE / 'DELIVERY_MANIFEST.json').read_text())
    bindings = {e['path']: e for e in delivery['files']}
    read_sources = {}

    def verified(path):
        path = Path(path)
        rel = str(path.relative_to(SOURCE))
        require('/train_C0/' in rel or '/train_C3/' in rel, 'only named train outputs may be read')
        require(not path.is_symlink(), 'symlink evidence forbidden')
        entry = bindings[rel]
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'evidence differs: ' + rel)
        read_sources[rel] = entry
        return path

    all_states = {}
    source_meta = {}
    for variant in ('C3', 'C0'):
        run = RUNS / ('train_' + variant)
        summary = json.loads(verified(run / 'summary.json').read_text())
        schedule = json.loads(verified(run / 'schedule.json').read_text())
        require(summary['executed_updates'] == 100 and summary['attack_objective'] == variant
                and summary['test_data_opened'] is False and summary['validation_data_opened'] is False,
                'incorrect training identity')
        require(len(schedule) == 100 and all(s['split'] == 'train' for s in schedule), 'non-train schedule')
        manifest = json.loads(verified(run / 'state_evidence/MANIFEST.json').read_text())
        require(manifest['status'] == 'complete' and manifest['executed_updates'] == 100, 'incomplete states')
        require(len(manifest['entries']) == 409, 'incomplete full state manifest')
        # Outer boundaries deliberately share global_step=0; only indexed
        # per-update phases are unique by (step, phase).
        selected = [e for e in manifest['entries'] if e['phase'] in ('gradients', 'pre_update', 'post_update')]
        entries = {(e['global_step'], e['phase']): e for e in selected}
        require(len(selected) == len(entries) == 300, 'per-update state identity collision')
        rows = []
        for step in range(1, 101):
            state = {}
            for phase in ('gradients', 'pre_update', 'post_update'):
                e = entries[step, phase]
                path = verified(run / e['path'])
                require(sha(path) == e['sha256'], 'state manifest differs')
                payload = torch.load(path, map_location='cpu', weights_only=True)
                meta = payload['metadata']
                require(meta['attack_objective'] == variant and meta['global_step'] == step
                        and meta['phase'] == phase and meta['candidate_id'] == summary['candidate_id'], 'state metadata differs')
                state[phase] = payload['state']
            pre, post, grads = state['pre_update'], state['post_update'], state['gradients']
            z, joint, cast = pre['latent_variable'], pre['joint_gradient'], pre['joint_gradient_cast_for_update']
            require(z.dtype == torch.bfloat16 and tuple(z.shape) == (1, 16, 64, 64), 'unexpected latent layout')
            require(joint.dtype == torch.float32 and pre['latent_lr'] == .05, 'update contract differs')
            require(torch.equal(joint.to(z), cast), 'saved cast differs')
            require(torch.equal(z, grads['latent_variable']), 'gradient/pre state differs')
            require(pre['gradient_state']['sha256'] == entries[step, 'gradients']['sha256'], 'gradient binding differs')
            require(post['pre_update_state']['sha256'] == entries[step, 'pre_update']['sha256'], 'post binding differs')
            proposed = .05 * cast
            reproduced = z - proposed
            require(torch.equal(reproduced, post['latents']), 'original bf16 arithmetic not exact')
            gstats = {name: stats(grads['per_model_gradients'][name]) for name in MODELS}
            rows.append(dict(step=step, scene=schedule[step-1], z=z.clone(), joint=joint.clone(),
                             proposed=proposed.clone(), actual=z.float() - post['latents'].float(),
                             gradient_stats=gstats))
        all_states[variant] = rows
        source_meta[variant] = dict(candidate_id=summary['candidate_id'], state_manifest_sha256=sha(run / 'state_evidence/MANIFEST.json'),
                                    original_arithmetic_exact_steps=100, train_scenes=len(schedule))

    pool = torch.cat([r['proposed'].double().abs().flatten() for r in all_states['C3']])
    quantile = float(torch.quantile(pool, plan['quantile'], interpolation='linear'))
    rounded = torch.tensor(quantile, dtype=torch.bfloat16)
    if float(rounded) > quantile:
        rounded = torch.nextafter(rounded, torch.zeros_like(rounded))
    cap = float(rounded)
    require(0 < cap <= quantile and math.isfinite(cap), 'invalid frozen cap')
    results, step_rows = {}, []
    for variant, rows in all_states.items():
        proposed_pool, actual_pool, clipped_pool, new_actual_pool = [], [], [], []
        count_clipped = count_changed = count_actual_over_cap = count_unchanged = 0
        clipped_steps = 0
        for r in rows:
            original = r['proposed']
            clipped = original.clamp(-cap, cap)
            counterfactual_z = r['z'] - clipped
            realized = r['z'].float() - counterfactual_z.float()
            mask = original.abs().float() > cap
            require(bool((clipped.abs().float() <= cap).all()), 'proposed cap violated')
            require(torch.equal(clipped[~mask], original[~mask]), 'non-tail update changed')
            original_next = r['z'] - original
            changed = int(torch.count_nonzero(counterfactual_z != original_next))
            nc = int(mask.sum())
            count_clipped += nc; count_changed += changed
            count_actual_over_cap += int((realized.abs() > cap).sum())
            count_unchanged += int(torch.count_nonzero(realized == 0))
            clipped_steps += int(nc > 0)
            orig_s, new_s = stats(r['actual']), stats(realized)
            step_rows.append(dict(variant=variant, global_step=r['step'], sample_id=r['scene']['sample_id'],
                dataset=r['scene']['dataset'], split='train', cap=cap, clipped_fraction=nc/original.numel(),
                realized_coordinates_changed_fraction=changed/original.numel(),
                proposed_rms=stats(original)['rms'], proposed_abs_max=float(original.abs().max()),
                original_realized_rms=orig_s['rms'], original_realized_abs_max=orig_s['abs_max'],
                clipped_proposed_rms=stats(clipped)['rms'],
                clipped_realized_rms=new_s['rms'], clipped_realized_abs_max=new_s['abs_max'],
                original_realized_zero_fraction=1-orig_s['nonzero_count']/original.numel(),
                clipped_realized_zero_fraction=1-new_s['nonzero_count']/original.numel(),
                realized_cosine=cosine(r['actual'],realized),
                fasterrcnn_raw_gradient_rms=r['gradient_stats']['fasterrcnn']['rms'],
                ssd300_raw_gradient_rms=r['gradient_stats']['ssd300']['rms']))
            proposed_pool.append(original.flatten().float()); actual_pool.append(r['actual'].flatten())
            clipped_pool.append(clipped.flatten().float()); new_actual_pool.append(realized.flatten())
        a, b, c, d = [torch.cat(v) for v in (proposed_pool, actual_pool, clipped_pool, new_actual_pool)]
        results[variant] = dict(source=source_meta[variant], proposed=stats(a), original_realized=stats(b),
            clipped_proposed=stats(c), clipped_fixed_state_realized=stats(d), clipped_steps=clipped_steps,
            clipped_fraction=count_clipped/a.numel(), changed_realized_fraction=count_changed/a.numel(),
            clipped_realized_over_nominal_cap_fraction=count_actual_over_cap/a.numel(),
            clipped_realized_zero_fraction=count_unchanged/a.numel(),
            proposed_squared_norm_retained=float(c.double().square().sum()/a.double().square().sum()),
            realized_squared_norm_retained=float(d.double().square().sum()/b.double().square().sum()),
            pooled_realized_cosine=cosine(b,d),
            exact_zero_gradient_steps={name:[r['step'] for r in rows if r['gradient_stats'][name]['nonzero_count']==0] for name in MODELS})
    # Recheck every source actually used, without loading any evaluation files.
    for rel, entry in read_sources.items():
        require(sha(SOURCE/rel) == entry['sha256'], 'source mutated during analysis')
    out.mkdir()
    result = dict(status='complete_train_only_observations', analysis_plan_sha256=sha(HERE/'ANALYSIS_PLAN.json'),
        script_sha256=sha(__file__), delivery_manifest_sha256=DELIVERY_SHA,
        threshold=dict(source='C3 train only', quantile=.99, pooled_quantile_float64=quantile,
                       bf16_floor_cap=cap, components=pool.numel(), future_rule='fixed across all future steps; do not recalibrate from validation'),
        variants=results, source_files=[read_sources[k] for k in sorted(read_sources)],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),
        limitations=['Each counterfactual uses its original saved latent and gradient; no updated trajectory or new loss is evaluated.',
                     'Nominal injection cap is not a strict cap on realized bf16 subtraction difference.',
                     'No evidence of attack improvement; no validation/test/model/remote access.'])
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (out/'PER_STEP.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(step_rows[0]));writer.writeheader();writer.writerows(step_rows)
    print(json.dumps({'threshold':result['threshold'],'variants':results,'verified_source_count':len(read_sources)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
