#!/usr/bin/env python3
"""New fixed-final500 eight-model development evaluation; historical code is read-only."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[_key] = '1'
R2_MANIFEST = '92af9063dbe07898eb562b0d324e9d1bd746467ce1edea408f9b5f8940f82c85'
JOINT_V2_MANIFEST = '6a9c07f0d1fcb7d1fac6b2435935ede8c45c73f13f7b9bc01bd47c57d2c23c74'
LONG_IDENTITY = 'r2_joint_clip120_followup_20261004'
SELECTION_CONTEXT = 'user_requested_120_after_six_group_development_results'
LONG_CANDIDATE_ID = LONG_IDENTITY + '-two-detector-foa-plus-fixed-depth'
LONG_BUDGET = {'detector_forwards': 4000, 'autograd_calls': 500, 'transformer_forward_calls': 500,
               'vae_encode_calls': 21, 'vae_decode_calls': 1021, 'noise_draws': 20,
               'depth_clean_forwards': 4000, 'depth_adv_forwards': 4000}
DEPTH_MODELS = ('monodepth2', 'dpt_large', 'depth_anything_v1', 'zoedepth')
SEMANTICS = {'monodepth2': 'mono_scale_ambiguous_depth', 'dpt_large': 'relative_inverse_depth',
             'depth_anything_v1': 'relative_inverse_depth', 'zoedepth': 'metric_depth'}
STATES = ('clean', 'zero', 'random', 'final')
COMPARISONS = (('clean', 'zero'), ('clean', 'random'), ('clean', 'final'), ('zero', 'final'))
SEED = 20260919


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_metric_functions(path):
    """Execute exactly the three original function definitions, without model imports."""
    import math
    import statistics
    import numpy as np
    import torch
    import torch.nn.functional as F
    names = {'depth_metrics', 'depth_summary', 'percentile_interval'}
    nodes = [node for node in ast.parse(Path(path).read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise ValueError('frozen depth metric definitions are missing or ambiguous')
    namespace = dict(math=math, statistics=statistics, np=np, torch=torch, F=F)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('an explicit lowercase 64-character SHA256 binding is required')
    return value


def bind_long_package(directory, expected_sha):
    """No placeholder/default SHA: the launcher must bind the newly sealed package."""
    require_sha(expected_sha)
    directory = Path(directory).resolve()
    manifest = directory / 'PACKAGE_MANIFEST.json'
    if manifest.is_symlink() or not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('long-training source package SHA mismatch')
    entries = json.loads(manifest.read_text(encoding='utf-8'))['files']
    names = set()
    for entry in entries:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or str(relative) in names:
            raise ValueError('unsafe or duplicate long-package member')
        names.add(str(relative))
        if any((directory / Path(*relative.parts[:i])).is_symlink() for i in range(len(relative.parts) + 1)):
            raise ValueError('long-package member symlinks forbidden')
        path = directory / relative
        if (not path.is_file() or path.stat().st_size != entry['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != require_sha(entry['sha256'])):
            raise ValueError('long-package member mismatch: ' + str(relative))
    if not {'clip_training.py', 'clip_evaluate.py', 'clip_audit_depth.py', 'clip_audit_detection.py'} <= names:
        raise ValueError('training/evaluation/auditor sources must all be sealed in the long package')
    return {'manifest_sha256': expected_sha, 'verified_members': len(entries)}


def validate_clip_policy(policy):
    """Only the explicitly requested post-evaluation cap120 policy is accepted."""
    if (not isinstance(policy, dict)
            or set(policy) != {'group_id', 'max_norm', 'grid_sha256', 'scope', 'position', 'norm_dtype'}):
        raise ValueError('complete cap120 gradient_clipping policy is required')
    if policy.get('group_id') != 'cap120':
        raise ValueError('only the independent cap120 group is accepted')
    require_sha(policy.get('grid_sha256'))
    for key, expected in {'scope': 'global_total_latent_gradient',
                          'position': 'after_100000_before_lr', 'norm_dtype': 'float32'}.items():
        if policy.get(key) != expected:
            raise ValueError('clipping policy differs: ' + key)
    threshold = policy.get('max_norm')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold != 120.0:
        raise ValueError('cap120 clipping threshold must be exactly 120.0')
    return policy


def read_bound_json(root, relative, expected_sha):
    require_sha(expected_sha)
    root = Path(root)
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe clipping evidence path')
    path = root / relative
    if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(len(relative.parts) + 1)):
        raise ValueError('clipping evidence symlinks forbidden')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('clipping evidence SHA mismatch: ' + str(relative))
    def reject(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def validate_clip_plan(run, policy):
    from clip_evidence import validate_followup_plan
    validate_clip_policy(policy)
    plan = read_bound_json(run, 'CLIP_PLAN.json', policy['grid_sha256'])
    validate_followup_plan(plan)
    groups = plan.get('groups')
    if (not isinstance(groups, list) or len(groups) != 1
            or groups[0].get('id') != 'cap120' or groups[0].get('max_norm') != 120.0
            or isinstance(groups[0].get('max_norm'), bool)
            or type(groups[0].get('executed_updates')) is not int or groups[0]['executed_updates'] != 500):
        raise ValueError('exactly one fixed cap120/500 followup group required')
    return plan


def validate_comparison_gate(run, summary, manifest=None):
    policy = validate_clip_policy(summary.get('gradient_clipping'))
    plan = validate_clip_plan(run, policy)
    gate = read_bound_json(run, 'CLIPPING_COMPARISON_GATE.json',
                          summary.get('clipping_comparison_gate_sha256'))
    off = policy['max_norm'] is None
    expected_count = 2041 if off else 4
    expected_status = 'exact_full_trajectory' if off else 'exact_initialization_and_first_forward'
    if (gate.get('status') != expected_status or type(gate.get('tolerance')) is not int
            or gate['tolerance'] != 0 or type(gate.get('compared_states')) is not int
            or gate['compared_states'] != expected_count):
        raise ValueError('clipping comparison gate mode/count/tolerance differs')
    for field in ('group_id', 'max_norm', 'grid_sha256'):
        if gate.get(field) != policy[field] or isinstance(gate.get('max_norm'), bool):
            raise ValueError('clipping comparison gate policy differs: ' + field)
    reference = gate.get('reference', {})
    if reference.get('candidate_id') != 'r2_joint_depth_long500_20261001-two-detector-foa-plus-fixed-depth':
        raise ValueError('clipping gate must bind the accepted original long500 reference')
    for field in ('summary_sha256', 'state_manifest_sha256'):
        require_sha(reference.get(field))
    if (reference['summary_sha256'] != plan['reference']['summary_sha256']
            or reference['state_manifest_sha256'] != plan['reference'].get('state_evidence_manifest_sha256')):
        raise ValueError('clipping gate/grid reference differs')
    if plan['reference']['training_manifest_sha256'] != summary.get('training_manifest_sha256'):
        raise ValueError('clipping grid/training data manifest differs')
    if manifest is not None:
        if (reference != manifest.get('reference_binding')
                or manifest.get('gradient_clipping') != policy
                or manifest.get('comparison_counts', {}).get('states_checked') != expected_count):
            raise ValueError('clipping comparison receipt differs from state manifest')
    from clip_evidence import require_matching_environment
    off = summary.get('off_control_binding', {})
    expected = plan['execution_off']
    fields = {'control_kind': expected['origin'], 'prior_package_sha256': expected['package_sha256'],
              'followup_package_sha256': summary.get('package_manifest_sha256'),
              'files': expected['files'], 'scientific_environment': expected['scientific_environment'],
              'compared_states': 2041, 'tolerance': 0}
    if any(type(off.get(key)) is not type(value) or off[key] != value for key, value in fields.items()):
        raise ValueError('cap120 execution OFF binding differs from frozen followup plan')
    runtime = read_bound_json(run, 'runtime_provenance.json', summary.get('runtime_provenance_sha256'))
    environment_gate = require_matching_environment(runtime, off)
    if (summary.get('execution_off_environment_gate') != environment_gate
            or runtime.get('execution_off_environment_gate') != environment_gate):
        raise ValueError('cap120 execution environment gate differs from recomputation')
    return gate


def run_role(summary, expected_package_sha):
    require_sha(expected_package_sha)
    if (summary.get('identity'), summary.get('candidate_id')) != (LONG_IDENTITY, LONG_CANDIDATE_ID):
        raise ValueError('only the independent fixed-final500 joint candidate is accepted')
    if summary.get('package_manifest_sha256') != expected_package_sha:
        raise ValueError('long training run differs from the explicitly bound sealed package')
    if (summary.get('status') != 'train_optimization_complete' or summary.get('executed_updates') != 500
            or summary.get('checkpoint_selection') != 'fixed final500 only'
            or summary.get('test_data_opened') is not False or summary.get('validation_data_opened') is not False):
        raise ValueError('completed isolated fixed-final500 training is required')
    if summary.get('accepted_r2_source_manifest_sha256') != R2_MANIFEST:
        raise ValueError('joint run is not bound to the accepted R2 source')
    if (summary.get('accepted_joint_v2_manifest_sha256') != JOINT_V2_MANIFEST
            or summary.get('run_role') != 'fresh_joint500_clip120_posthoc_followup_training' or summary.get('fresh_start') is not True
            or summary.get('resume_enabled') is not False):
        raise ValueError('fresh long500 run must bind the frozen joint-v2 implementation')
    if summary.get('exploratory_posthoc') is not True or summary.get('selection_context') != SELECTION_CONTEXT:
        raise ValueError('explicit post-evaluation exploratory selection context required')
    weight = summary.get('depth_weight')
    if isinstance(weight, bool) or weight != 1. or summary.get('depth_beta') != .1:
        raise ValueError('long evaluation accepts only fixed depth weight one and beta 0.1')
    if summary.get('training_depth_models') != list(DEPTH_MODELS[:2]) or summary.get('training_models') != ['fasterrcnn', 'ssd300']:
        raise ValueError('joint run depth model identities differ')
    if summary.get('config', {}).get('outer_rounds') != 20:
        raise ValueError('long training must execute 20 outer rounds')
    budget = summary.get('budget', {})
    if any(budget.get(field) != LONG_BUDGET for field in ('planned', 'attempted', 'completed')):
        raise ValueError('long training did not complete its exact 500-step call budget')
    policy = validate_clip_policy(summary.get('gradient_clipping'))
    return 'joint_fixed_depth_weight_one_final500_posthoc_' + policy['group_id'], ('fasterrcnn', 'ssd300', *DEPTH_MODELS[:2])


def checked_long_run(run, engine, parent, r2, expected_package_sha):
    """Check the new lean completion schema and fixed500 patch, never the old 409-state loader."""
    import torch
    summary = r2.read_json(r2.bound_file(run, 'summary.json'))
    run_role(summary, expected_package_sha)
    for field, relative in (('training_manifest_sha256', '01_公共数据集/runtime_v1/train_validation_manifest.csv'),
                            ('model_manifest_sha256', '02_模型/model_manifest.json')):
        if summary.get(field) != r2.sha(parent / relative):
            raise ValueError('long training/evaluation asset mismatch: ' + field)
    manifest = r2.read_json(r2.bound_file(run, 'state_evidence/MANIFEST.json', require_sha(summary.get('state_evidence_manifest_sha256'))))
    if (manifest.get('schema') != 'r2-joint-clip500-lean-state-manifest-v1' or manifest.get('status') != 'complete'
            or manifest.get('executed_updates') != 500 or manifest.get('candidate_id') != LONG_CANDIDATE_ID):
        raise ValueError('incomplete or foreign lean-state manifest')
    expected_order = [('initialization', 0)]
    expected_updates = [0]
    for outer in range(20):
        expected_order.append(('outer_start', 0))
        expected_updates.append(outer * 25)
        expected_order.extend((phase, step) for step in range(outer * 25 + 1, outer * 25 + 26)
                              for phase in ('before_denoise', 'pre_backward', 'pre_update', 'post_update'))
        expected_updates.extend(count for step in range(outer * 25 + 1, outer * 25 + 26)
                                for count in (step - 1, step - 1, step - 1, step))
        expected_order.append(('outer_end', 0))
        expected_updates.append((outer + 1) * 25)
    entries = manifest.get('entries', [])
    if ([(entry.get('phase'), entry.get('global_step')) for entry in entries] != expected_order
            or [entry.get('index') for entry in entries] != list(range(1, 2042))
            or [entry.get('updates_applied') for entry in entries] != expected_updates):
        raise ValueError('complete ordered 2041-record lean trajectory required')
    if len(manifest.get('recovery_checkpoints', [])) != 20:
        raise ValueError('20 outer-round recovery checkpoints are required')
    for entry in entries:
        require_sha(entry.get('state_fingerprint'))
        if entry.get('recording_preserved_rng') is not True:
            raise ValueError('lean-state recording did not preserve RNG')
    ledger_path = r2.bound_file(run, 'state_evidence/ledger.jsonl', require_sha(manifest.get('ledger_sha256')))
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
    if ledger != entries:
        raise ValueError('lean ledger differs from completed manifest entries')
    for outer, recovery in enumerate(manifest['recovery_checkpoints'], 1):
        if recovery.get('completed_outer_round') != outer or recovery.get('executed_updates') != outer * 25:
            raise ValueError('outer recovery sequence differs')
        path = r2.bound_file(run, recovery['path'], require_sha(recovery.get('sha256')))
        if path.stat().st_size != recovery.get('bytes'):
            raise ValueError('outer recovery file size differs')
    initialization = manifest.get('initialization_state', {})
    initial_path = r2.bound_file(run, initialization['path'], require_sha(initialization.get('sha256')))
    if initial_path.stat().st_size != initialization.get('bytes'):
        raise ValueError('initialization evidence size differs')
    for entry in entries:
        expected_stored = initialization if entry['phase'] == 'initialization' else None
        if entry['phase'] == 'outer_end':
            expected_stored = {key: manifest['recovery_checkpoints'][entry['updates_applied'] // 25 - 1][key]
                               for key in ('path', 'sha256', 'bytes')}
        if entry.get('stored_state') != expected_stored:
            raise ValueError('lean stored-state pointer differs from initialization/recovery manifest')
    validate_comparison_gate(run, summary, manifest)
    checkpoint_bindings = summary.get('checkpoint_bindings', {})
    if set(checkpoint_bindings) != {'checkpoint_0100.pt', 'checkpoint_0250.pt', 'checkpoint_0500.pt'}:
        raise ValueError('only the declared 100/250/500 observation checkpoint set is accepted')
    for name, count in (('checkpoint_0100.pt', 100), ('checkpoint_0250.pt', 250), ('checkpoint_0500.pt', 500)):
        path = r2.bound_file(run, name, require_sha(checkpoint_bindings[name]))
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('metadata', {}).get('updates') != count:
            raise ValueError('checkpoint update count differs: ' + name)
    patches = {}
    for state, name in (('zero', 'zero_step_patch.pt'), ('final', 'final_patch.pt')):
        patches[state] = engine.load_patch(r2.bound_file(run, name, require_sha(summary.get('patch_bindings', {}).get(name))))
    if r2.tree_difference(patches['final'], engine.load_patch(Path(run) / 'checkpoint_0500.pt')):
        raise ValueError('final patch differs from the fixed checkpoint500')
    patches['random'] = engine.common_random_patch()
    return summary, patches


def depth_preflight(manifest, checked_file):
    definitions = json.loads(Path(manifest).read_text(encoding='utf-8'))['models']
    checks = []
    for model in DEPTH_MODELS:
        matches = [entry for entry in definitions if entry['id'] == model]
        if len(matches) != 1 or not matches[0].get('artifacts'):
            raise ValueError('unbound depth model: ' + model)
        for item in matches[0]['artifacts']:
            checks.append(dict(checked_file(item['path'], item['sha256']), model_id=model))
    return {'inputs_ready': all(row['status'] == 'pass' for row in checks), 'checks': checks}


def summarize_depth(metrics, rows):
    output = []
    for model in DEPTH_MODELS:
        for reference, evaluated in COMPARISONS:
            for domain in ('kitti', 'nuscenes', 'combined'):
                subset = [row for row in rows if row['model'] == model
                          and row['reference_state'] == reference and row['evaluated_state'] == evaluated
                          and (domain == 'combined' or row['dataset'] == domain)]
                value = metrics.depth_summary(subset, 1000, SEED)
                value.update(model=model, dataset=domain, reference_state=reference,
                             evaluated_state=evaluated, depth_representation=SEMANTICS[model],
                             reference='same_model_prediction_on_' + reference + '_scene')
                output.append(value)
    return output


def evaluate_depth(args, engine, metrics, selected, patches, device, role, trained_models, r2):
    import torch
    from model_adapters import build_depth_adapter
    car, mask = engine.load_car(args.car, args.car_mask, device)
    patches = {state: patch.to(device) for state, patch in patches.items()}
    raw_root = args.output / 'depth_raw'
    (raw_root / 'masks').mkdir(parents=True)
    detector_inputs = r2.read_json(args.output / 'rendered_input_hashes.json')['rows']
    input_reference = {(row['dataset'], row['sample_id'], row['state']): row
                       for row in detector_inputs if row['model'] == 'fasterrcnn'}
    rows, entries, geometry, input_rows, masks = [], [], [], [], {}
    counts = dict.fromkeys(DEPTH_MODELS, 0)
    for model in DEPTH_MODELS:
        adapter = build_depth_adapter(model, device, model_manifest=args.model_manifest)
        try:
            with torch.no_grad():
                for index, record in enumerate(selected):
                    context = engine.build_context(record, car, mask, device)
                    key = (record.dataset, record.sample_id)
                    mask_hash = r2.tensor_sha(context.object_mask)
                    if key not in masks:
                        name = 'depth_raw/masks/%03d.pt' % index
                        torch.save(context.object_mask.detach().cpu(), args.output / name)
                        masks[key] = {'path': name, 'sha256': r2.sha(args.output / name), 'tensor_sha256': mask_hash}
                    elif masks[key]['tensor_sha256'] != mask_hash:
                        raise ValueError('object mask changed across depth models')
                    payload = {'schema': 'joint-depth-evaluation-output-v1', 'model': model,
                               'sample_id': record.sample_id, 'dataset': record.dataset, 'split': 'validation',
                               'semantics': SEMANTICS[model], 'object_mask': masks[key], 'states': {},
                               'input_sha256': {}, 'support_sha256': {}}
                    for state in STATES:
                        if state == 'clean':
                            image = context.benign
                            expected = input_reference[(*key, 'zero')]['clean_input_sha256']
                        else:
                            render = engine.RoadRGBView(patches[state]).render(context)
                            image = render.image
                            geometry.append(engine.invariant_record(context, render, record, model, state))
                            expected_row = input_reference[(*key, state)]
                            expected = expected_row['rendered_input_sha256']
                            support_hash = r2.tensor_sha(render.support)
                            if support_hash != expected_row['support_tensor_sha256']:
                                raise ValueError('depth/detection support differs')
                            payload['support_sha256'][state] = support_hash
                        image_hash = r2.tensor_sha(image)
                        if image_hash != expected:
                            raise ValueError('depth/detection rendered input differs: ' + str((*key, state)))
                        prediction = adapter.predict_with_diagnostics(image)
                        counts[model] += 1
                        if prediction.metadata.get('evaluation_semantics') != SEMANTICS[model]:
                            raise ValueError('depth evaluation semantics differs: ' + model)
                        value = prediction.stages.get('evaluation_output')
                        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                            raise ValueError('missing floating evaluation_output')
                        value = value.detach().cpu().contiguous()
                        if value.ndim == 3:
                            value = value.unsqueeze(1)
                        if value.shape != context.object_mask.shape or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                            raise ValueError('invalid or non-canvas depth evaluation_output')
                        payload['states'][state] = value
                        payload['input_sha256'][state] = image_hash
                        input_rows.append({'model': model, 'sample_id': record.sample_id, 'dataset': record.dataset,
                                           'split': 'validation', 'state': state, 'input_sha256': image_hash})
                        del prediction
                    object_mask = context.object_mask.detach().cpu()
                    for reference, evaluated in COMPARISONS:
                        row = {'method': role, 'model': model, 'sample_id': record.sample_id,
                               'dataset': record.dataset, 'split': 'validation', 'reference_state': reference,
                               'evaluated_state': evaluated, 'used_in_training': model in trained_models}
                        row.update(metrics.depth_metrics(payload['states'][evaluated], payload['states'][reference],
                                                         object_mask, semantics=SEMANTICS[model]))
                        rows.append(row)
                    name = 'depth_raw/%s_%03d.pt' % (model, index)
                    torch.save(payload, args.output / name)
                    entries.append({'model': model, 'sample_id': record.sample_id, 'dataset': record.dataset,
                                    'split': 'validation', 'path': name, 'sha256': r2.sha(args.output / name),
                                    'object_mask': masks[key], 'semantics': SEMANTICS[model],
                                    'tensor_sha256': {state: r2.tensor_sha(payload['states'][state]) for state in STATES}})
                    print('DEPTH_VALIDATION %s %d/40 %s' % (model, index + 1, record.sample_id), flush=True)
                    del payload
        finally:
            del adapter
            torch.cuda.empty_cache()
    if len(rows) != 640 or len(entries) != 160 or len(geometry) != 480 or counts != dict.fromkeys(DEPTH_MODELS, 160):
        raise ValueError('depth coverage differs from fixed four-model 40-scene protocol')
    r2.write_json(args.output / 'depth_per_sample.json', rows)
    r2.write_json(args.output / 'depth_geometry_audit.json', {'status': 'pass', 'checks': geometry})
    r2.write_json(args.output / 'depth_input_hashes.json', {'rows': input_rows, 'all_equal_to_detection_inputs': True})
    manifest = {'schema': 'joint-depth-raw-manifest-v1', 'entries': entries, 'states': list(STATES),
                'unique_scenes': 40, 'model_forward_count': 640, 'metric_input_stage': 'evaluation_output',
                'comparisons': [list(pair) for pair in COMPARISONS], 'test_data_opened': False}
    r2.write_json(raw_root / 'MANIFEST.json', manifest)
    result = {'status': 'development_validation_complete', 'model_forward_count': 640, 'records': 640,
              'evaluation_models': list(DEPTH_MODELS), 'evaluation_model_scene_forwards': counts,
              'summaries': summarize_depth(metrics, rows), 'summary_rows': 48,
              'metric': 'clean-reference prediction shift; no ground-truth depth error or TNR_depth',
              'direction_semantics': {'monodepth2': 'larger means farther; scale ambiguous',
                                     'dpt_large': 'smaller native inverse quantity means farther',
                                     'depth_anything_v1': 'smaller native inverse quantity means farther',
                                     'zoedepth': 'larger means farther'},
              'no_cross_model_raw_depth_or_signed_inverse_mean': True, 'bootstrap_resamples': 1000,
              'bootstrap_seed': SEED, 'zero_to_final': 'direct original metric on final and zero tensors',
              'raw_manifest_sha256': r2.sha(raw_root / 'MANIFEST.json')}
    r2.write_json(args.output / 'depth_summary.json', result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r2-source', 'child', 'run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--training-package-manifest-sha256', required=True,
                        help='Required actual SHA of the newly sealed clipping-sweep PACKAGE_MANIFEST.json')
    parser.add_argument('--training-source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--weights-root', type=Path)
    parser.add_argument('--source-prefix', type=Path)
    parser.add_argument('--source-root', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if socket.gethostname() != args.expected_hostname:
        raise ValueError('live hostname mismatch')
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError('fresh output with an existing parent is required')
    if any(path.is_symlink() for path in (args.output, *args.output.absolute().parents)):
        raise ValueError('output symlinks forbidden')
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.run == args.output or args.run in args.output.parents:
        raise ValueError('evaluation output must be outside training')
    if (args.source_prefix is None) != (args.source_root is None):
        raise ValueError('source remapping requires both paths')
    sys.path.insert(0, str(args.r2_source.resolve()))
    training = module_from_path('_joint_eval_original_training', args.r2_source / 'training.py')
    training.verify_tree(args.r2_source, R2_MANIFEST)
    training.verify_tree(args.child, training.CHILD_MANIFEST_SHA)
    long_package = bind_long_package(args.training_source, args.training_package_manifest_sha256)
    if hashlib.sha256((args.training_source / 'clip_evaluate.py').read_bytes()).hexdigest() != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('executing evaluator differs from the explicitly sealed package')
    r2 = module_from_path('_joint_eval_original_evaluate', args.r2_source / 'evaluate.py')
    r2.configure()
    import torch
    import torchvision
    from importlib.metadata import version
    if (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0], version('diffusers')) != ('2.8.0', '0.23.0', '0.38.0'):
        raise ValueError('frozen runtime versions differ')
    engine, parent = r2.load_engine(args.child)
    summary, patches = checked_long_run(args.run, engine, parent, r2, args.training_package_manifest_sha256)
    role, trained = run_role(summary, args.training_package_manifest_sha256)
    if summary.get('validation_data_opened') is not False or summary.get('checkpoint_selection') != 'fixed final500 only':
        raise ValueError('training must be validation-isolated with fixed final500')
    if summary.get('historical_source_manifest_sha256') != training.CHILD_MANIFEST_SHA:
        raise ValueError('training parent source binding differs')
    metric_path = parent / '04_统一评估/evaluator.py'
    metrics = load_metric_functions(metric_path)
    args.model_manifest = parent / '02_模型/model_manifest.json'
    args.car, args.car_mask = parent / '01_公共数据集/car_00.jpg', parent / '01_公共数据集/car_00_mask.jpg'
    args.run_output = args.run
    records = engine.training_records(parent / '01_公共数据集/runtime_v1/train_validation_manifest.csv', args.source_prefix, args.source_root)
    selected = engine.validation_rows(records)
    if Counter(record.dataset for record in selected) != {'kitti': 20, 'nuscenes': 20}:
        raise ValueError('frozen 20+20 validation rows required')
    from road_runtime import checked_file
    checks = engine.preflight(records, r2.MODELS, args.model_manifest, args.car, args.car_mask, args.weights_root)
    checks['depth'] = depth_preflight(args.model_manifest, checked_file)
    if not checks['inputs_ready'] or not checks['depth']['inputs_ready'] or not torch.cuda.is_available():
        raise RuntimeError('evaluation assets or CUDA unavailable')
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        r2.write_json(args.output / 'input_preflight.json', checks)
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        detector = r2.evaluate(args, engine, records, summary, patches, device)
        # The reused function performs fixed scene/model/state arithmetic only.
        # Replace its historical budget labels before writing any task summary.
        detector.update(method=role, training_models=list(trained), updates=500, checkpoint_selection='fixed_final500',
                        detector_arithmetic='unchanged frozen R2 evaluate.evaluate; explicit clip500 budget and group metadata',
                        evaluation_model_roles={model: ('source' if model in trained else 'transfer') for model in r2.MODELS})
        rows = r2.read_json(args.output / 'per_sample.json')
        for row in rows:
            row['method'] = role
        r2.write_json(args.output / 'per_sample.json', rows)
        r2.write_json(args.output / 'detection_summary.json', detector)
        depth = evaluate_depth(args, engine, metrics, selected, patches, device, role, trained, r2)
        torch.cuda.synchronize(device)
        with (args.output / 'CLIP_PLAN.json').open('xb') as stream:
            stream.write((args.run / 'CLIP_PLAN.json').read_bytes())
        result = {'schema': 'r2-joint-clip120-eight-model-development-evaluation-v1', 'status': 'development_validation_complete',
                  'candidate_id': summary['candidate_id'], 'method': role, 'run': str(args.run),
                  'run_summary_sha256': r2.sha(args.run / 'summary.json'), 'training_models': list(trained),
                  'designated_core_models': ['fasterrcnn', 'ssd300', 'monodepth2', 'dpt_large'],
                  'evaluation_models': [*r2.MODELS, *DEPTH_MODELS], 'scene_count': 40, 'updates': 500,
                  'training_budget_updates': 500, 'training_outer_rounds': 20,
                  'gradient_clipping': summary['gradient_clipping'],
                  'selection_context': SELECTION_CONTEXT, 'exploratory_posthoc': True,
                  'comparison_with_unclipped500': 'user-requested 120 after viewing six-group development results; exploratory followup, not a preregistered grid point',
                  'clipping_comparison_gate_sha256': summary['clipping_comparison_gate_sha256'],
                  'evaluation_forward_total': 1280, 'detection_summary_rows': 48, 'depth_summary_rows': 48,
                  'validation_scenes': [{'sample_id': record.sample_id, 'dataset': record.dataset, 'split': record.split} for record in selected],
                  'depth_semantics': SEMANTICS, 'test_data_opened': False, 'formal_test_allowed': False,
                  'automatic_promotion': False, 'checkpoint_selection': 'fixed_final500', 'eot': None,
                  'source_bindings': {'r2_manifest_sha256': R2_MANIFEST, 'child_manifest_sha256': training.CHILD_MANIFEST_SHA,
                                      'clip_training_package_manifest_sha256': args.training_package_manifest_sha256,
                                      'r2_evaluate_sha256': r2.sha(args.r2_source / 'evaluate.py'),
                                      'depth_evaluator_sha256': r2.sha(metric_path), 'depth_evaluator_relative_path': 'parent_v17/04_统一评估/evaluator.py',
                                      'training_manifest_sha256': summary['training_manifest_sha256'],
                                      'model_manifest_sha256': r2.sha(args.model_manifest), 'wrapper_sha256': r2.sha(Path(__file__))},
                  'clip_training_source_package': long_package,
                  'hostname': socket.gethostname(), 'seconds': time.monotonic() - started,
                  'runtime': {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': version('numpy'),
                              'transformers': version('transformers'), 'cpu_threads': torch.get_num_threads()},
                  'detection_summary_sha256': r2.sha(args.output / 'detection_summary.json'),
                  'depth_summary_sha256': r2.sha(args.output / 'depth_summary.json')}
        result['artifact_bindings'] = {str(path.relative_to(args.output)): r2.sha(path)
                                       for path in sorted(args.output.rglob('*')) if path.is_file()}
        r2.write_json(args.output / 'summary.json', result)
        print(json.dumps({'status': result['status'], 'forwards': 1280, 'method': role}, ensure_ascii=False), flush=True)
    except BaseException as error:
        r2.write_json(args.output / 'failure.json', {'status': 'failed_retained', 'type': type(error).__name__,
                                                  'error': str(error), 'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
