#!/usr/bin/env python3
"""CPU-only U0/U1 validation tables, unchanged AP50/TNR and fixed final100."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

if __package__:
    from .inputs import ROOT, IDENTITY, candidate_run_path, load_protocol, parent_evaluator, sha, verify_inputs
    from .evaluate_validation import (artifact_bindings, bound_file, read_json, safe_file, write_json,
                                      require_current_objective, OBJECTIVE_DESCRIPTIONS)
else:
    from inputs import ROOT, IDENTITY, candidate_run_path, load_protocol, parent_evaluator, sha, verify_inputs
    from evaluate_validation import (artifact_bindings, bound_file, read_json, safe_file, write_json,
                                     require_current_objective, OBJECTIVE_DESCRIPTIONS)

MODELS = ('fasterrcnn', 'ssd300')
DOMAINS = ('kitti', 'nuscenes', 'combined')
STATES = ('zero', 'random', 'final')


def row_key(row):
    return row['model'], row['dataset'], row['sample_id'], row['state']


def validate_rows(rows):
    if len(rows) != 240 or len({row_key(row) for row in rows}) != 240:
        raise ValueError('exactly 240 unique validation/model/state rows required')
    ids = {}
    for model in MODELS:
        for state in STATES:
            subset = [r for r in rows if r['model'] == model and r['state'] == state]
            if Counter(r['dataset'] for r in subset) != {'kitti': 20, 'nuscenes': 20}:
                raise ValueError('each model/state requires the same 20+20 validation scenes')
            identities = {(r['dataset'], r['sample_id']) for r in subset}
            if len(identities) != 40 or (ids and identities != ids):
                raise ValueError('validation identities differ between model/states')
            ids = identities
    for row in rows:
        if row.get('split') != 'validation' or row['model'] not in MODELS or row['state'] not in STATES:
            raise ValueError('test/train or unknown evaluation row')
        for field in ('clean_candidates', 'adversarial_candidates', 'clean_car_candidates', 'adversarial_car_candidates'):
            candidates = row.get(field)
            if not isinstance(candidates, list):
                raise ValueError('missing full metric candidates')
            for item in candidates:
                if (set(item) != {'score', 'iou', 'label'} or type(item['label']) is not int or
                        item['label'] not in ((3,) if '_car_' in field else (3, 6, 8)) or
                        any(type(item[key]) not in (int, float) or not math.isfinite(item[key]) for key in ('score', 'iou')) or
                        not 0 <= item['score'] <= 1 or not 0 < item['iou'] <= 1):
                    raise ValueError('invalid original target-vehicle candidate')
    return {row_key(row): row for row in rows}


def summarize_rows(rows, *, bootstrap_samples=1000):
    engine, output = parent_evaluator(), []
    for model in MODELS:
        for state in STATES:
            for domain in DOMAINS:
                subset = [r for r in rows if r['model'] == model and r['state'] == state
                          and (domain == 'combined' or r['dataset'] == domain)]
                value = engine.detection_summary(subset, bootstrap_samples, 20260919, allow_zero_reference=True)
                value.update(model=model, state=state, dataset=domain)
                output.append(value)
    return output


def zero_to_final(rows, *, bootstrap_samples=1000):
    indexed = {row_key(row): row for row in rows}
    derived = []
    for row in rows:
        if row['state'] != 'final':
            continue
        zero = indexed[(row['model'], row['dataset'], row['sample_id'], 'zero')]
        derived.append(dict(row, clean_candidates=zero['adversarial_candidates'],
                            clean_car_candidates=zero['adversarial_car_candidates']))
    output, engine = [], parent_evaluator()
    for model in MODELS:
        for domain in DOMAINS:
            subset = [r for r in derived if r['model'] == model and (domain == 'combined' or r['dataset'] == domain)]
            value = engine.detection_summary(subset, bootstrap_samples, 20260919, allow_zero_reference=True)
            value.update(model=model, dataset=domain, reference_state='zero', evaluated_state='final',
                         tnr_undefined_reason=None if value['tnr_defined'] else 'reference_AP50_is_zero')
            output.append(value)
    return output


def load_evaluation(path, objective, *, root=ROOT, bootstrap_samples=1000):
    if objective not in ('U0','U1'):
        raise ValueError('only current U0/U1 evaluations are eligible')
    path = candidate_run_path(path, root)
    summary = read_json(safe_file(path, 'summary.json'))
    require_current_objective(summary,objective)
    if (summary.get('candidate_id') != IDENTITY + '-ours-det-only-' + objective or
            summary.get('update_variant') != objective or summary.get('task_variant') != 'det-only' or
            summary.get('status') != 'development_validation_complete' or
            summary.get('test_data_opened') is not False or summary.get('formal_test_allowed') is not False or
            summary.get('checkpoint_selection') != 'fixed_final100' or
            summary.get('evaluation_models') != list(MODELS) or
            type(summary.get('evaluation_forward_total')) is not int or summary.get('evaluation_forward_total') != 320 or
            type(summary.get('records')) is not int or summary.get('records') != 240 or
            type(summary.get('updates')) is not int or summary.get('updates') != 100 or
            type(summary.get('seed')) is not int or summary.get('seed') != 20260919 or
            summary.get('evaluation_model_scene_forwards') != dict.fromkeys(MODELS, 160)):
        raise ValueError('comparison requires completed v24 final100 validation with original budget')
    if (summary.get('training_objective_models') != list(MODELS) or
            summary.get('training_objective_description') != OBJECTIVE_DESCRIPTIONS[objective] or
            summary.get('training_target_classes') != [3] or
            summary.get('evaluation_target_classes') != [3,6,8] or
            summary.get('evaluation_car_auxiliary_classes') != [3]):
        raise ValueError('both-detector training and original evaluation objective descriptions differ')
    if summary.get('protocol_sha256') != sha(Path(root) / 'protocol/training_protocol.json'):
        raise ValueError('evaluation protocol differs')
    engine = parent_evaluator()
    if (summary.get('parent_metric_engine_candidate_id') != engine.CANDIDATE_ID or
            summary.get('parent_metric_engine_source_sha256') != sha(Path(engine.__file__))):
        raise ValueError('evaluation metric engine identity differs')
    run_name = summary.get('run_output')
    if not isinstance(run_name, str) or Path(run_name).is_absolute() or '..' in Path(run_name).parts:
        raise ValueError('invalid source training run binding')
    source_run = candidate_run_path(Path(root) / run_name, root)
    training = read_json(bound_file(source_run, 'summary.json', summary.get('run_summary_sha256')))
    require_current_objective(training,objective)
    for field in ('candidate_id', 'update_variant', 'task_variant', 'protocol_sha256', 'calibration_sha256',
                  'normalizers', 'state_evidence_manifest_sha256', 'training_manifest_sha256', 'model_manifest_sha256'):
        if summary.get(field) != training.get(field):
            raise ValueError('evaluation/training provenance differs: ' + field)
    for name, field in (('attack_trace.jsonl', 'run_trace_sha256'), ('calibration.json', 'calibration_sha256'),
                        ('state_evidence/MANIFEST.json', 'state_evidence_manifest_sha256'),
                        ('runtime_provenance.json', 'training_runtime_provenance_sha256')):
        bound_file(source_run, name, summary.get(field))
    if summary.get('training_runtime_provenance_sha256') != training.get('runtime_provenance_sha256'):
        raise ValueError('training runtime provenance binding differs')
    bindings = summary.get('artifact_bindings', {})
    required = {'per_sample.json', 'geometry_audit.json', 'rendered_input_hashes.json', 'random_control_patch.pt'}
    if not isinstance(bindings, dict) or not required.issubset(bindings):
        raise ValueError('missing evaluation artifact bindings')
    for name, digest in bindings.items():
        bound_file(path, name, digest)
    if set(bindings) != {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'summary.json'}:
        raise ValueError('evaluation artifact file set differs')
    rows = read_json(path / 'per_sample.json')
    indexed = validate_rows(rows)
    # Metadata only: original manifest, never source image bytes or model weights.
    original_records = engine.training_records(Path(engine.ROOT) / '01_公共数据集/runtime_v1/train_validation_manifest.csv')
    fixed_ids = {(r.dataset, r.sample_id) for r in engine.validation_rows(original_records)}
    if {(r['dataset'], r['sample_id']) for r in rows} != fixed_ids:
        raise ValueError('validation sample identities differ from frozen manifest')
    recomputed = summarize_rows(rows, bootstrap_samples=bootstrap_samples)
    if recomputed != summary.get('summaries'):
        raise ValueError('saved summary differs from original metric recomputation')
    geometry = read_json(path / 'geometry_audit.json')
    if (geometry.get('status') != 'pass' or geometry.get('unique_scenes') != 40 or
            geometry.get('checked_model_scene_states') != 240 or len(geometry.get('checks', [])) != 240 or
            {row_key(r) for r in geometry['checks']} != set(indexed) or
            any(not all(r['checks'].values()) or r.get('support_pixels', 0) <= 0 for r in geometry['checks'])):
        raise ValueError('geometry evidence is incomplete or failed')
    hashes = read_json(path / 'rendered_input_hashes.json')
    if (hashes.get('schema') != 'v24-passive-validation-input-sha-v1' or
            hashes.get('all_240_rendered_inputs_hashed') is not True or
            hashes.get('additional_model_forwards') != 0 or len(hashes.get('rows', [])) != 240 or
            {row_key(r) for r in hashes['rows']} != set(indexed)):
        raise ValueError('full passive input hash coverage differs')
    for row in hashes['rows']:
        for key in ('clean_input_sha256', 'rendered_input_sha256', 'support_tensor_sha256'):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise ValueError('invalid typed input SHA')
    return {'path': path, 'summary': summary, 'rows': rows, 'index': indexed,
            'hashes': {row_key(r): r for r in hashes['rows']}, 'summaries': recomputed,
            'zero_to_final': zero_to_final(rows, bootstrap_samples=bootstrap_samples)}


def compare(a, b):
    """A fixed descriptive comparison; prediction differences are observations."""
    require_current_objective(a['summary'],'U0')
    require_current_objective(b['summary'],'U1')
    if set(a['index']) != set(b['index']):
        raise ValueError('U0/U1 validation identities differ')
    for field in ('protocol_sha256', 'training_manifest_sha256', 'model_manifest_sha256', 'seed',
                  'package_manifest_sha256', 'parent_metric_engine_source_sha256'):
        if a['summary'].get(field) != b['summary'].get(field):
            raise ValueError('U0/U1 comparison provenance differs: ' + field)
    for state in ('zero', 'random'):
        if a['summary']['control_tensor_sha256'][state] != b['summary']['control_tensor_sha256'][state]:
            raise ValueError('common control patch differs: ' + state)
    differences = []
    for key in sorted(a['index']):
        ah, bh = a['hashes'][key], b['hashes'][key]
        fields = ['clean_input_sha256', 'support_tensor_sha256']
        if key[-1] in ('zero', 'random'):
            fields.append('rendered_input_sha256')
        if any(ah[field] != bh[field] for field in fields):
            raise ValueError('supposedly shared clean/control input differs: ' + repr(key))
        ar, br = a['index'][key], b['index'][key]
        fields = ['clean_candidates', 'clean_car_candidates']
        if key[-1] in ('zero', 'random'):
            fields += ['adversarial_candidates', 'adversarial_car_candidates']
        changed = [field for field in fields if ar[field] != br[field]]
        if changed:
            differences.append({'model': key[0], 'dataset': key[1], 'sample_id': key[2],
                                'state': key[3], 'differing_candidate_fields': changed})
    metrics = ('target_vehicle_ap50_clean', 'target_vehicle_ap50_adversarial', 'tnr_det',
               'target_vehicle_ap50_drop', 'car_only_ap50_clean', 'car_only_ap50_adversarial')
    tables, delta = [], []
    for av, bv in zip(a['summaries'] + a['zero_to_final'], b['summaries'] + b['zero_to_final']):
        reference = av.get('reference_state', 'clean')
        state = av.get('evaluated_state', av.get('state'))
        if (av['model'], av['dataset'], state) != (bv['model'], bv['dataset'], bv.get('evaluated_state', bv.get('state'))):
            raise ValueError('summary order differs')
        identity = {'model': av['model'], 'dataset': av['dataset'], 'reference_state': reference, 'evaluated_state': state}
        tables.append(identity | {'U0': av, 'U1': bv})
        delta.append(identity | {'U1_minus_U0': {k: bv[k] - av[k] if av[k] is not None and bv[k] is not None else None for k in metrics},
                                 'undefined_difference_reason': None if av['tnr_defined'] and bv['tnr_defined'] else 'at_least_one_reference_AP50_is_zero'})
    return {'identity': IDENTITY, 'status': 'complete_observations', 'task_variant': 'det-only',
            'comparison': 'fixed_U0_U1_final100_validation_descriptive', 'test_data_opened': False,
            'training_objective_definitions': OBJECTIVE_DESCRIPTIONS,
            'training_objective_models': list(MODELS),
            'formal_test_allowed': False, 'candidate_promoted': False, 'additional_model_forwards': 0,
            'total_existing_validation_forwards': 640, 'total_existing_validation_rows': 480,
            'common_control_inputs_exact': True, 'common_input_hash_rows_compared': 240,
            'shared_control_predictions_exact': not differences,
            'shared_prediction_differences': differences,
            'warnings': ([] if not differences else ['Shared clean/zero/random inputs are identical, but candidate predictions differ. Separate observations retained; no common prediction cache or exact counterfactual claim.']),
            'full_raw_input_scope': 'only first scene of each dataset per evaluation; other scenes have typed input hashes and complete metric candidate lists',
            'tables': tables, 'differences': delta,
            'source_summary_sha256': {'U0': sha(a['path'] / 'summary.json'), 'U1': sha(b['path'] / 'summary.json')},
            'protocol_sha256': a['summary']['protocol_sha256'],
            'claim_boundary': 'No significance, winner selection, baseline long-run, depth/transfer matrix, or test claim. A single fixed seed and final100 per objective.'}


def markdown(report):
    lines = ['# U0/U1 固定 final100 validation 对照', '',
             '两者均为 Ours det-only，40 张 validation（KITTI/nuScenes 各20）；检测 AP50/TNR 沿用父版公式。TNR 越大表示目标车辆 AP50 相对下降越大。', '',
             'U0与U1都使用同一个C3双Contextual损失。U0保留原更新，U1仅对最终待注入向量限幅至0.1767578125，之后不重新归一化。两组均为本轮新训练，不是RGB/Adam baseline。', '',
             '两组训练均针对car类3；评价仍保留原车辆组3/6/8及car-only辅助指标，未把训练标签选择改成新的评价公式。', '',
             '全部数值是各自独立实测；仅作描述性差值，不表示统计显著或已选出正式方法。', '',
             '| 域 | 模型 | 对照 | U0 参考 AP50 | U0 评价 AP50 | U0 TNR | U1 参考 AP50 | U1 评价 AP50 | U1 TNR | ΔTNR (U1−U0) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def number(value, percent=False):
        return '未定义（参考 AP50 为0）' if value is None else (f'{100*value:.2f}%' if percent else f'{value:.6f}')
    for table, difference in zip(report['tables'], report['differences']):
        a, b = table['U0'], table['U1']
        row = [table['dataset'], table['model'], table['reference_state'] + '→' + table['evaluated_state'],
               number(a['target_vehicle_ap50_clean']), number(a['target_vehicle_ap50_adversarial']), number(a['tnr_det'], True),
               number(b['target_vehicle_ap50_clean']), number(b['target_vehicle_ap50_adversarial']), number(b['tnr_det'], True),
               number(difference['U1_minus_U0']['tnr_det'], True)]
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', 'ΔTNR 百分数表示百分点差；JSON 同时保存原 car-only AP50、AP50绝对差及原bootstrap 95%区间。', '',
              'clean→zero、clean→final、zero→final 为三态对照；clean→random 为附加控制。zero→final 由同次既有预测按原公式在CPU派生，不新增前向。', '',
              '全部240渲染输入保留原dtype/shape/byte SHA；完整raw输入仅每域首例。共同控制预测存在差异时保留各自观测，不用缓存伪造一致。']
    lines += ['', *report['warnings']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--u0-evaluation', required=True, type=Path)
    parser.add_argument('--u1-evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    verified = verify_inputs(require_sealed=True)
    load_protocol()
    output = candidate_run_path(args.output)
    sources = [candidate_run_path(args.u0_evaluation), candidate_run_path(args.u1_evaluation)]
    if any(output == p or p in output.parents or output in p.parents for p in sources):
        raise ValueError('comparison output must be separate from evidence')
    if output.exists():
        raise FileExistsError('no comparison output overwrite')
    output.mkdir(parents=True)
    try:
        a, b = [load_evaluation(path, objective) for path, objective in zip(sources, ('U0', 'U1'))]
        if any(x['summary'].get('package_manifest_sha256') != verified['manifest_sha256'] for x in (a, b)):
            raise ValueError('evaluation is not bound to this sealed package')
        report = compare(a, b)
        write_json(output / 'comparison.json', report)
        (output / 'COMPARISON.md').write_text(markdown(report))
        summary = {k: v for k, v in report.items() if k not in ('tables', 'differences', 'shared_prediction_differences')}
        summary.update(shared_prediction_difference_rows=len(report['shared_prediction_differences']),
                       table_rows=len(report['tables']), artifact_bindings=artifact_bindings(output),
                       package_manifest_sha256=verified['manifest_sha256'])
        write_json(output / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    except BaseException as error:
        write_json(output / 'failure.json', {'status': 'failed_retained', 'error': str(error),
                                            'type': type(error).__name__, 'test_data_opened': False})
        raise


if __name__ == '__main__':
    main()
