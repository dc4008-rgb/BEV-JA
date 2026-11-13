#!/usr/bin/env python3
"""Evaluate one actual completed Ours5 ablation on all200 inputs and eight targets."""
import argparse
import json
from pathlib import Path
import random
import resource
import shutil
import signal
import time
from types import SimpleNamespace

from ablation_evaluation_bridge import (METHOD_VARIANTS, collection, load_cases,
                                        module, need, parent_sources)
from eval_contract import bound, descriptor, plain, read, save
from runtime_admission import admission

REQUEST_FIELDS = {'schema', 'method_id', 'training_collection', 'original_selection',
                  'child_manifest', 'parent_v17', 'engine_source', 'metric_source',
                  'model_manifest', 'car', 'car_mask', 'training_manifest', 'weights_root',
                  'source_remap', 'yolov5_model_binding', 'output', 'admission',
                  'dependency_initialization'}


def validate_request(q):
    need(set(q) == REQUEST_FIELDS
         and q['schema'] == 'ours5-b3-nuscenes100-evaluation-request-v1'
         and q['method_id'] in METHOD_VARIANTS, 'closed first-two ablation EVAL request required')
    return q


def install_resource_guard(admitted, output):
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    need(remaining > 0, 'finite admission timer required')
    deadline = time.monotonic() + remaining
    def guard(*_):
        if time.monotonic() >= deadline or time.time() >= admitted['deadline_unix']:
            raise TimeoutError('admitted evaluation window ended')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > admitted['max_RSS_bytes']:
            raise RuntimeError('admitted evaluation RSS exceeded')
        if shutil.disk_usage(output if output.exists() else output.parent).free < admitted['disk_reserve_bytes']:
            raise RuntimeError('admitted evaluation disk reserve reached')
    guard()
    signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, min(1.0, remaining), 1.0)


def run(q, request_descriptor):
    admitted = admission(q)
    out = plain(q['output'])
    need(not out.exists() and out.parent.is_dir(), 'new unused evaluation namespace required')
    from dependency_bootstrap import initialize
    dependency_observation = initialize(q['dependency_initialization'], admitted['deadline_unix'])
    install_resource_guard(admitted, out)
    parent = plain(q['parent_v17'])
    sources = parent_sources(q['child_manifest'], parent)
    for name, relative in (('engine_source', '05_运行/evaluate_road_comparison.py'),
                           ('metric_source', '04_统一评估/evaluator.py'),
                           ('model_manifest', '02_模型/model_manifest.json')):
        need(q[name]['path'] == str(parent / relative) and q[name]['path'] in sources,
             'literal original engine/metric/model manifest required')
        bound(q[name])
    for name in ('car', 'car_mask', 'training_manifest'):
        bound(q[name])
    value = collection(q)
    protected = [parent, plain(q['yolov5_model_binding']['repo_root'])]
    protected.extend(plain(row['summary']['path']).parent for row in value['rows'])
    need(not any(out == path or out.is_relative_to(path) for path in protected),
         'output overlaps original source or completed training')
    engine = module('_ablation_original_road_engine', q['engine_source'])
    import eight_target_library as shared
    cases = load_cases(q, value, engine, shared)
    checks = engine.preflight([case['record'] for case in cases],
        ('fasterrcnn', 'ssd300', 'fcos'), Path(q['model_manifest']['path']),
        Path(q['car']['path']), Path(q['car_mask']['path']),
        None if q['weights_root'] is None else Path(q['weights_root']))
    need(checks['inputs_ready'], 'original RGB/vehicle/detector assets required')
    from road_runtime import checked_file as asset_checked_file
    definitions = {item['id']: item for item in read(q['model_manifest'])['models']}
    depth_checks = []
    for model in shared.DEPTH_MODELS:
        need(definitions[model].get('artifacts'), 'native depth assets required')
        depth_checks.extend({**asset_checked_file(item['path'], item['sha256']), 'model': model}
                            for item in definitions[model]['artifacts'])
    need(all(item['status'] == 'pass' for item in depth_checks), 'all original depth assets required')
    checks['depth'] = depth_checks
    import numpy as np
    import torch
    import torchvision
    from importlib.metadata import version
    need((torch.__version__.split('+')[0], torchvision.__version__.split('+')[0],
          version('transformers')) == ('2.8.0', '0.23.0', '5.7.0'),
         'original evaluation runtime versions required')
    random.seed(shared.SEED)
    np.random.seed(shared.SEED)
    torch.manual_seed(shared.SEED)
    torch.cuda.manual_seed_all(shared.SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metrics = shared.load_metric_functions(q['metric_source']['path'])
    args = SimpleNamespace(output=out, model_manifest=Path(q['model_manifest']['path']),
        car=Path(q['car']['path']), car_mask=Path(q['car_mask']['path']),
        weights_root=None if q['weights_root'] is None else Path(q['weights_root']),
        yolov5_model_binding=q['yolov5_model_binding'])
    out.mkdir()
    (out / 'tensor_pool/blobs').mkdir(parents=True)
    shared.bind_tensor_pool(out / 'tensor_pool')
    save(out / 'ACTUAL_REQUEST.json', q)
    save(out / 'ACTUAL_INPUT_PREFLIGHT.json', checks)
    random_patch = engine.common_random_patch()
    random_path = out / 'random_control_patch.pt'
    engine.save_patch(random_path, random_patch, {'role': 'common_random_control_only', 'seed': shared.SEED})
    random_binding = dict(descriptor(random_path), tensor_sha256=shared.tensor_sha(random_patch),
                          role='common_random_control_only')
    started = time.monotonic()
    try:
        result = shared.evaluate_ablation_cases(args, cases, engine, metrics,
            SimpleNamespace(tensor_sha=shared.tensor_sha), torch.device('cuda:0'),
            random_patch, random_binding)
        pool = shared.collect_pool_bindings(out)
        result.update(actual_request=request_descriptor,
                      training_collection=q['training_collection'],
                      tensor_pool_root=str(out / 'tensor_pool'), tensor_pool_bindings=pool,
                      common_random_patch=random_binding,
                      actual_dependency_initialization=dependency_observation,
                      elapsed_seconds=time.monotonic() - started)
        full = save(out / 'ABLATION_FULL200_ALL8_EVALUATION_FULL.json', result)
        returned = dict(schema='ours5-ablation-full200-evaluation-returned-v1',
            method_id=q['method_id'], ablation_variant=value['ablation_variant'],
            actual_request=request_descriptor, training_collection=q['training_collection'],
            full_result=full, sample_count=100, model_forward_count=3200,
            detection_summary=descriptor(out / 'detection_summary.json'),
            depth_summary=descriptor(out / 'depth_summary.json'),
            per_sample=descriptor(out / 'per_sample.json'),
            depth_per_sample=descriptor(out / 'depth_per_sample.json'),
            detection_manifest=descriptor(out / 'raw_predictions/MANIFEST.json'),
            depth_manifest=descriptor(out / 'depth_raw/MANIFEST.json'),
            raw_prediction_container_count=800, object_mask_container_count=100,
            tensor_blob_count=len(pool), tensor_blob_bytes=sum(item['bytes'] for item in pool.values()),
            global_AP_uses_all100_candidates=True, depth_aggregation='per_image_macro',
            independent_CPU_acceptance_pending=True, new_optimizer_updates=0)
        return save(out / 'ABLATION_FULL200_ALL8_EVALUATION_RETURNED.json', returned)
    except BaseException as error:
        save(out / 'FAILED_RETAINED.json', dict(error_type=type(error).__name__,
            error=str(error), actual_request=request_descriptor,
            elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    request_descriptor = descriptor(args.request)
    need(request_descriptor['sha256'] == args.request_sha256, 'actual request SHA differs')
    request = validate_request(read(request_descriptor))
    if args.execute:
        print(json.dumps(run(request, request_descriptor), sort_keys=True))
    else:
        print(json.dumps({'status': 'METADATA_ONLY', 'request': request_descriptor}))
