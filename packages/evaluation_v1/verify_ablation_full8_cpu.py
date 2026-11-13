#!/usr/bin/env python3
"""CPU reconstruction of stored full8 predictions, global scalars and retention."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import stat
import time

from ablation_evaluation_bridge import module, need
from eval_contract import descriptor, plain, read, save


def checked_admission(q):
    admitted = q['admission']
    need(set(admitted) == {'expected_hostname', 'expected_boot_id', 'deadline_unix',
                           'wall_seconds', 'max_RSS_bytes', 'disk_reserve_bytes'},
         'closed finite CPU admission required')
    need(socket.gethostname() == admitted['expected_hostname']
         and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == admitted['expected_boot_id'],
         'actual CPU host/boot differs')
    for key in ('deadline_unix', 'wall_seconds', 'max_RSS_bytes', 'disk_reserve_bytes'):
        need(type(admitted[key]) in (int, float) and math.isfinite(admitted[key])
             and admitted[key] > 0, 'finite positive CPU bound required')
    remaining = min(admitted['wall_seconds'], admitted['deadline_unix'] - time.time())
    need(0 < remaining <= 12 * 3600 and admitted['disk_reserve_bytes'] >= 1024 ** 3,
         'current finite CPU window and reserve required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU deadline')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    return admitted


def retain_opaque(binding, guard):
    path = plain(binding['path'])
    before = path.lstat()
    fields = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                            value.st_uid, value.st_gid, value.st_size,
                            value.st_mtime_ns, value.st_ctime_ns)
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
         'private regular native file required')
    hashes = []
    for _ in range(2):
        guard()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            need(fields(os.fstat(stream.fileno())) == fields(before), 'native FD identity differs')
            for block in iter(lambda: stream.read(64 * 1024), b''):
                digest.update(block)
            os.fsync(stream.fileno())
            need(fields(os.fstat(stream.fileno())) == fields(before), 'native file changed while retained')
        hashes.append(digest.hexdigest())
    need(fields(path.lstat()) == fields(before)
         and hashes[0] == hashes[1] == binding['sha256']
         and before.st_size == binding['bytes'], 'native full2H9 binding differs')
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(binding, full9=list(fields(before)), file_and_parent_fsync=True)


def verify(q, request_descriptor):
    admitted = checked_admission(q)
    out = plain(q['output'])
    need(not out.exists() and out.parent.is_dir(), 'new unused CPU output required')
    def guard():
        need(time.time() < admitted['deadline_unix'], 'CPU deadline passed')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= admitted['max_RSS_bytes'],
             'CPU RSS bound exceeded')
        need(shutil.disk_usage(out.parent).free >= admitted['disk_reserve_bytes'],
             'CPU filesystem reserve reached')
    guard()
    eq = read(q['evaluation_request'])
    returned = read(q['evaluation_result'])
    need(returned['actual_request'] == q['evaluation_request']
         and returned['method_id'] == eq['method_id']
         and returned['sample_count'] == 200 and returned['model_forward_count'] == 6400,
         'actual completed full8 evaluation required')
    root = plain(eq['output'])
    need(not out.is_relative_to(root) and not root.is_relative_to(out),
         'independent CPU output must be outside original EVAL tree')
    full = read(returned['full_result'], cap=32 * 1024 ** 2)
    need(full['actual_request'] == q['evaluation_request'] and full['method_id'] == eq['method_id']
         and full['training_collection'] == eq['training_collection'], 'full result origin differs')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(), 'CPU verification must not initialize CUDA')
    engine = module('_ablation_CPU_original_road_engine', eq['engine_source'])
    import eight_target_library as shared
    shared.bind_tensor_pool(root / 'tensor_pool')
    metrics = shared.load_metric_functions(eq['metric_source']['path'])
    detection_rows = read(returned['per_sample'], cap=32 * 1024 ** 2)
    depth_rows = read(returned['depth_per_sample'], cap=32 * 1024 ** 2)
    detection_manifest = read(returned['detection_manifest'])
    depth_manifest = read(returned['depth_manifest'])
    collection = read(eq['training_collection'])
    samples = collection['rows']
    ordered_ids = [row['sample']['sample_id'] for row in samples]
    need(len(ordered_ids) == len(set(ordered_ids)) == 200,
         'CPU requires the same unique original200 collection')
    need(len(detection_rows) == 2400 and len(depth_rows) == 3200,
         'complete detector/depth candidate row counts required')
    det_by_key = {(row['model'], row['sample_id'], row['state']): row for row in detection_rows}
    dep_by_key = {(row['model'], row['sample_id'], row['reference_state'], row['evaluated_state']): row
                  for row in depth_rows}
    need(len(det_by_key) == 2400 and len(dep_by_key) == 3200,
         'duplicate candidate/comparison row')
    expected_ids = set(ordered_ids)
    for model in shared.DETECTION_MODELS:
        ids = [entry['sample_id'] for entry in detection_manifest['entries'] if entry['model'] == model]
        need(len(ids) == 200 and set(ids) == expected_ids, 'exact200 native detector cases required')
        for state in shared.STATES[1:]:
            need({row['sample_id'] for row in detection_rows
                  if row['model'] == model and row['state'] == state} == expected_ids,
                 'exact200 detector cases in every state required')
    for model in shared.DEPTH_MODELS:
        ids = [entry['sample_id'] for entry in depth_manifest['entries'] if entry['model'] == model]
        need(len(ids) == 200 and set(ids) == expected_ids, 'exact200 native depth cases required')
        for reference, evaluated in shared.COMPARISONS:
            need({row['sample_id'] for row in depth_rows
                  if row['model'] == model and row['reference_state'] == reference
                  and row['evaluated_state'] == evaluated} == expected_ids,
                 'exact200 depth cases in every comparison required')
    for rows in (detection_rows, depth_rows):
        for row in rows:
            need(row['method'] == eq['method_id'] and row['method_id'] == eq['method_id'],
                 'stored scientific row method differs')
            expected = row['model'] in shared.role_metadata(row)['source_models']
            need(row['used_in_training'] is expected, 'actual four-source model role differs')
    products = {}
    counts = Counter()
    for entry in detection_manifest['entries']:
        guard()
        path = shared.checked_file(root, entry['path'])
        binding = descriptor(path)
        need(binding['sha256'] == entry['sha256'], 'detection native descriptor differs')
        products[str(path)] = binding
        payload = shared.load_raw(path)
        need(payload['sample_id'] == entry['sample_id'] and payload['method_id'] == eq['method_id'],
             'native detector case origin differs')
        clean = engine.detection_candidates(payload['clean'], payload['target_box'], engine.VEHICLE_CLASS_IDS)
        clean_car = engine.detection_candidates(payload['clean'], payload['target_box'], engine.CAR_CLASS_IDS)
        for state in shared.STATES:
            actual_hash = {key: shared.tensor_sha(value) for key, value in payload[state].items()}
            need(actual_hash == entry['tensor_sha256'][state], 'native detector tensor differs')
            counts[entry['model']] += 1
            if state == 'clean':
                continue
            row = det_by_key[entry['model'], entry['sample_id'], state]
            need(row['clean_candidates'] == clean and row['clean_car_candidates'] == clean_car,
                 'native clean candidates differ')
            need(row['adversarial_candidates'] == engine.detection_candidates(
                     payload[state], payload['target_box'], engine.VEHICLE_CLASS_IDS)
                 and row['adversarial_car_candidates'] == engine.detection_candidates(
                     payload[state], payload['target_box'], engine.CAR_CLASS_IDS),
                 'native adversarial candidates differ')
    for entry in depth_manifest['entries']:
        guard()
        path = shared.checked_file(root, entry['path'])
        binding = descriptor(path)
        need(binding['sha256'] == entry['sha256'], 'depth native descriptor differs')
        products[str(path)] = binding
        payload = shared.load_raw(path)
        need(payload['sample_id'] == entry['sample_id'] and payload['method_id'] == eq['method_id'],
             'native depth case origin differs')
        mask_binding = entry['object_mask']
        mask_path = shared.checked_file(root, mask_binding['path'])
        actual_mask = descriptor(mask_path)
        need(actual_mask['sha256'] == mask_binding['sha256'], 'native object mask descriptor differs')
        products[str(mask_path)] = actual_mask
        mask = shared.load_raw(mask_path)
        need(shared.tensor_sha(mask) == mask_binding['tensor_sha256'], 'native object mask differs')
        for state in shared.STATES:
            need(shared.tensor_sha(payload['states'][state]) == entry['tensor_sha256'][state],
                 'native depth tensor differs')
            counts[entry['model']] += 1
        for reference, evaluated in shared.COMPARISONS:
            actual = metrics.depth_metrics(payload['states'][evaluated], payload['states'][reference],
                                          mask, semantics=shared.SEMANTICS[entry['model']])
            row = dep_by_key[entry['model'], entry['sample_id'], reference, evaluated]
            need(all(key in row and row[key] == value for key, value in actual.items()),
                 'native depth metric reconstruction differs')
    need(counts == Counter(dict.fromkeys(shared.DETECTION_MODELS + shared.DEPTH_MODELS, 800)),
         'native retained coverage must be6400 states over the eight models')
    need(len(products) == 1800, 'all1600 prediction containers and200 object masks required')
    detection = read(returned['detection_summary'])
    depth = read(returned['depth_summary'])
    need(shared.summarize_detection(metrics, detection_rows) == detection['summaries'] == full['detection']['summaries'],
         'all48 global detection summaries must match exactly')
    need(shared.summarize_depth(metrics, depth_rows) == depth['summaries'] == full['depth']['summaries'],
         'all48 macro depth summaries must match exactly')
    pool = shared.collect_pool_bindings(root)
    need(pool == full['tensor_pool_bindings'], 'complete native CAS binding set differs')
    for binding in pool.values():
        path = root / 'tensor_pool' / binding['path']
        products[str(path)] = {key: binding[key] for key in ('bytes', 'sha256')}
        products[str(path)]['path'] = str(path)
    random_binding = {key: full['common_random_patch'][key] for key in ('path', 'bytes', 'sha256')}
    products[random_binding['path']] = random_binding
    need(len(products) <= 16384, 'finite declared native/CAS catalogue required')
    retained = [retain_opaque(products[path], guard) for path in sorted(products)]
    need(not torch.cuda.is_initialized(), 'CPU verification initialized CUDA')
    out.mkdir()
    catalogue = save(out / 'NATIVE_FULL2H9_RETENTION_CATALOGUE.json',
                     dict(native_products=retained, tensor_blob_count=len(pool)))
    verified = dict(schema='ours5-ablation-full200-independent-CPU-verified-v1',
        accepted=True, method_id=eq['method_id'], sample_count=200,
        actual_request=request_descriptor, evaluation_result=q['evaluation_result'],
        training_collection=eq['training_collection'], native_prediction_state_count=6400,
        detection_summary_count=48, depth_summary_count=48,
        all_summaries_exact_JSON_equal=True, native_candidates_reconstructed=True,
        native_depth_metrics_reconstructed=True,
        native_prediction_container_count=1800, tensor_blob_count=len(pool),
        retained_file_count=len(retained), retained_bytes=sum(item['bytes'] for item in retained),
        native_retention_catalogue=catalogue, model_forward_count=0, optimizer_updates=0)
    result = save(out / 'ABLATION_FULL200_CPU_VERIFIED.json', verified)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    request_descriptor = descriptor(args.request)
    need(request_descriptor['sha256'] == args.request_sha256, 'actual CPU request SHA differs')
    request = read(request_descriptor)
    need(set(request) == {'schema', 'evaluation_request', 'evaluation_result', 'output', 'admission'}
         and request['schema'] == 'ours5-ablation-full200-independent-CPU-request-v1',
         'closed independent CPU request required')
    if args.execute:
        print(json.dumps(verify(request, request_descriptor), sort_keys=True))
    else:
        print(json.dumps({'status': 'METADATA_ONLY', 'request': request_descriptor}))
