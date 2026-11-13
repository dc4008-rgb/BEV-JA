"""Bindings between actual completed ablation cases and the original evaluator."""
import importlib.util
import io
from pathlib import Path
import sys
import time

from eval_contract import bound, descriptor, plain, read

METHOD_VARIANTS = {'ours5_b3_full_nuscenes100': 'custom'}
MASTER_SHA256 = 'eb4fa9bf9d394d90ed2b764de3d5ca54f26f2d0d3072050e8f9086ee522458f8'
SELECTION_SHA256 = '3c8432679cd4926ec8dea6e4e8500ca1a95786f43cfe23155d2f53a1c730680b'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, binding):
    data = bound(binding, cap=32 * 1024 ** 2)
    spec = importlib.util.spec_from_file_location(name, binding['path'])
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    exec(compile(data, binding['path'], 'exec'), value.__dict__)
    return value


def parent_sources(binding, root):
    manifest = read(binding, cap=2 * 1024 ** 2)
    files = manifest['files']
    need(len(files) <= 4096, 'finite original child source manifest required')
    allowed = set()
    for item in files:
        if item['path'].startswith('parent_v17/'):
            path = root.parent / item['path']
            bound(dict(item, path=str(path)), cap=32 * 1024 ** 2)
            allowed.add(str(path))
    for relative in ('05_运行/evaluate_road_comparison.py',
                     '03_方法/production/road_runtime.py',
                     '03_方法/production/road_patch_contract.py',
                     '04_统一评估/evaluator.py'):
        need(str(root / relative) in allowed, 'original road/ROI/metric source missing')
    return allowed


def collection(q):
    value = read(q['training_collection'])
    need(value['schema'] == 'ours5-b3-nuscenes100-training-collection-v1'
         and value['method_id'] == q['method_id']
         and value['ablation_variant'] == METHOD_VARIANTS[q['method_id']]
         and value['sample_count'] == 100 and value['updates_per_sample'] == 125
         and value['effective_training_updates'] == 12500,
         'actual nuScenes100 completed ablation collection required')
    need(value['master_reference']['sha256'] == MASTER_SHA256,
         'original master reference required')
    master = read(value['master_reference'])
    rows = value['rows']
    need([row['sample'] for row in rows] == [s for s in master['master_samples'] if s['dataset'] == 'nuscenes'],
         'actual collection must follow exact original200 master order')
    need(q['original_selection']['sha256'] == SELECTION_SHA256,
         'exact original selection required')
    selected = [s for s in read(q['original_selection'])['samples'] if s['dataset'] == 'nuscenes']
    fields = ('sample_id', 'case_id', 'dataset', 'source_split',
              'source_sha256', 'road_quad_norm')
    need(len(selected) == 100, 'nuScenes selection has100 cases')
    for sample, row in zip(selected, rows):
        need(all(sample[key] == row['sample'][key] for key in fields),
             'original selection/master case differs')
    return value


def load_cases(q, value, engine, shared):
    import torch
    mapping = q['source_remap']
    need(mapping is None or set(mapping) == {'source_prefix', 'source_root'},
         'explicit source remap required')
    records = engine.training_records(
        Path(q['training_manifest']['path']),
        None if mapping is None else Path(mapping['source_prefix']),
        None if mapping is None else Path(mapping['source_root']))
    by_id = {record.sample_id: record for record in records}
    cases = []
    for row in value['rows']:
        sample = row['sample']
        returned = read(row['returned'])
        summary = read(row['summary'])
        need(returned['schema'] == 'ours5-ablation-case-returned-v1'
             and returned['method_id'] == q['method_id']
             and returned['ablation_variant'] == value['ablation_variant']
             and returned['sample'] == sample and returned['summary'] == row['summary']
             and returned['executed_updates'] == 125
             and summary['executed_updates'] == 125
             and summary['ablation_variant'] == value['ablation_variant']
             and summary['patch_bindings'] == returned['patch_bindings'],
             'case RETURN, summary and declared variant must agree')
        record = by_id[sample['sample_id']]
        need((record.dataset, record.split, record.source_sha256, record.road_quad_norm) ==
             (sample['dataset'], sample['source_split'], sample['source_sha256'], sample['road_quad_norm']),
             'original source record and geometry differ')
        need(descriptor(record.source_path)['sha256'] == sample['source_sha256'],
             'actual pristine RGB differs')
        patches, patch_bindings = {}, {}
        for state, name in (('zero', 'zero_step_patch.pt'), ('final', 'final_patch.pt')):
            path = plain(row['summary']['path']).parent / name
            binding = dict(summary['patch_bindings'][name], path=str(path))
            raw = bound(binding, cap=32 * 1024 ** 2)
            payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            need(payload['metadata'] == returned['native_metadata_expected'][state],
                 'exact native metadata differs from actual case RETURN')
            patch = engine.load_patch(path)
            need(patch.dtype == torch.float32 and patch.shape == (1, 3, 64, 128)
                 and bool(torch.isfinite(patch).all())
                 and bool(((patch >= 0) & (patch <= 1)).all()),
                 'canonical own-input patch required')
            need(descriptor(path) == binding, 'native patch changed during decoding')
            patches[state] = patch
            patch_bindings[state] = dict(binding, tensor_sha256=shared.tensor_sha(patch))
        binding = dict(sample, method_id=q['method_id'],
                       ablation_variant=value['ablation_variant'], updates_per_sample=125,
                       actual_case_return=row['returned'], actual_training_summary=row['summary'],
                       actual_core_measurement=row['actual_measurement'],
                       actual_total_measurement=row['actual_total_measurement'],
                       patches=patch_bindings)
        cases.append(dict(sample=sample, record=record, patches=patches, binding=binding))
    return cases


def make_cpu_request(eval_request_descriptor, eval_result_descriptor, output_directory,
                     deadline_unix, max_RSS_bytes, disk_reserve_bytes):
    q = read(eval_request_descriptor)
    returned = read(eval_result_descriptor)
    need(returned['actual_request'] == eval_request_descriptor
         and returned['method_id'] == q['method_id']
         and returned['sample_count'] == 100 and returned['model_forward_count'] == 3200,
         'actual complete EVAL must precede CPU request')
    remaining = deadline_unix - time.time()
    need(0 < remaining <= 12 * 3600, 'finite current CPU window required')
    admission = {key: q['admission'][key] for key in ('expected_hostname', 'expected_boot_id')}
    admission.update(deadline_unix=deadline_unix, wall_seconds=remaining,
                     max_RSS_bytes=max_RSS_bytes, disk_reserve_bytes=disk_reserve_bytes)
    return dict(schema='ours5-b3-nuscenes100-independent-CPU-request-v1',
                evaluation_request=eval_request_descriptor, evaluation_result=eval_result_descriptor,
                output=str(plain(output_directory)), admission=admission)
