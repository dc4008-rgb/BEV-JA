"""Immutable provenance and the fixed v29 C0 U1 nominal-lr comparison contract."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'parent_v17'
IDENTITY = 'normalized-v29-c0-u1-lr005-20260922'
LEARNING_RATE = 0.05
VARIANT_LABEL = 'lr005'
UPDATE_VARIANT = 'U1'
EVALUATION_MODELS = ('fasterrcnn', 'ssd300', 'maskrcnn', 'fcos')
SOURCE_CANDIDATE_ID = 'normalized-v25-c0-clip-20260921'
SOURCE_MANIFEST_SHA = 'bbaab32d1c2c7538a9591970309c02f6c41a3bccbf146193a64b984ed276c2fb'
MATHEMATICAL_OBJECTIVE = 'C0'
PARENT_MANIFEST_SHA = '36ab8bd31d40e56e06c15120baa7751177e41b36eda32ad9aa56748728420352'
LINEAGE_SHA = '947bb042d780e7f704473fe4c6d7a1934305c505ff3a5b2017db218f9332e068'
SD_MANIFEST_SHA = '0c21d98782b4a88f524b9377d779d7143d51fea72f390f068f479caeed36b1dc'
PROTOCOL_PATH = ROOT / 'protocol/training_protocol.json'
CALIBRATION_IDS = (
    'kitti_000007', 'kitti_000046',
    'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404',
    'nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def setup_parent_paths():
    for directory in ('03_方法/production','02_模型','01_公共数据集',
                      '04_统一评估','03_方法/ours/upstream_snapshot'):
        path = str(PARENT / directory)
        if path not in sys.path:
            sys.path.insert(0, path)


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError('module identity collision: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def parent_runner():
    setup_parent_paths()
    return _module('_v29_lr005_parent_training', PARENT / '05_运行/run_ours_stateful_validation.py')


def parent_evaluator():
    setup_parent_paths()
    return _module('_v29_lr005_parent_evaluator', PARENT / '05_运行/evaluate_road_comparison.py')


def candidate_run_path(path, root=ROOT):
    path, allowed = Path(path).resolve(), (Path(root) / 'runs').resolve()
    try:
        relative = path.relative_to(allowed)
    except ValueError as error:
        raise ValueError('output/input run must belong to this v29 candidate') from error
    if not relative.parts:
        raise ValueError('a named run subdirectory is required')
    return path


def protocol_contract():
    return {
        'identity': IDENTITY,
        'scope': 'det_only_C0_U1_nominal_lr_sweep_fixed100',
        'source_identity': SOURCE_CANDIDATE_ID, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
        'source_role': 'read-only implementation provenance; historical lr0.05 results are not the current control',
        'learning_rate_label': VARIANT_LABEL, 'latent_lr': LEARNING_RATE,
        'learning_rate_grid': [0.05, 0.1, 0.2],
        'same_batch_new_control': LEARNING_RATE == 0.05,
        'intervention': 'only nominal latent learning rate changes; fixed cap means realized steps need not scale by 2x or 4x',
        'validation_role': 'already-observed development validation; C0 U1 selected from prior observations, not an untouched holdout',
        'transfer_role': 'Mask R-CNN/FCOS have no training gradients; their validation observations are not an untouched holdout',
        'mathematical_objective': MATHEMATICAL_OBJECTIVE,
        'update_variants': ['U1'],
        'clip_threshold': 0.1767578125,
        'clip_threshold_source': 'reference_analysis/RESULT.json',
        'threshold_reused_without_refit': True,
        'threshold_origin': 'old v23 C3 TRAIN p99; neither C0/C1 nor validation refitted',
        'batch_training_runs': 3,
        'batch_training_before_any_validation': True,
        'clip_rule': 'bf16 original proposed step -> componentwise clamp -> bf16 subtraction; no renormalization',
        'threshold_changes_during_run': False,
        'objectives': ['U1'],
        'candidate_ids': {o: IDENTITY + '-ours-det-only-' + o for o in ('U1',)},
        'task_variant': 'det-only',
        'training_models': ['fasterrcnn','ssd300'],
        'test_access_allowed': False, 'formal_test_allowed': False,
        'transfer_models_allowed': False, 'transfer_evaluation_allowed': True, 'dual_allowed': False,
        'long_run_400_allowed': False, 'resume_enabled': False,
        'automatic_retry': False, 'automatic_candidate_promotion': False,
        'parent_manifest_sha256': PARENT_MANIFEST_SHA,
        'lineage_sha256': LINEAGE_SHA, 'sd35_manifest_sha256': SD_MANIFEST_SHA,
        'stateful_sd35': {
            'num_inference_steps':100,'strength':0.25,'outer_rounds':4,
            'latent_lr':LEARNING_RATE,'step_feedback_mode':'latent_chain',
            'guidance_scale':4.5,'seed':20260919,
            'prompt':'top-down aerial view of clean asphalt road surface',
            'negative_prompt':'vehicle, car, obstacle',
        },
        'dtype':'bf16', 'detector_dtype':'float32',
        'road_eot': {'samples_per_update':2,'brightness':[0.9,1.1],
                     'seed_rule':'seed + 1000 * global_step + draw'},
        'canonical_patch_hw':[64,128],
        'train_schedule': {'updates':100,'seed':20260919,'dataset_counts':{'kitti':50,'nuscenes':50},
                           'rule':'parent seeded_round_robin over frozen train rows',
                           'file':'protocol/train_schedule.json'},
        'calibration': {
            'sample_ids':list(CALIBRATION_IDS),'split':'train','dataset_counts':{'kitti':2,'nuscenes':2},
            'initialization':'common unoptimized road crop from first frozen calibration train record',
            'eot':False,'rule':'mean(abs(raw objective loss)) per model, fixed before optimization',
            'floor':1e-6,'scale_reused_from_v20':False,
            'normalizer_separately_computed_for_each_objective':True,
        },
        'objective_definitions': {'U1': 'Both detectors retain original post-NMS max(vehicle score * target IoU), vehicle classes 3/6/8.'},
        'training_target_classes': [3,6,8],
        'loss_source': 'frozen v21 C0/C1 dispatch; C1 changes SSD only',
        'victim_counter_semantics': 'original predict callbacks including native postprocessing; C1 uses passive SSD hooks on same predict',
        'gradient_control':'unchanged parent L2-unit equal mean plus RMS; U1 additionally clips final proposed injection',
        'near_zero_epsilon':1e-8,
        'checkpoint_steps':[25,50,75,100],
        'selection':'fixed final100 only; no best checkpoint, seed or validation filtering',
        'full_state_evidence':True,
        'completed_run_state_files':409,
        'state_phase_counts':{'initialization':1,'outer_start':4,'before_denoise':100,
                              'gradients':100,'pre_update':100,'post_update':100,'outer_end':4},
        'state_evidence_role':'diagnostic reconstruction and arithmetic audit; not certified exact CUDA resume',
        'budget_per_run': {
            'calibration_model_forwards':8,'optimization_pre_model_forwards':400,
            'optimization_post_model_forwards':400,'victim_model_forwards_total':808,
            'autograd_calls':200,'transformer_forward_calls':100,
            'vae_encode_calls':5,'vae_decode_calls':205,'noise_draws':4,
        },
        'evaluation': {'split':'validation','scenes':40,'dataset_counts':{'kitti':20,'nuscenes':20},
                       'models':list(EVALUATION_MODELS),'states':['zero','random','final'],
                       'reference':'clean','model_forwards_per_run':640,'rows_per_run':480,'summary_rows_per_run':48,
                       'bootstrap_resamples':1000,'seed':20260919,
                       'metric':'parent target-vehicle AP50 relative reduction TNR; no metric changes',
                       'all_training_before_evaluation':True},
        'stage_order':['train_U1','validation_U1'],
        'stage_timeouts_seconds':{'train':900,'validation':900,'comparison':180},
        'minimum_start_free_disk_bytes':6*1024**3,
        'minimum_stage_free_disk_bytes':2*1024**3,
        'required_torch_base_version':'2.8.0','required_torchvision_base_version':'0.23.0',
        'required_diffusers_base_version':'0.38.0',
        'deterministic_algorithms':False,
        'historical_v17_exact_replay_required':False,
        'v18_replay_difference_and_v19_strict_failure_preserved':True,
        'U1_nonzero_gradient_does_not_establish_attack_improvement':True,
        'runtime_gate':'bound hostname and input SHA; original assets and complete SD snapshot; empty GPU',
    }


def load_protocol():
    data = json.loads(PROTOCOL_PATH.read_text())
    # JSON representation distinguishes bool/int/float, unlike Python equality.
    if json.dumps(data,sort_keys=True) != json.dumps(protocol_contract(),sort_keys=True):
        raise ValueError('v29 fixed protocol differs')
    return data


def _safe_file(base, name):
    base, name = Path(base), Path(name)
    if name.is_absolute() or not name.parts or '..' in name.parts:
        raise ValueError('unsafe manifest path')
    for index in range(len(name.parts)+1):
        if (base/Path(*name.parts[:index])).is_symlink():
            raise ValueError('symlink input is not permitted')
    return base/name


def _verify_entries(base, entries):
    seen = set()
    for entry in entries:
        name = entry['path']
        if name in seen:
            raise ValueError('duplicate manifest entry')
        seen.add(name)
        path = _safe_file(base, name)
        if not path.is_file() or (entry.get('bytes') is not None and path.stat().st_size != entry['bytes']) or sha(path) != entry['sha256']:
            raise ValueError('input artifact differs: '+name)
    return seen


def verify_inputs(require_sealed=True):
    if sha(PARENT/'PACKAGE_MANIFEST.json') != PARENT_MANIFEST_SHA:
        raise ValueError('parent manifest differs')
    parent = json.loads((PARENT/'PACKAGE_MANIFEST.json').read_text())
    names = _verify_entries(PARENT,parent['files'])
    actual = {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}
    if actual != names | {'PACKAGE_MANIFEST.json'} or len(names) != 63:
        raise ValueError('parent 64-file snapshot differs')
    if sha(ROOT/'protocol/lineage.json') != LINEAGE_SHA:
        raise ValueError('lineage differs')
    lineage = json.loads((ROOT/'protocol/lineage.json').read_text())
    _verify_entries(ROOT,lineage['vendor_v20_files']+lineage['reference_files'])
    protocol = load_protocol()
    setup_parent_paths()
    from data import calibration_records, seeded_round_robin
    from road_runtime import training_records
    records = training_records(PARENT/'01_公共数据集/runtime_v1/train_validation_manifest.csv')
    expected = [{'sample_id':r.sample_id,'dataset':r.dataset,'split':r.split}
                for r in seeded_round_robin(records,100,20260919)]
    actual_schedule = json.loads((ROOT/'protocol/train_schedule.json').read_text())
    if actual_schedule != expected or any(r['split']!='train' for r in actual_schedule):
        raise ValueError('train schedule differs')
    if [r.sample_id for r in calibration_records(records)] != list(CALIBRATION_IDS):
        raise ValueError('calibration records differ')
    result = {'identity':IDENTITY,'parent_files_verified':64,'vendor_files_verified':1,
              'reference_files_verified':len(lineage['reference_files']),'protocol_sha256':sha(PROTOCOL_PATH),
              'schedule_sha256':sha(ROOT/'protocol/train_schedule.json'),
              'training_updates_per_run':100,'scope':protocol['scope'],
              'source_images_opened':False,'models_loaded':False,'sealed':False,
              'latent_lr':LEARNING_RATE,'learning_rate_label':VARIANT_LABEL,'update_variant':UPDATE_VARIANT}
    manifest_path = ROOT/'PACKAGE_MANIFEST.json'
    if require_sealed or manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('identity') != IDENTITY or manifest.get('formal_test_allowed') is not False:
            raise ValueError('package identity differs')
        files = _verify_entries(ROOT,manifest['files'])
        # Runtime evidence is written only under runs, and is not signed input.
        actual_inputs = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
                         and p.relative_to(ROOT).parts[0] != 'runs'}
        if actual_inputs != files | {'PACKAGE_MANIFEST.json'}:
            raise ValueError('signed input file set differs')
        result.update(sealed=True,manifest_sha256=sha(manifest_path),payload_files=len(files))
    return result
