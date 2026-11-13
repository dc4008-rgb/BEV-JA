"""Offline, standard (never ShakeDrop) YOLOv5s target for the old scene canvas.

The repository/weights must already exist and have independent literal source
bindings. No torch.hub, attempt_load, auto-install, download or AutoShape is
used. This module makes no claim that the old T-SEA fork is official v6.2.
"""
from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys

from eval_contract import bound, descriptor, plain, read

WEIGHT_SHA256 = '8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee'
OFFICIAL_V62_COMMIT = 'd3ea0df8b9f923685ce5f2555c303b8eddbf83fd'
OFFICIAL_V62_YOLO_SHA = '9d4881f96be447a85974931cfc5bec2b9e82994a92a413892de2cd8df1e943db'
OFFICIAL_V62_GENERAL_SHA = 'aa0fc07703e70423a66549f74a3e29ef7a1342cefe5178e2f1c1a614748b18ab'
OFFICIAL_STANDARD_YAML_SHA = 'd17de90c1a69a54d95b56d710d3688557b1ca7817558931630c6c7052c47f258'
INPUT_HW = (320, 1024)
CONFIDENCE_THRESHOLD = 0.0001
NMS_IOU = 0.6
MAX_DETECTIONS = 1000
# Standard COCO80 contiguous index -> original COCO category id. In particular
# car/bus/truck = 2/5/7 -> 3/6/8; class0/person never aliases car.
COCO80_TO91 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
              37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
              53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
              88, 89, 90)


@contextmanager
def offline(config_directory=None):
    """Fail closed if an imported model helper tries a network fallback."""
    if config_directory is not None:
        config_directory = plain(config_directory)
        if not config_directory.is_dir():
            raise ValueError('existing new operation config directory required')
    original = socket.socket
    original_connect = socket.create_connection
    original_dns = socket.getaddrinfo
    original_popen = subprocess.Popen
    def blocked(*args, **kwargs):
        raise RuntimeError('network access forbidden inside offline target loader')
    class OfflineSocket(original):
        connect = blocked
        connect_ex = blocked
    socket.socket = OfflineSocket
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    subprocess.Popen = blocked
    old = os.environ.get('YOLOv5_AUTOINSTALL')
    old_config = {k: os.environ.get(k) for k in ('YOLOV5_CONFIG_DIR', 'MPLCONFIGDIR')}
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    if config_directory is not None:
        os.environ['YOLOV5_CONFIG_DIR'] = str(config_directory)
        os.environ['MPLCONFIGDIR'] = str(config_directory)
    try:
        yield
    finally:
        socket.socket = original
        socket.create_connection = original_connect
        socket.getaddrinfo = original_dns
        subprocess.Popen = original_popen
        if old is None:
            os.environ.pop('YOLOv5_AUTOINSTALL', None)
        else:
            os.environ['YOLOv5_AUTOINSTALL'] = old
        for k, value in old_config.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def validate_standard_yaml(value):
    if (value.get('nc') != 80 or value.get('depth_multiple') != .33
            or value.get('width_multiple') != .5):
        raise ValueError('standard COCO YOLOv5s80 architecture required')
    rows = value.get('backbone', []) + value.get('head', [])
    allowed = {'Conv', 'C3', 'SPPF', 'nn.Upsample', 'Concat', 'Detect'}
    if len(rows) != 25 or any(r[2] not in allowed for r in rows):
        raise ValueError('standard graph only; no ShakeDrop/hook/custom layers')
    if rows[-1][2] != 'Detect' or rows[-1][0] != [17, 20, 23]:
        raise ValueError('standard three-scale head required')
    return value


def remap_prediction(prediction):
    import torch
    if (prediction.ndim != 2 or prediction.shape[1] != 6
            or not torch.isfinite(prediction).all()):
        raise ValueError('finite post-NMS xyxy/conf/COCO80 rows required')
    native = prediction[:, 5]
    if not torch.equal(native, native.round()) or ((native < 0) | (native >= 80)).any():
        raise ValueError('unknown YOLO class id')
    mapping = torch.tensor(COCO80_TO91, device=native.device, dtype=torch.int64)
    labels80 = native.to(torch.int64)
    boxes = prediction[:, :4].clone()
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, INPUT_HW[1])
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, INPUT_HW[0])
    return {'boxes': boxes, 'scores': prediction[:, 4].clone(),
            'labels': mapping[labels80], 'native_COCO80_labels': labels80}


def construct_with_observed_initialization(factory, cfg):
    """Observe upstream stride/FLOP forwards before any checkpoint state load.

    An ordinary Module pre-hook misses the constructor's direct self.forward
    call. This temporary class method wrapper passes inputs/output untouched,
    consumes no RNG and calls the original exactly once. It is restored before
    loading pretrained state or any benchmark scene forward.
    """
    original = factory.forward
    observed = []
    overflow = False
    def counted(instance, image, *args, **kwargs):
        nonlocal overflow
        if len(observed) >= 16:
            overflow = True
            raise RuntimeError('unexpected unbounded constructor probe count')
        row = {'input_shape': list(image.shape), 'device': str(image.device),
               'dtype': str(image.dtype), 'checkpoint_state_loaded': False,
               'self_type': type(instance).__module__ + '.' + type(instance).__name__,
               'scope': 'UNTRAINED_FACTORY_INITIALIZATION_NOT_BENCHMARK_SCENE',
               'returned': False}
        observed.append(row)
        try:
            value = original(instance, image, *args, **kwargs)
        except BaseException as error:
            row['error_type'] = type(error).__name__
            row['error'] = str(error)
            raise
        row['returned'] = True
        return value
    factory.forward = counted
    try:
        model = factory(cfg, ch=3, nc=80).float()
    finally:
        factory.forward = original
    if overflow or not observed or not observed[0]['returned']:
        raise ValueError('constructor stride forward or complete counter failed')
    return model, {'scope': 'UNTRAINED_FACTORY_INITIALIZATION_ONLY',
        'forward_invocation_count': len(observed), 'forwards': observed,
        'factory_returned_normally': True,
        'counter_wrapper_restored_before_checkpoint_load': factory.forward is original,
        'not_training_or_attack_efficacy_or_benchmark_scene_calls': True}


def load_local_target(model_binding, device, runtime_config_directory):
    """Load a ROOT-observed, reviewed v6.2 repository and exact existing weight.

    Source binding has repo_root, official_tag='v6.2', git_commit, files[],
    architecture_yaml descriptor and checkpoint descriptor. The source review
    itself is also bound. Tag/commit is not inferred from a filename.
    """
    import torch
    import yaml
    expected = {'repo_root', 'official_tag', 'git_commit', 'source_review', 'files',
                'architecture_yaml', 'checkpoint', 'checkpoint_origin_proof'}
    if set(model_binding) != expected or model_binding['official_tag'] != 'v6.2':
        raise ValueError('explicit independently reviewed v6.2 source binding required')
    commit = model_binding['git_commit']
    if commit != OFFICIAL_V62_COMMIT:
        raise ValueError('exact officialv6.2 release commit required')
    read(model_binding['source_review'], cap=2 * 1024 ** 2)
    read(model_binding['checkpoint_origin_proof'], cap=2 * 1024 ** 2)
    root = plain(model_binding['repo_root'])
    files = model_binding['files']
    if not 1 <= len(files) <= 4096 or sum(d['bytes'] for d in files) > 64 * 1024 ** 2:
        raise ValueError('finite complete source closure required')
    allowed_paths = set()
    for d in files:
        p = plain(d['path'])
        p.relative_to(root)
        bound(d, cap=2 * 1024 ** 2)
        allowed_paths.add(str(p))
    by_path = {d['path']: d for d in files}
    for relative, sha in [('models/yolo.py', OFFICIAL_V62_YOLO_SHA),
                          ('utils/general.py', OFFICIAL_V62_GENERAL_SHA)]:
        if by_path.get(str(root / relative), {}).get('sha256') != sha:
            raise ValueError('original officialv6.2 model/NMS source differs')
    yaml_d = model_binding['architecture_yaml']
    if yaml_d['path'] not in allowed_paths:
        raise ValueError('architecture not in reviewed source closure')
    if yaml_d['sha256'] != OFFICIAL_STANDARD_YAML_SHA:
        raise ValueError('ordinary official YOLOv5s YAML required, not SD YAML')
    cfg = validate_standard_yaml(yaml.safe_load(bound(yaml_d)))
    weight = model_binding['checkpoint']
    if weight['sha256'] != WEIGHT_SHA256 or not 1024 ** 2 < weight['bytes'] < 32 * 1024 ** 2:
        raise ValueError('only exact existing source-bound COCO checkpoint is admitted')
    if plain(weight['path']).stat().st_size != weight['bytes']:
        raise ValueError('checkpoint actual size differs before any model construction')
    before = descriptor(weight['path'])
    if before != weight:
        raise ValueError('checkpoint whole bytes differ')
    # Fresh evaluation process only: do not replace a live trainer's modules.
    if any(n == 'models' or n.startswith('models.') or n == 'utils' or n.startswith('utils.')
           for n in sys.modules):
        raise RuntimeError('fresh process required for standard YOLO module namespace')
    sys.path.insert(0, str(root))
    with offline(runtime_config_directory):
        module = importlib.import_module('models.yolo')
        general = importlib.import_module('utils.general')
        factory = getattr(module, 'Model', None) or module.DetectionModel
        model, initialization = construct_with_observed_initialization(factory, cfg)
        # This is the one explicitly trusted whole-SHA legacy checkpoint. It
        # contains a model object, so weights_only=False is required, disclosed,
        # and never used for an arbitrary unbound sample file.
        checkpoint = torch.load(weight['path'], map_location='cpu', weights_only=False)
        saved = checkpoint.get('ema') if checkpoint.get('ema') is not None else checkpoint['model']
        if any('shakedrop' in type(m).__name__.lower() for m in saved.modules()):
            raise ValueError('checkpoint is a perturbed model, not standard COCO weights')
        validate_standard_yaml(saved.yaml)
        names = saved.names
        if isinstance(names, dict):
            names = [names[i] for i in range(80)]
        if len(names) != 80 or [names[i] for i in (2, 5, 7)] != ['car', 'bus', 'truck']:
            raise ValueError('actual checkpoint COCO80 vehicle class mapping differs')
        model.load_state_dict(saved.float().state_dict(), strict=True)
        model.names = list(names)
    if descriptor(weight['path']) != before:
        raise ValueError('checkpoint changed during strict state load')
    for name, mod in tuple(sys.modules.items()):
        if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
            p = getattr(mod, '__file__', None)
            if p and str(plain(p)) not in allowed_paths:
                raise ValueError('imported YOLO source outside exact reviewed closure: ' + str(p))
    names = [type(m).__name__.lower() for m in model.modules()]
    if any('shakedrop' in n or 'dropout' in n for n in names):
        raise ValueError('standard target may not contain stochastic perturbation modules')
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('unregistered model forward hooks forbidden')
    if int(model.stride.max()) != 32 or model.model[-1].nc != 80:
        raise ValueError('standard YOLOv5s stride32/COCO80 head required')
    model = model.to(device).float().eval().requires_grad_(False)
    return StandardTarget(model, general.non_max_suppression, model_binding, initialization)


class StandardTarget:
    def __init__(self, model, nms, binding, initialization):
        self.model, self.nms, self.binding = model, nms, binding
        self.initialization_observation = initialization

    def predict(self, image):
        import torch
        if (image.shape != (1, 3, *INPUT_HW) or image.dtype != torch.float32
                or not torch.isfinite(image).all() or not ((image >= 0) & (image <= 1)).all()):
            raise ValueError('exact original320x1024 RGB[0,1] singleton required')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('frozen eval target required')
        with torch.inference_mode():
            raw = self.model(image, augment=False, visualize=False)[0]
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 85 or not torch.isfinite(raw).all():
                raise ValueError('complete COCO80 decoded anchors required')
            # Same one fixed postprocessing protocol for every method/state.
            # Native box coordinates already match320x1024; no extra resizing.
            out = self.nms(raw.clone(), conf_thres=CONFIDENCE_THRESHOLD,
                           iou_thres=NMS_IOU, classes=None, agnostic=False,
                           multi_label=True, max_det=MAX_DETECTIONS)
            if len(out) != 1:
                raise ValueError('NMS did not return the complete singleton')
            result = remap_prediction(out[0])
            result['native_decoded_anchors'] = raw.detach().cpu()
            return result


class SharedTargetAdapter:
    """List-returning interface expected by the unchanged shared evaluator."""
    def __init__(self, target):
        self.target = target
        self.decoded_anchor_records = []

    def predict(self, image):
        from eval_contract import compact_tensor_sha
        result = self.target.predict(image)
        raw = result.pop('native_decoded_anchors')
        self.decoded_anchor_records.append({'invocation_index': len(self.decoded_anchor_records),
            'input_sha256': compact_tensor_sha(image), 'shape': list(raw.shape),
            'dtype': str(raw.dtype), 'tensor_sha256': compact_tensor_sha(raw),
            'full_anchor_BODY_retained': False})
        return [result]
