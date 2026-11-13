"""Original finite evaluation admission plus exact reviewed CUDA UUID codec.

No model is constructed by this module. CUDA identities remain real runtime
observations; the codec merely supports actual Torch16-byte and UUID-string
representations, retaining exact selected GPU identity comparison.
"""
import math
from pathlib import Path
import re
import signal
import socket
import time
import uuid


def need(v, message):
    if not v:
        raise ValueError(message)


def cuda_uuid(value):
    if not isinstance(value, (str, bytes)):
        import torch
        need(type(value) is torch._C._CUuuid, "only the trusted native torch._C._CUuuid CUDA type")
        octets = value.bytes
        need(type(octets) is list and len(octets) == 16
             and all(type(x) is int and 0 <= x <= 255 for x in octets),
             "native CUDA UUID must contain exactly sixteen uint8 values")
        actual = str(uuid.UUID(bytes=bytes(octets)))
        native = str(value)
        need(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", native) is not None
             and str(uuid.UUID(native)) == actual, "native CUDA UUID bytes and native string agree")
        return "GPU-" + actual
    if isinstance(value, bytes) and len(value) == 16:
        return "GPU-" + str(uuid.UUID(bytes=value))
    need(isinstance(value, str) and re.fullmatch(
        r"(?i:GPU)-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value) is not None,
        "actual CUDA0 UUID available, no NVML-only substitution")
    return "GPU-" + str(uuid.UUID(value[4:]))



def admission(q):
    import torch
    a = q['admission']
    if set(a) != {'expected_hostname', 'expected_boot_id', 'expected_GPU_UUID',
                  'deadline_unix', 'wall_seconds', 'max_RSS_bytes', 'disk_reserve_bytes'}:
        raise ValueError('fresh explicit finite runtime admission required')
    if socket.gethostname() != a['expected_hostname']:
        raise RuntimeError('actual host differs')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != a['expected_boot_id']:
        raise RuntimeError('actual boot differs')
    for k in ('deadline_unix', 'wall_seconds', 'max_RSS_bytes', 'disk_reserve_bytes'):
        if type(a[k]) not in (int, float) or not math.isfinite(a[k]) or a[k] <= 0:
            raise ValueError('finite positive runtime bound required')
    if not 1 <= a['wall_seconds'] <= 12 * 3600 or a['disk_reserve_bytes'] < 1024 ** 3:
        raise ValueError('fixed finite eval wall and disk reserve required')
    remaining = min(a['wall_seconds'], a['deadline_unix'] - time.time())
    if remaining < 1:
        raise TimeoutError('new evaluation admission expired')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('finite eval wall')))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exact one admitted GPU required')
    actual = cuda_uuid(torch.cuda.get_device_properties(0).uuid)
    if actual != cuda_uuid(a['expected_GPU_UUID']):
        raise RuntimeError('actual GPU UUID differs')
    return a

