"""Literal file bindings for new Ours5 ablation evaluation."""
from pathlib import Path
import hashlib
import json
import os
import stat

def plain(value):
    p = Path(value).absolute()
    if '..' in p.parts or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('literal absolute nonsymlink path required')
    return p

def descriptor(path):
    p = plain(path)
    before = p.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 ** 3:
        raise ValueError('bounded regular model/evaluation file required')
    stable = lambda x: (x.st_dev, x.st_ino, x.st_mode, x.st_uid, x.st_gid,
                        x.st_nlink, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    h = hashlib.sha256()
    with p.open('rb') as f:
        if stable(os.fstat(f.fileno())) != stable(before):
            raise ValueError('original path/FD identity differs')
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
        if stable(os.fstat(f.fileno())) != stable(before):
            raise ValueError('original FD changed while hashing')
    if stable(p.lstat()) != stable(before):
        raise ValueError('original path changed while hashing')
    return {'path': str(p), 'bytes': before.st_size, 'sha256': h.hexdigest()}

def bound(d, *, cap=16 * 1024 ** 2):
    if not isinstance(d, dict) or set(d) != {'path', 'bytes', 'sha256'}:
        raise ValueError('closed descriptor required')
    if type(d['bytes']) is not int or not 0 <= d['bytes'] <= cap:
        raise ValueError('descriptor exceeds fixed input bound')
    p = plain(d['path'])
    a = p.stat()
    if a.st_size != d['bytes']:
        raise ValueError('actual file size differs before whole hashing')
    if descriptor(p) != d:
        raise ValueError('whole original bytes differ: ' + str(p))
    b = p.read_bytes()
    z = p.stat()
    stat = lambda x: (x.st_dev, x.st_ino, x.st_mode, x.st_uid, x.st_gid,
                      x.st_nlink, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    if (stat(a) != stat(z) or len(b) != d['bytes']
            or hashlib.sha256(b).hexdigest() != d['sha256']):
        raise ValueError('original changed during bounded read')
    return b

def read(d, *, cap=16 * 1024 ** 2):
    return json.loads(bound(d, cap=cap), parse_constant=lambda x: (
        _ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))

def save(path, value):
    p = plain(path)
    b = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                    separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(b) > 32 * 1024 ** 2:
        raise ValueError('new scalar metadata exceeds32MiB')
    with p.open('xb') as f:
        f.write(b); f.flush(); os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return descriptor(p)


def compact_tensor_sha(x):
    """Original shared evaluator fingerprint for scene/prediction evidence."""
    import torch
    x = x.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(x.dtype), 'shape': list(x.shape)},
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + x.view(torch.uint8).numpy().tobytes()).hexdigest()
