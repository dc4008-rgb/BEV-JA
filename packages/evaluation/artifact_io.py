"""Lossless evidence encoding. No dtype changes, tensor truncation or RNG draws."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

RAW_STORAGE = {'encoding': 'gzip_torch_save_weights_only', 'compression_level': 9,
               'gzip_mtime': 0, 'lossless': True, 'typed_tensor_sha256_unchanged': True}
TENSOR_MARKER = '__advrm_lossless_tensor_blob_v1__'
CAS_STORAGE = {**RAW_STORAGE, 'encoding': 'gzip_torch_save_tensor_references_v1',
               'content_address': 'dtype_shape_and_tensor_bytes_sha256',
               'all_scientific_forwards_executed': True,
               'prediction_reuse_or_inference_cache': False}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_torch_lossless(path, payload, *, pool_root=None):
    """Write the original torch payload directly to a new compressed evidence file."""
    import torch
    path = Path(path)
    if not str(path).endswith('.pt.gz'):
        raise ValueError('new lossless evidence must have a .pt.gz suffix')
    if any(part.is_symlink() for part in (path, *path.absolute().parents)):
        raise ValueError('evidence symlink forbidden')
    if pool_root is not None:
        payload = externalize_tensors(payload, pool_root)
    with path.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream,
                           compresslevel=RAW_STORAGE['compression_level'], mtime=0) as compressed:
            torch.save(payload, compressed)
        stream.flush()
        os.fsync(stream.fileno())
    return {'sha256': file_sha(path), 'bytes': path.stat().st_size,
            'serialization': 'torch-weights-only-gzip-v1'}


def load_torch_lossless(path, *, pool_root=None):
    """Decode without modifying torch dtype, shape, raw bits or metadata."""
    import torch
    path = Path(path)
    if str(path).endswith('.pt.gz'):
        with gzip.open(path, 'rb') as stream:
            payload = torch.load(stream, map_location='cpu', weights_only=True)
    elif path.suffix == '.pt':
        payload = torch.load(path, map_location='cpu', weights_only=True)
    else:
        raise ValueError('only original .pt or lossless .pt.gz evidence may be decoded')
    return materialize_tensors(payload, pool_root) if pool_root is not None else payload


def typed_tensor_sha(value):
    """Same dtype-and-shape-bound hash used by the frozen raw prediction audits."""
    import torch
    value = value.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True,
                        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_fingerprint(value):
    """Exact frozen evidence.fingerprint algorithm, for CPU state reconstruction."""
    import torch
    result = hashlib.sha256()
    def consume(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            result.update(str(('tensor', str(x.dtype), tuple(x.shape))).encode())
            result.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict{')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                consume(key)
                consume(item[key])
            result.update(b'}')
        elif isinstance(item, (tuple, list)):
            result.update((type(item).__name__ + '[').encode())
            for child in item:
                consume(child)
            result.update(b']')
        else:
            result.update((type(item).__name__ + ':' + repr(item) + ';').encode())
    consume(value)
    return result.hexdigest()


def checked_pool(root):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError('existing absolute nonsymlink frozen tensor pool required')
    blob_root = root / 'blobs'
    if not blob_root.is_dir() or blob_root.is_symlink():
        raise ValueError('predeclared tensor pool blobs directory required')
    return root


def externalize_tensors(value, pool_root):
    """Deduplicate exact computed tensors. This never skips a victim-model call."""
    import torch
    root = checked_pool(pool_root)
    def encode(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest = typed_tensor_sha(tensor)
            path = root / 'blobs' / (digest + '.pt.gz')
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise ValueError('tensor blob type differs')
                prior = load_torch_lossless(path)
                if not isinstance(prior, torch.Tensor) or typed_tensor_sha(prior) != digest:
                    raise ValueError('existing content-addressed tensor differs or is incomplete')
            else:
                save_torch_lossless(path, tensor)
            return {TENSOR_MARKER: {'tensor_sha256': digest, 'dtype': str(tensor.dtype),
                    'shape': list(tensor.shape), 'path': 'blobs/' + digest + '.pt.gz',
                    'sha256': file_sha(path), 'bytes': path.stat().st_size}}
        if isinstance(item, dict):
            if TENSOR_MARKER in item:
                raise ValueError('reserved tensor marker in original payload')
            return {key: encode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [encode(child) for child in item]
        if isinstance(item, tuple):
            return tuple(encode(child) for child in item)
        return item
    return encode(value)


def materialize_tensors(value, pool_root):
    import torch
    root = checked_pool(pool_root)
    def decode(item):
        if isinstance(item, dict) and TENSOR_MARKER in item:
            if set(item) != {TENSOR_MARKER}:
                raise ValueError('foreign tensor reference fields')
            bound = item[TENSOR_MARKER]
            if set(bound) != {'tensor_sha256', 'dtype', 'shape', 'path', 'sha256', 'bytes'}:
                raise ValueError('incomplete content-addressed tensor binding')
            digest = bound['tensor_sha256']
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in '0123456789abcdef' for c in digest) or
                    bound['path'] != 'blobs/' + digest + '.pt.gz'):
                raise ValueError('tensor pool path escapes frozen namespace')
            path = root / bound['path']
            if (path.is_symlink() or not path.is_file() or type(bound['bytes']) is not int or
                    path.stat().st_size != bound['bytes'] or file_sha(path) != bound['sha256']):
                raise ValueError('content-addressed blob SHA/bytes differs')
            tensor = load_torch_lossless(path)
            if (not isinstance(tensor, torch.Tensor) or str(tensor.dtype) != bound['dtype'] or
                    list(tensor.shape) != bound['shape'] or typed_tensor_sha(tensor) != digest):
                raise ValueError('decoded content-addressed tensor typed SHA differs')
            return tensor
        if isinstance(item, dict):
            return {key: decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        if isinstance(item, tuple):
            return tuple(decode(child) for child in item)
        return item
    return decode(value)


def tensor_blob_bindings(value):
    """Extract persisted external bindings for full retention admission and audit."""
    result = {}
    def collect(item):
        if isinstance(item, dict) and TENSOR_MARKER in item:
            binding = item[TENSOR_MARKER]
            key = binding['path']
            if key in result and result[key] != binding:
                raise ValueError('same tensor pool path has inconsistent bindings')
            result[key] = binding
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
    collect(value)
    return result
