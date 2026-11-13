"""Original accepted native-tensor to NumPy conversion."""
from atomic_counts import need

def numpy_value(value, *, labels=False, mask=False):
    import numpy as np
    import torch
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.numel() <= 16_777_216,
         "bounded original CPU tensor required")
    if mask:
        need(value.dtype == torch.bool or value.is_floating_point(), "native mask must be boolean or floating")
        a = value.double().numpy() if value.is_floating_point() else value.numpy()
        need(np.isfinite(a).all() and ((a == 0) | (a == 1)).all(), "exact binary saved ROI only")
        return a.astype(np.bool_, copy=False)
    if labels:
        need(not value.is_floating_point() and value.dtype != torch.bool, "native integer labels required")
        return value.to(torch.int64).numpy()
    need(value.is_floating_point() and bool(torch.isfinite(value).all()), "finite native floats required")
    return value.double().numpy()
