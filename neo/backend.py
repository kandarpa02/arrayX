import warnings
import numpy as np

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def get_xp(device):
    if device is None or device == 'cpu':
        return np
    elif device == 'cuda':
        if not HAS_CUPY:
            warnings.warn(
                "CUDA backend requested, not available. Falling back to NumPy.",
                RuntimeWarning
            )
            return np
        return cp
    else:
        raise ValueError(f"Unknown device: {device}")
