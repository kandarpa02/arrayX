import numpy as np

try:
    import cupy as cp #type:ignore
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def get_xp(device):
    if device == 'cpu' or device is None:
        return np
    
    elif device == 'cuda':
        if not HAS_CUPY:
            raise RuntimeError("CuPy not installed. Please install CuPy to use CUDA backend.")
        return cp
    else:
        raise ValueError(f"Unknown device: {device}")
