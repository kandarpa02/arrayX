import warnings
import numpy as np

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def get_xp(device: str):
    if device is None or device == 'cpu':
        return np
    elif device == 'cuda':
        if not HAS_CUPY:
            raise RuntimeError(
                "CUDA backend requested, but CuPy is not installed."
            )
        return cp
    else:
        raise ValueError(f"Unknown device: {device}")
