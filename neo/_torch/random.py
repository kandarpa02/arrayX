import time
from .user_functions import Lite
from neo._torch import neolib
from ._helper import _device, _dtype
import time

def _seed(seed):
    neolib.manual_seed(seed if seed is not None else int(time.time()))

def rand(shape, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.rand(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def randn(shape, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.randn(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def randint(low, high, size, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.randint(
        low,
        high,
        size,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def randperm(n, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.randperm(
        n,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def normal(mean=0.0, std=1.0, size=(), dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.normal(
        mean,
        std,
        size=size,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def uniform(low=0.0, high=1.0, size=(), dtype='', device='', seed=None):
    _seed(seed)
    return Lite(
        (high - low) * neolib.rand(
            size,
            dtype=_dtype(dtype) if dtype else None,
            device=_device(device) if device else None
        ) + low
    )
