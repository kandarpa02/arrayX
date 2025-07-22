import time
from .user_functions import Lite
from neo._torch import neolib

def _seed(seed):
    neolib.manual_seed(seed if seed is not None else int(time.time()))


def rand(shape, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.rand(shape, dtype=dtype or None, device=device or None))


def randn(shape, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.randn(shape, dtype=dtype or None, device=device or None))


def randint(low, high, size, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.randint(low, high, size, dtype=dtype or None, device=device or None))


def randperm(n, dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.randperm(n, dtype=dtype or None, device=device or None))


def normal(mean=0.0, std=1.0, size=(), dtype='', device='', seed=None):
    _seed(seed)
    return Lite(neolib.normal(mean, std, size=size, dtype=dtype or None, device=device or None))


# Bonus: Uniform variant
def uniform(low=0.0, high=1.0, size=(), dtype='', device='', seed=None):
    _seed(seed)
    return Lite((high - low) * neolib.rand(size, dtype=dtype or None, device=device or None) + low)
