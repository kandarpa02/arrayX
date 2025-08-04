import time
from .user_functions import data
from neo._torch import neolib
from ._helper import _device, _dtype
import time
import torch
import hashlib

class RNGKey:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def generator(self, device='cpu'):
        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        return g

    def split(self, n=2):
        return [RNGKey(self.seed + i + 1) for i in range(n)]

    def __repr__(self):
        return f"RNGKey({self.seed})"


def _resolve_generator(key=None, device='cpu'):
    if key is not None:
        return key.generator(device)
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(time.time()))
        return g


def rand(shape, dtype='', device='', key=None):
    return data(neolib.rand(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device or 'cpu')
    ))


def randn(shape, dtype='', device='', key=None):
    return data(neolib.randn(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device or 'cpu')
    ))


def randint(low, high, size, dtype='', device='', key=None):
    return data(neolib.randint(
        low,
        high,
        size,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device or 'cpu')
    ))


def randperm(n, dtype='', device='', key=None):
    return data(neolib.randperm(
        n,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device or 'cpu')
    ))


def normal(mean=0.0, std=1.0, size=(), dtype='', device='', key=None):
    return data(neolib.normal(
        mean,
        std,
        size=size,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device or 'cpu')
    ))


def uniform(low=0.0, high=1.0, size=(), dtype='', device='', key=None):
    gen = _resolve_generator(key, device or 'cpu')
    return data(
        (high - low) * neolib.rand(
            size,
            dtype=_dtype(dtype) if dtype else None,
            device=_device(device) if device else None,
            generator=gen
        ) + low
    )