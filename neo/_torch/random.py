import time
from .user_functions import lite
from neo._torch import neolib
from ._helper import _device, _dtype, _auto_device
from typing import Any
import torch

class RNGKey:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def generator(self, device=None):
        """Return a torch.Generator on given device (None => auto_device)"""
        dev = _auto_device() if device in ('', None) else _device(device)
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)
        return g

    def split(self, n=2):
        return [RNGKey(self.seed + i + 1) for i in range(n)]

    def __repr__(self):
        return f"RNGKey({self.seed})"


def _resolve_generator(key=None, device=None):
    dev = _auto_device() if device in ('', None) else _device(device)
    if key is not None:
        return key.generator(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(int(time.time()))
    return g


def rand(shape, dtype:Any='', device:Any=None, key=None):
    return lite(neolib.rand(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device)
    ))


def randn(shape, dtype:Any='', device:Any=None, key=None):
    return lite(neolib.randn(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device)
    ))


def randint(low, high, size, dtype:Any='', device:Any=None, key=None):
    return lite(neolib.randint(
        low, high, size,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device)
    ))


def randperm(n, dtype:Any='', device:Any=None, key=None):
    return lite(neolib.randperm(
        n,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device)
    ))


def normal(mean=0.0, std=1.0, size=(), dtype:Any='', device:Any=None, key=None):
    return lite(neolib.normal(
        mean, std, size=size,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None,
        generator=_resolve_generator(key, device)
    ))


def uniform(low=0.0, high=1.0, size=(), dtype:Any='', device:Any=None, key=None):
    gen = _resolve_generator(key, device)
    return lite(
        (high - low) * neolib.rand(
            size,
            dtype=_dtype(dtype) if dtype else None,
            device=_device(device) if device else None,
            generator=gen
        ) + low
    )
