import time
import torch
from ._helper import _device, _dtype, _auto_device
from typing import Any
import torch


class RNGKey:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def generator(self):
        """Always return a CPU torch.Generator with this seed"""
        g = torch.Generator(device="cpu")  # torch only supports CPU
        g.manual_seed(self.seed)
        return g

    def split(self, n=2):
        """JAX-style splitting: just bump the seed deterministically"""
        return [RNGKey(self.seed + i + 1) for i in range(n)]

    def __repr__(self):
        return f"RNGKey({self.seed})"


def _resolve_generator(key=None):
    if key is not None:
        return key.generator()
    g = torch.Generator(device="cpu")
    g.manual_seed(int(time.time()))
    return g


def rand(shape, dtype: Any = '', key=None):
    out = torch.rand(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        generator=_resolve_generator(key)
    ).numpy()
    return torch.as_tensor(out, device=_auto_device())


def randn(shape, dtype: Any = '', key=None):
    out = torch.randn(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        generator=_resolve_generator(key)
    ).numpy()
    return torch.as_tensor(out, device=_auto_device())


def randint(low, high, size, dtype: Any = '', key=None):
    out = torch.randint(
        low, high, size,
        dtype=_dtype(dtype) if dtype else None,
        generator=_resolve_generator(key)
    ).numpy()
    return torch.as_tensor(out, device=_auto_device())


def randperm(n, dtype: Any = '', key=None):
    out = torch.randperm(
        n,
        dtype=_dtype(dtype) if dtype else None,
        generator=_resolve_generator(key)
    ).numpy()
    return torch.as_tensor(out, device=_auto_device())


def normal(mean=0.0, std=1.0, size=(), dtype: Any = '', key=None):
    out = torch.normal(
        mean, std, size=size,
        dtype=_dtype(dtype) if dtype else None,
        generator=_resolve_generator(key)
    ).numpy()
    return torch.as_tensor(out, device=_auto_device())


def uniform(low=0.0, high=1.0, size=(), dtype: Any = '', key=None):
    out = (high - low) * torch.rand(
        size,
        dtype=_dtype(dtype) if dtype else None,
        generator=_resolve_generator(key)
    ).numpy() + low
    return torch.as_tensor(out, device=_auto_device())