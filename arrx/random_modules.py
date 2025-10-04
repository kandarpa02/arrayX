from .src.Dtype import uint32, float32
import random
import math, time
from typing import Optional
from .user_api.basic import Constant, Variable
from .src.Tensor.base import placeholder, vector, matrix, scalar

TensorLike = placeholder | vector | matrix | scalar

def RNGKey(seed: int):
    """
    Create a uint32 key.
    """
    import numpy as np
    return Constant(
        [np.uint32(seed), np.uint32(seed ^ 0x9E3779B9)],  # golden ratio constant
        name='key'
    )

def split(key: vector, n=2):
    """
    Split a key into n new statistically decorrelated keys.
    """

    def _mix(k0, k1, salt):
        k0 = (k0 ^ (k1 >> 16)) * 0x85ebca6b
        k0 = ((k0 << 13) | (k0 >> 19)) & 0xFFFFFFFF
        k1 = (k1 ^ (k0 >> 16)) * 0xc2b2ae35
        return (k0 ^ salt) & 0xFFFFFFFF, (k1 + salt) & 0xFFFFFFFF

    k0 = int(key.value[0].item()) #type:ignore
    k1 = int(key.value[1].item()) #type:ignore
    out = []
    for i in range(n):
        nk0, nk1 = _mix(k0, k1, i + 1)
        out.append(Constant([nk0, nk1], dtype=uint32()))
    return tuple(out)


def fill_engine(shape, fill_fn):
    """
    For using custom random methods
    
    Arguments: 
    shape: takes the shape
    fill_fn: the algorithm for generating distributions
    """
    if not shape:
        return fill_fn()
    return [fill_engine(shape[1:], fill_fn) for _ in range(shape[0])]


# Uniform dist
def uniform(*shape, key: Optional[TensorLike] = None, a=0.0, b=1.0):
    """
    Generates random numbers uniformly distributed in [a, b).
    Uses the provided key, or auto-seeds from current time if key is None.
    """
    def fill_fn():
        nonlocal key
        if key is None:
            seed = int(time.time() * 1e6) % 2**32
            key = RNGKey(seed) #type:ignore
        k0 = key.value[0].item() #type:ignore
        k1 = key.value[1].item() #type:ignore
        result = ((k0 * 1664525 + k1 * 1013904223) % 2**32) / 2**32
        new_k0 = (k0 + 1) % 2**32
        new_k1 = (k1 + 1) % 2**32
        key = Constant([new_k0, new_k1]) #type:ignore
        return a + (b - a) * result
    raw = fill_engine(shape, fill_fn=fill_fn)
    return Variable(raw)


# # Normal dist
def normal(*shape, key: Optional[TensorLike] = None, mu=0.0, sigma=1.0):
    """
    Generates random numbers normally distributed with mean `mu` and std deviation `sigma`.
    Uses the provided key, or auto-seeds from current time if key is None.
    """
    def fill_fn():
        nonlocal key
        if key is None:
            seed = int(time.time() * 1e6) % 2**32
            key = RNGKey(seed) #type:ignore
        k0 = key.value[0].item() #type:ignore
        k1 = key.value[1].item() #type:ignore
        # Box-Muller transform
        u1 = ((k0 * 1664525 + k1 * 1013904223) % 2**32) / 2**32
        new_k0 = (k0 + 1) % 2**32
        new_k1 = (k1 + 1) % 2**32
        key = Constant([new_k0, new_k1]) #type:ignore
        u1 = max(u1, 1e-10)  # Avoid log(0)
        u2 = ((new_k0 * 1664525 + new_k1 * 1013904223) % 2**32) / 2**32
        z0 = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
        return mu + sigma * z0
    raw = fill_engine(shape, fill_fn=fill_fn)
    return Variable(raw)
