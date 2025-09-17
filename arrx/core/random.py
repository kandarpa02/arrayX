from .Array import ArrayImpl
from .Dtype import uint32, float32
import random
import math, time
from typing import Optional

def RNGKey(seed: int) -> ArrayImpl:
    """
    Creates a random number generator key from an integer seed.
    """
    return ArrayImpl([0, seed], dtype=uint32())

def split(key: ArrayImpl, n=2):
    """
    Splits a RNG key into 'n' new keys using deterministic mixing.
    """
    stack = []
    k0 = key._rawbuffer[0].item()
    k1 = key._rawbuffer[1].item()
    for i in range(1, n+1):
        new_k0 = (k0 ^ (k1 << 5) ^ i) % 2**32
        new_k1 = (k1 ^ (k0 >> 3) ^ i) % 2**32
        stack.append(ArrayImpl([new_k0, new_k1], dtype=uint32()))
    return tuple(stack)

def random_engine(shape, fill_fn):
    if not shape:
        return fill_fn()
    return [random_engine(shape[1:], fill_fn) for _ in range(shape[0])]

def uniform(*shape, key: Optional[ArrayImpl] = None, a=0.0, b=1.0):
    """
    Generates random numbers uniformly distributed in [a, b).
    Uses the provided key, or auto-seeds from current time if key is None.
    """
    def fill_fn():
        nonlocal key
        if key is None:
            seed = int(time.time() * 1e6) % 2**32
            key = RNGKey(seed)
        k0 = key._rawbuffer[0].item()
        k1 = key._rawbuffer[1].item()
        result = ((k0 * 1664525 + k1 * 1013904223) % 2**32) / 2**32
        new_k0 = (k0 + 1) % 2**32
        new_k1 = (k1 + 1) % 2**32
        key = ArrayImpl([new_k0, new_k1], dtype=uint32())
        return a + (b - a) * result
    raw = random_engine(shape, fill_fn=fill_fn)
    return ArrayImpl(raw, dtype=float32())

def normal(*shape, key: Optional[ArrayImpl] = None, mu=0.0, sigma=1.0):
    """
    Generates random numbers normally distributed with mean `mu` and std deviation `sigma`.
    Uses the provided key, or auto-seeds from current time if key is None.
    """
    def fill_fn():
        nonlocal key
        if key is None:
            seed = int(time.time() * 1e6) % 2**32
            key = RNGKey(seed)
        k0 = key._rawbuffer[0].item()
        k1 = key._rawbuffer[1].item()
        # Box-Muller transform
        u1 = ((k0 * 1664525 + k1 * 1013904223) % 2**32) / 2**32
        new_k0 = (k0 + 1) % 2**32
        new_k1 = (k1 + 1) % 2**32
        key = ArrayImpl([new_k0, new_k1], dtype=uint32())
        u1 = max(u1, 1e-10)  # Avoid log(0)
        u2 = ((new_k0 * 1664525 + new_k1 * 1013904223) % 2**32) / 2**32
        z0 = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
        return mu + sigma * z0
    raw = random_engine(shape, fill_fn=fill_fn)
    return ArrayImpl(raw, dtype=float32())
