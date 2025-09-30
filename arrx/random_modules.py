# from .Core.Array import ArrayImpl
# from .Core.Dtype import uint32, float32
# import random
# import math, time
# from typing import Optional

# def RNGKey(seed: int):
#     """
#     Create a uint32 key.
#     """
#     from arrx import lib
#     return ArrayImpl(
#         [lib.uint32(seed), lib.uint32(seed ^ 0x9E3779B9)],  # golden ratio constant
#         dtype=uint32()
#     )


# def split(key: ArrayImpl, n=2):
#     """
#     Split a key into n new statistically decorrelated keys.
#     """

#     def _mix(k0, k1, salt):
#         k0 = (k0 ^ (k1 >> 16)) * 0x85ebca6b
#         k0 = ((k0 << 13) | (k0 >> 19)) & 0xFFFFFFFF
#         k1 = (k1 ^ (k0 >> 16)) * 0xc2b2ae35
#         return (k0 ^ salt) & 0xFFFFFFFF, (k1 + salt) & 0xFFFFFFFF

#     k0 = int(key._rawbuffer[0].item())
#     k1 = int(key._rawbuffer[1].item())
#     out = []
#     for i in range(n):
#         nk0, nk1 = _mix(k0, k1, i + 1)
#         out.append(ArrayImpl([nk0, nk1], dtype=uint32()))
#     return tuple(out)


# def fill_engine(shape, fill_fn):
#     """
#     For using custom random methods
    
#     Arguments: 
#     shape: takes the shape
#     fill_fn: the algorithm for generating distributions
#     """
#     if not shape:
#         return fill_fn()
#     return [fill_engine(shape[1:], fill_fn) for _ in range(shape[0])]


# # Uniform dist
# def uniform(*shape, key: Optional[ArrayImpl] = None, a=0.0, b=1.0):
#     """
#     Generates random numbers uniformly distributed in [a, b).
#     Uses the provided key, or auto-seeds from current time if key is None.
#     """
#     def fill_fn():
#         nonlocal key
#         if key is None:
#             seed = int(time.time() * 1e6) % 2**32
#             key = RNGKey(seed)
#         k0 = key._rawbuffer[0].item()
#         k1 = key._rawbuffer[1].item()
#         result = ((k0 * 1664525 + k1 * 1013904223) % 2**32) / 2**32
#         new_k0 = (k0 + 1) % 2**32
#         new_k1 = (k1 + 1) % 2**32
#         key = ArrayImpl([new_k0, new_k1], dtype=uint32())
#         return a + (b - a) * result
#     raw = fill_engine(shape, fill_fn=fill_fn)
#     return ArrayImpl(raw, dtype=float32())


# # # Normal dist
# def normal(*shape, key: Optional[ArrayImpl] = None, mu=0.0, sigma=1.0):
#     """
#     Generates random numbers normally distributed with mean `mu` and std deviation `sigma`.
#     Uses the provided key, or auto-seeds from current time if key is None.
#     """
#     def fill_fn():
#         nonlocal key
#         if key is None:
#             seed = int(time.time() * 1e6) % 2**32
#             key = RNGKey(seed)
#         k0 = key._rawbuffer[0].item()
#         k1 = key._rawbuffer[1].item()
#         # Box-Muller transform
#         u1 = ((k0 * 1664525 + k1 * 1013904223) % 2**32) / 2**32
#         new_k0 = (k0 + 1) % 2**32
#         new_k1 = (k1 + 1) % 2**32
#         key = ArrayImpl([new_k0, new_k1], dtype=uint32())
#         u1 = max(u1, 1e-10)  # Avoid log(0)
#         u2 = ((new_k0 * 1664525 + new_k1 * 1013904223) % 2**32) / 2**32
#         z0 = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
#         return mu + sigma * z0
#     raw = fill_engine(shape, fill_fn=fill_fn)
#     return ArrayImpl(raw, dtype=float32())
