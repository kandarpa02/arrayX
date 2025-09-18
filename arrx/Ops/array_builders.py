from ..typing import Array
from ..Core.Array import ArrayImpl, ArrayStorage
from arrx import lib, float32

"""
Some essential array creation functions like ones, zeros and empty with multy device support
"""

def ones(*shape, dtype=None):
    if dtype is None:
        dtype = float32
    _ones = lib.ones(shape)
    return ArrayImpl(_ones, dtype=dtype)

def zeros(*shape, dtype=None):
    if dtype is None:
        dtype = float32
    _zeros = lib.zeros(shape)
    return ArrayImpl(_zeros, dtype=dtype)

def empty(*shape, dtype=None):
    if dtype is None:
        dtype = float32
    _emp = lib.empty(shape)
    return ArrayImpl(_emp, dtype=dtype)

