from ..typing import Array
from ..Core.Array import ArrayImpl, ArrayStorage
from arrx import lib

"""
Some essential array creation functions like ones, zeros and empty with multy device support
"""

def ones(*shape, dtype=None):
    _ones = lib.ones(shape)
    return ArrayImpl(_ones)

def zeros(*shape, dtype=None):
    _zeros = lib.zeros(shape)
    return ArrayImpl(_zeros)

def empty(*shape, dtype=None):
    _emp = lib.empty(shape)
    return ArrayImpl(_emp)

