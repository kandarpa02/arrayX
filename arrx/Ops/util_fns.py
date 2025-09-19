from ..Core.Array import ArrayImpl, _unbroadcast
from .basic_math import *
from .array_builders import *
from arrx import lib
import numba
from itertools import chain

def precompile_(args, fn):
    jitted_fn = numba.jit(fn)
    jitted_fn(*args)
    return jitted_fn

def where(condition, x, y):
    condition = shift(condition)
    x = shift(x)
    y = shift(y)
    out = ArrayImpl(lib.where(condition._rawbuffer, x._rawbuffer, y._rawbuffer), parents=(condition, x, y))

    def _grad_where(grad):
        grad = shift(grad)
        grad_condition = None  # where gradient typically ignored for condition
        grad_x = _unbroadcast(grad * condition, x._rawbuffer.shape)
        grad_y = _unbroadcast(grad * (1 - condition), y._rawbuffer.shape)
        return grad_condition, grad_x, grad_y

    out.bwd_fn = _grad_where
    return out

def arange(start, stop=None, step=1, dtype=None):
    """
    ArrayX equivalent of numpy.arange / cupy.arange.
    """
    if stop is None:
        out = lib.arange(0, start, step, dtype=dtype)
    else:
        out = lib.arange(start, stop, step, dtype=dtype)

    return ArrayImpl(out.tolist(), parents=())