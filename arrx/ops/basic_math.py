from ..typing import Array
from ..core.Array import ArrayImpl, ArrayStorage
import numpy as np
from arrx import array

def shift(data):
    if isinstance(data, ArrayImpl):
        return data
    if isinstance(data, ArrayStorage):
        return ArrayImpl(data)
    # If scalar or array, wrap as ArrayStorage then as ArrayImpl
    return ArrayImpl(ArrayStorage(np.array(data)), parents=(), bwd_fn=None)


# Logarithm and Expo
def log(x:Array):
    x = shift(x)
    out = ArrayImpl(x, (x,), lambda grad: grad * (1/x))
    out.parents = (x,)
    return out

def log10(x:Array):
    x = shift(x)
    out = ArrayImpl(x, (x,), lambda grad: grad * (1/x)) * shift(np.log10(x._rawbuffer))
    out.parents = (x,)
    return out

