from ..src.Tensor.base import placeholder, vector, scalar, matrix
from ..src.Tensor.utils import name_filler
from typing import Callable

def data_shift(x, dtype=None, fun_name=lambda:None):
    import numpy as np
    from arrx.src.Tensor.utils import lib
    if isinstance(x, lib.ndarray|np.ndarray):
        return x 
    elif isinstance(x, int|float|list):
        return lib.array(x, dtype=dtype)
    else:
        raise TypeError(
            f"{fun_name} expects its optional data argument to be array like objects but found {type(x)}, "
            f"check your input data. "
        )


def Variable(data=None, *, shape=[], dtype=None, name=None) -> placeholder|vector|scalar|matrix:
    flag = False
    if data is not None:
        flag = True
        data = data_shift(data, dtype, Variable)
    
    if flag:
        out = placeholder.place(*data.shape, name=name) #type:ignore
        out.grad_required = True
        out.value = data

    else:
        out = placeholder.place(*shape, name=name)
        out.grad_required = True
    
    out.expr = name_filler.get_name('const') if name is None else name

    return out

def Constant(data=None, *, shape=[], dtype=None, name=None) -> placeholder|vector|scalar|matrix:
    flag = False
    if data is not None:
        flag = True
        data = data_shift(data, dtype, Constant)
    
    if flag:
        out = placeholder.place(*data.shape, name=name) #type:ignore
        out.value = data

    else:
        out = placeholder.place(*shape, name=name)

    out.expr = name_filler.get_name('const') if name is None else name

    return out
