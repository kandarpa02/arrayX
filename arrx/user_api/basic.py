from ..src.Tensor.base import placeholder, vector, scalar, matrix
from typing import Callable

def variable(data=None, *, shape=[], name=None) -> placeholder|vector|scalar|matrix:
    flag = False
    if data is not None:
        flag = True
    
    if flag:
        out = placeholder.place(*data.shape, name=name) #type:ignore
        out.grad_required = True
        out.value = data

    else:
        out = placeholder.place(*shape, name=name)
        out.grad_required = True

    return out

def constant(data=None, *, shape=[], name=None) -> placeholder|vector|scalar|matrix:
    flag = False
    if data is not None:
        flag = True
    
    if flag:
        out = placeholder.place(*data.shape, name=name) #type:ignore
        out.value = data

    else:
        out = placeholder.place(*shape, name=name)

    return out
