from ..Tensor.base import placeholder, scalar, matrix, vector
from typing import Any, Sequence
from .utils import (
    _unbroadcast,
    broadcast_shape,
    filler_name,
    data_by_dim,
    reduced_shape,
    broadcast_to,
    matmul_shape,
    reshape_shape,
    transpose_shape,
)

from .utils import lib
from ..errors import ShapeError


Tensorlike = placeholder|scalar|vector|matrix|int|float

def where(condition:Tensorlike, x:Tensorlike, y:Tensorlike, name=None):
    """
    Elementwise conditional selection.
    
    Returns a placeholder representing:
        out[i] = x[i] if condition[i] else y[i]

    Arguments:
        condition : placeholder or bool ndarray (broadcastable to x/y)
        x, y      : placeholders or ndarrays
        name      : optional string for naming this node
    
    Returns:
        placeholder object of broadcasted shape
    """
    condition = placeholder._make_place(condition) #type:ignore
    x = placeholder._make_place(x)                 #type:ignore
    y = placeholder._make_place(y)                 #type:ignore

    _shape = broadcast_shape(condition.shape, broadcast_shape(x.shape, y.shape)) #type:ignore
    obj = placeholder.object(*_shape)
    out = obj(_shape, f"OPS.where({condition.name}, {x.name}, {y.name})") #type:ignore
    out.parents = (condition, x, y)

    def _grad_where(grad):
        # grad flows only through selected branch
        g_cond = None  # condition not differentiable
        g_x = _unbroadcast(grad * condition, x.shape) #type:ignore
        g_y = _unbroadcast(grad * (placeholder.ones(*condition.shape) - condition), y.shape) #type:ignore
        return g_cond, g_x, g_y

    out.grad_fn = _grad_where
    return out

