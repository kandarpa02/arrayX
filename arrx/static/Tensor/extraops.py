from ..Tensor.base import placeholder, scalar, matrix, vector
from typing import Any, Sequence
from ..utils import (
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

from arrx import lib
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


def cond(pred, true_fn, false_fn, name=None):
    """
    Graph-level conditional branching.

    Arguments:
        pred     : bool scalar placeholder (or Python bool)
        true_fn  : callable returning a placeholder (executed if pred is True)
        false_fn : callable returning a placeholder (executed if pred is False)
        name     : optional node name

    Returns:
        placeholder from the chosen branch
    """
    if isinstance(pred, bool):
        branch_out = true_fn() if pred else false_fn()
    else:
        # symbolic pred → we must still represent the conditional
        # (both branches are attached, actual choice deferred to runtime)
        t_out = true_fn()
        f_out = false_fn()
        _shape = broadcast_shape(t_out.shape, f_out.shape)
        obj = placeholder.object(*_shape)
        branch_out = obj(_shape, f"where({pred.name}, true, false)")
        branch_out.parents = (pred, t_out, f_out)

        def _grad_cond(grad):
            g_pred = None  # not differentiable
            g_true = _unbroadcast(grad * pred, t_out.shape)
            g_false = _unbroadcast(grad * (1 - pred), f_out.shape)
            return g_pred, g_true, g_false

        branch_out.grad_fn = _grad_cond

    return branch_out
