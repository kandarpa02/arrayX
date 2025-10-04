from .variable_scope import *
from ..initializers import xavier_init, zero_init
from ...typing import MatrixLike


def dense(
        x, 
        units, 
        name, 
        initializer=xavier_init, 
        *args, rng=None, reset=False, **kwargs
        
        ):

    if not isinstance(x, MatrixLike):
        raise TypeError(
            f"{type(x)} cannot be passed in {dense}, "
            f"it expects a MatrixLike (matrix/vector) object which is of x.ndim > 0 but found ndim={x.ndim}."
        )
    
    with variable_scope(name, reset=reset):
        w = get_variable('kernel', [units, x.shape[-1]], initializer=initializer, rng=rng)
        b = get_variable('bias', [units,], initializer=zero_init)
    return x @ w.transpose() + b #type:ignore

