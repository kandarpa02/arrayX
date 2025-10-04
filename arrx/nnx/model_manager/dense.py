from .variable_scope import *
from ..initializers import xavier_init, zero_init

def dense(x, units, name, rng=None, reset=False):
    with variable_scope(name, reset=True):
        w = get_variable('w', [x.shape[-1], units], initializer=xavier_init, rng=rng)
        b = get_variable('b', [units,], initializer=zero_init)
    return x @ w + b

