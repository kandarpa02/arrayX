from ..src.Tensor.base import placeholder
from ..typing import NDarray as ND
import numpy as np
from typing import Union

NDarray = Union[ND, np.ndarray]

def keys_recurse(d, name=False):
    keys_= []
    for k, v in d.items():
        if isinstance(k, placeholder):
            if name:
                keys_.append(k.expr)
            else:
                keys_.append(k)
        
        elif isinstance(v, dict):
            keys_.extend(keys_recurse(v, name=name))

    return tuple(keys_)

def value_recurse(d):
    value_= []
    for v in d.values():
        if isinstance(v, NDarray):
            value_.append(v)

        elif isinstance(v, dict):
            value_.extend(value_recurse(v))

    return tuple(value_)

def flatten_params(d):
    flat = {}

    def _flatten(subdict):
        for k, v in subdict.items():
            if isinstance(v, dict):
                _flatten(v)
            else:
                flat[k.expr] = v

    _flatten(d)
    return flat


class ParamDict(dict):
    def to_list(self):
        return value_recurse(self)
    
    def variables(self, name=False, idx=None):
        if idx is None:
            return keys_recurse(self, name=name)
        return keys_recurse(self, name=name)[idx]
    
    def put_kwargs(self):
        return flatten_params(self)