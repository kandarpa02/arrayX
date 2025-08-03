# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.

from neo._src.autograd import FUNCTION_REGISTER
from neo._torch import neolib
from typing import Callable
import warnings

def neo_function(fn):
    warnings.warn(
        "'@neo_function' is deprecated. Please use '@function' instead.",
        DeprecationWarning,
        stacklevel=2 
    )
    return function(fn)  


def function(fn_object: Callable):
    from neo._torch.lite_tensor import LiteTensor
    from neo._src.autograd import GRAPH_MANAGER

    def wrapped(*args, **kwargs):
        op = fn_object()

        valargs = []
        valargs_strict = []
        auxargs = []

        for arg in args:
            if isinstance(arg, LiteTensor):
                valargs.append(arg.data)
                valargs_strict.append(arg)
            elif isinstance(arg, (neolib.Tensor, int, float)):
                valargs.append(arg)
            elif isinstance(arg, (bool, type(None), tuple, list)):
                auxargs.append(arg)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        for key, val in kwargs.items():
            if isinstance(val, (int, bool)):
                auxargs.append(val)
            else:
                raise TypeError(f"Unsupported keyword argument {key}={val} in function")

        out_val = op.forward(*valargs, *auxargs)
        out = LiteTensor(out_val)
        
        node = GRAPH_MANAGER.Node(out, tuple(valargs_strict), op.backward)
        GRAPH_MANAGER.TapeContext.add_node(node)
        return out
        
    return wrapped
