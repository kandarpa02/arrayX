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

    def unwrap(data):
        return data.data if isinstance(data, LiteTensor) else data

    def wrapped(*values):
        op = fn_object()
        valargs = []
        boolargs = []

        for arg in values:
            if isinstance(arg, LiteTensor):
                valargs.append(arg.data)
            elif isinstance(arg, neolib.Tensor):
                valargs.append(arg)
            elif isinstance(arg, (bool, type(None), int)): 
                boolargs.append(arg)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        newargs = valargs + boolargs
        out_val = op.forward(*newargs)
        out = LiteTensor(out_val)

        node = GRAPH_MANAGER.Node(out, values, op.backward)
        GRAPH_MANAGER.TapeContext.add_node(node)
        return out

    return wrapped
