from neo._src.autograd import GRAPH_MANAGER, FUNCTION_REGISTER
# from neo.numpy.Array import Array 
from typing import Callable
import warnings
from neo.backend import get_xp

def neo_function(fn):
    warnings.warn(
        "'@neo_function' is deprecated. Please use '@function' instead.",
        DeprecationWarning,
        stacklevel=2 
    )
    return function(fn)  

    
def function(fn_object: Callable):
    from neo.numpy.Array import Array

    def wrapped(*arrays):
        device = arrays[0].device
        xp = get_xp(device=device)
        op = fn_object(device)
        valargs = []
        boolargs = []

        for arg in arrays:
            if isinstance(arg, Array):
                valargs.append(arg.value)
            elif isinstance(arg, xp.ndarray):
                valargs.append(arg)
            elif isinstance(arg, (bool, type(None), int)): 
                boolargs.append(arg)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        newargs = valargs + boolargs
        out_val = op.forward(*newargs)
        out = Array(out_val, device=device)

        node = GRAPH_MANAGER.Node(out, arrays, op.backward)
        GRAPH_MANAGER.TapeContext.add_node(node)
        return out

    return wrapped
