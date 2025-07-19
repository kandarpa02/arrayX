from neo._src.autograd import GRAPH_MANAGER, FUNCTION_REGISTER
# from neo.numpy.Array import Array 
from typing import Callable

    
def neo_function(fn_object: Callable):
    from neo.numpy.Array import Array

    def wrapped(*arrays):
        op = fn_object()
        valargs = []
        boolargs = []

        for arg in arrays:
            if isinstance(arg, Array):
                valargs.append(arg.value)
            elif isinstance(arg, (bool, type(None), int)): 
                boolargs.append(arg)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        newargs = valargs + boolargs
        out_val = op.forward(*newargs)
        out = Array(out_val, device=arrays[0].device)

        node = GRAPH_MANAGER.Node(out, arrays, op.backward)
        GRAPH_MANAGER.TapeContext.add_node(node)
        return out

    return wrapped
