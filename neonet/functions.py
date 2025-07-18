from neonet._src.autograd import GRAPH_MANAGER, FUNCTION_REGISTER
from neonet.numpy.Array import Array
from typing import Callable

def fn_forward(fn_object:Callable):
    def wrapped(*arrays):
        op = fn_object()
        newargs = []
        for arr in list(arrays):
            newargs.append(arr.value)
        out_val = op.forward(*newargs)
        out = Array(out_val, device=arrays[0].device)
        
        node = GRAPH_MANAGER.Node(out, arrays, op.backward)
        GRAPH_MANAGER.TapeContext.add_node(node)
        return out
    
    return wrapped
    

