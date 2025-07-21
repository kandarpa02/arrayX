from neo._src.autograd import Node, Tape, TapeContext
from typing import Callable
import numpy as np
# from neonet.numpy import array
from neo.backend import get_xp

def define_device(x):
    return 'cpu' if isinstance(x, np.ndarray) else 'cuda'

def rectify_shapes(val):
    return val.reshape(1) if val.ndim < 1 else val

def unpack_tuple(tup):
    return {f'x{i+1}': value for i, value in enumerate(tup)}

def if_xnary(grads):
    def _fix(g):
        if g.ndim == 0:
            return g.reshape(1)
        elif g.ndim == 1:
            return g[None, :]
        return g

    if isinstance(grads, tuple):
        return tuple(_fix(g) for g in grads)
    else:
        return _fix(grads)

def value_and_grad(fn: Callable, debug=False):
    def wrapped_function(*args):
        tape = Tape()
        TapeContext.push(tape.nodes)
        out = fn(*args)
        TapeContext.pop()
        
        device = define_device(out.value)
        xp = get_xp(device=device)
        
        if xp.isscalar(out.value):
            out_grad = xp.array(1.0, dtype=out.value.dtype)
        else:
            out_grad = xp.ones_like(out.value, dtype=out.value.dtype)
        
        grad_dict = {id(out): out_grad}
        
        for node in reversed(tape.nodes):
            node_out_grad = grad_dict.get(id(node.output))
            if node_out_grad is None:
                continue
                
            grad_inputs = node.bwd_fn(grad=node_out_grad)
            if grad_inputs is None:
                continue
                
            # Handle both scalar and array gradients
            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)
                
            for parent, grad in zip(node.parents, grad_inputs):
                if grad is None:
                    continue
                    
                pid = id(parent)
                if pid in grad_dict:
                    grad_dict[pid] += grad
                else:
                    grad_dict[pid] = grad
                    
        return out, grad_dict
    
    return wrapped_function
