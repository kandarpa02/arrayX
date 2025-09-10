from .graph import Tape, TapeContext
from typing import Callable
import numpy as np
from xtorch.core.tensor import DeviceBuffer, GradBuffer, shift

def backward(out, initial_grad=None):
    grads = {id(out): initial_grad if initial_grad is not None else out.ones_like()}
    stack = [out]
    
    while stack:
        node = stack.pop()
        grad = grads[id(node)]
        
        if isinstance(node, GradBuffer) and node.bwd_fn:
            parent_grads = node.bwd_fn(grad)
            for parent, parent_grad in zip(node.parents, parent_grads):
                if parent is None:
                    continue
                pid = id(parent)
                if pid in grads:
                    grads[pid] = grads[pid] + parent_grad
                else:
                    grads[pid] = parent_grad
                    if isinstance(parent, GradBuffer):
                        stack.append(parent)
    return grads

def grad(fn):
    def wrapper(*args):
        args = [shift(arg) for arg in args]
        out = fn(*args)
        grads = backward(out)
        # Wrap the gradients as GradBuffer for higher-order differentiation
        out_grads = [shift(grads.get(id(arg), None)) for arg in args]
        return out_grads[0] if len(out_grads) == 1 else out_grads
    return wrapper

