from neo._src.autograd import Node, Tape, TapeContext
from typing import Callable
import numpy as np
# from neonet.numpy import array
# from neonet.backend import get_xp

def define_device(x):
    device = 'cpu'
    if not isinstance(x.value, np.ndarray):
        device = 'cuda'
    return device

def rectify_shapes(val):
    val = val.reshape(1) if (val.ndim < 1) else val
    return val

def unpack_tuple(tup):
        variables = {f'x{i+1}': value for i, value in enumerate(tup)}
        return variables


def if_xnary(grads):
    def _fix(g):
        if g.ndim == 0:
            return g.reshape(1)
        elif g.ndim == 1:
            return g[None, :]  # promotes to (1, D) if needed
        return g


    if isinstance(grads, tuple):
        return tuple(_fix(g) for g in grads)
    else:
        return _fix(grads)


def value_and_grad(fn: Callable):
    def wrapped_function(*args):
        tape = Tape()
        TapeContext.push(tape.nodes)
        out = fn(*args)
        device = define_device(out)
        TapeContext.pop()

        grad_dict = {}
        grad_dict[id(out)] = out.ones_like().value

        for node in reversed(tape.nodes):
            node_out_grad = grad_dict.get(id(node.output))
            if node_out_grad is None:
                continue
                
            grad_inputs = node.bwd_fn(
                grad=node_out_grad
                )
            
            if grad_inputs is None:
                continue

            grad_inputs = if_xnary(grads=grad_inputs)
                
            for parent, grad in zip(node.parents, grad_inputs):
                pid = id(parent)
                if pid in grad_dict:
                    grad_dict[pid] += grad
                else:
                    grad_dict[pid] = grad
        
        arg_grads = {arg: grad_dict.get(id(arg), 0) for arg in args}
        return out, arg_grads

    return wrapped_function

