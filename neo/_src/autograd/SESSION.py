# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.

from neo._src.autograd import Node, Tape, TapeContext
from typing import Callable
from neo._torch import neolib

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
    

def value_and_grad(fn: Callable):
    def wrapped_function(*args):
        tape = Tape()
        TapeContext.push(tape)
        out = fn(*args)
        TapeContext.pop()

        out_grad = neolib.ones_like(out.data)

        grad_dict = {id(out): out_grad}
        
        # for node in reversed(tape):
        #     node_out_grad = grad_dict.get(id(node.output))
        #     if node_out_grad is None:
        #         continue

        #     grad_inputs = node.bwd_fn(grad=node_out_grad)
        #     if grad_inputs is None:
        #         continue

        #     if not isinstance(grad_inputs, tuple):
        #         grad_inputs = (grad_inputs,)

        #     if len(grad_inputs) < len(node.parents):
        #         grad_inputs += (None,) * (len(node.parents) - len(grad_inputs))
                
        #     for parent, grad in zip(node.parents, grad_inputs):
        #         if grad is None:
        #             continue

        #         pid = id(parent)
        #         if pid in grad_dict:
        #             grad_dict[pid].add_(grad) 
        #         else:
        #             grad_dict[pid] = grad.clone()  
        for node in reversed(tape):
            node_out_grad = grad_dict.get(id(node.output))
            if node_out_grad is None:
                continue

            grad_inputs = node.bwd_fn(grad=node_out_grad)
            
            node.output = None
            node.bwd_fn = None

            if grad_inputs is None:
                continue

            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)

            if len(grad_inputs) < len(node.parents):
                grad_inputs += (None,) * (len(node.parents) - len(grad_inputs))

            for parent, grad in zip(node.parents, grad_inputs):
                if grad is None:
                    continue

                pid = id(parent)
                if pid in grad_dict:
                    grad_dict[pid].add_(grad)
                else:
                    grad_dict[pid] = grad.clone()

                del grad

            node.parents = None

        input_grads = {}
        for arg in args:
            grad = grad_dict.get(id(arg))
            if grad is not None:
                input_grads[arg] = grad

        return out, input_grads

    return wrapped_function
