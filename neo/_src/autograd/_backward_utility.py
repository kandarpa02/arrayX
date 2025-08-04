# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.

from neo._src.autograd import Tape, TapeContext
from typing import Callable, List, Any
from neo._torch import neolib
from neo._torch.lite_tensor import LiteTensor

def check_dict(x):
    if isinstance(x, dict):
        return x.values()
    else:
        return x

def _computing_value_and_grad(fn: Callable, safe=False):
    def wrapped_function(args:list|tuple|dict):
        import torch
        torch.set_grad_enabled(False)

        tape = Tape()
        TapeContext.push(tape)
        if isinstance(args, (tuple, list)):
            out = fn(*args)
        else:
            out = fn(*list(args.values()))
        if not hasattr(out, 'data'):
            print(out)
            raise TypeError(
                f"value_and_grad expected `fn` to return a scalar-like LiteTensor, "
                f"but got {type(out)}: {out}"
        )
        TapeContext.pop()

        out_grad = neolib.ones_like(out.data)
        grad_dict = {id(out): out_grad}

        any_cuda = out_grad.is_cuda  

        for node in reversed(tape):
            node_out_id = id(node.output)
            node_out_grad = grad_dict.pop(node_out_id, None)
            if node_out_grad is None:
                continue

            grads = node.bwd_fn(grad=node_out_grad)

            node.output = None
            node.bwd_fn = None

            if grads is None:
                node.parents = None
                continue

            if not isinstance(grads, tuple):
                grads = (grads,)
            if len(grads) < len(node.parents):
                grads = grads + (None,) * (len(node.parents) - len(grads))

            for parent, grad in zip(node.parents, grads):
                if grad is None:
                    continue

                if grad.is_cuda:
                    any_cuda = True

                pid = id(parent)
                if pid in grad_dict:
                    grad_dict[pid].add_(grad.clone() if safe else grad)
                else:
                    grad_dict[pid] = grad.clone() if safe else grad

                del grad  

            node.parents = None 
            del node  

        input_grads = {}
        for arg in check_dict(args):
            grad = grad_dict.get(id(arg))
            if grad is not None:
                input_grads[arg] = LiteTensor(grad)

        if any_cuda:
            torch.cuda.empty_cache()

        grads_list = list(input_grads.values())
        grad_out = grads_list[0] if len(grads_list) == 1 else grads_list

        return out, grad_out

    return wrapped_function


class build_computation_graph:
    def __init__(self, function:Callable=None, inputs:list|tuple|dict=None): #type: ignore
        self._function = function
        self._variables = inputs
        self.val, self.grad = None, None

    def backward(self):
        self.val, self.grad = _computing_value_and_grad(self._function)(self._variables)

    def output(self):
        return self.val
    
    def gradient(self):
        return self.grad
    
    def __call__(self, fn):
        self._function = fn
        self.val, self.grad = _computing_value_and_grad(fn)(self._variables)
        return self