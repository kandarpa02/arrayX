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
        import gc
        import torch
        torch.set_grad_enabled(False)
        gc.collect()

        tape = Tape()
        TapeContext.push(tape)
        out = fn(*args)
        TapeContext.pop()
        

        out_grad = neolib.ones_like(out.data)
        grad_dict = {id(out): out_grad}

        any_cuda = out_grad.is_cuda
        leaky_nodes = []

        for node in reversed(tape):
            node_out_id = id(node.output)
            node_out_grad = grad_dict.pop(node_out_id, None)

            if node_out_grad is None:
                try:
                    shape = tuple(node.output.shape)
                except Exception:
                    shape = "unknown"
                print(f"[UNUSED] Node output id={node_out_id}, shape={shape}")
                
                # Track in leaky list
                leaky_nodes.append(node)

                # Clear aggressively
                node.output = None
                node.bwd_fn = None
                node.parents = None
                del node
                continue

            with torch.no_grad():
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
                existing_grad = grad_dict.get(pid)
                if existing_grad is not None:
                    existing_grad.add_(grad)
                else:
                    grad_dict[pid] = grad.clone()

                del grad

            node.parents = None
            del node

        input_grads = {}
        for arg in args:
            grad = grad_dict.get(id(arg))
            if grad is not None:
                input_grads[arg] = grad

        if any_cuda:
            torch.cuda.empty_cache()

        if leaky_nodes:
            print(f"\n[LEAKY] Total unused nodes: {len(leaky_nodes)}")

        for arg in args:
            arg.data = None  # clear backing tensor
        del args             # remove Python-level ref
        grad_dict.clear()
        del grad_dict
        del tape
        gc.collect() # force garbage collection

        return out, input_grads

    return wrapped_function
