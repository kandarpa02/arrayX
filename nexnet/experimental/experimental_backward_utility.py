# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.


from nexnet._src.autograd import Tape, TapeContext
from typing import Callable

def check_dict(x):
    if isinstance(x, dict):
        return x.values()
    else:
        return x
    

def _compute(fn: Callable, safe=False, end_node:int=-1):
   
    def wrapped_function(args:list|tuple|dict):
        import torch, nexnet
        torch.set_grad_enabled(False)  # Disables PyTorch autograd to avoid interference

        if not nexnet.record_tape.is_enabled():
            raise RuntimeError(f"node history is empty, that has likely caused by disabling neo.record_tape()"
                            "\n in that case set it True neo.record_tape.set(True)")
        
        tape = Tape()
        TapeContext.push(tape)

        # Evaluate the function with inputs; trace begins here

        if isinstance(args, (tuple, list)):
            _out = fn(*args)
        elif isinstance(args, dict):
            _out = fn(args)
        else:
            raise TypeError(f"input type [{type(args)}] is not supported, expected {list}, {tuple} or {dict}")
        
        # PICKING THE END NODE FOR GRAD
        if isinstance(_out, tuple):
            out = _out[end_node]
        else:
            out = _out

        if not hasattr(out, 'data'):
            print(out)
            raise TypeError(
                f"_compute `fn` to return a scalar-like LiteTensor, "
                f"but got {type(out)}: {out}"
        )
        TapeContext.pop()

        out_grad = torch.ones_like(out.data)
        grad_dict = {id(out): out_grad}

        any_cuda = out_grad.is_cuda  

        for node in reversed(tape):
            node_out_id = id(node.output)
            node_out_grad = grad_dict.pop(node_out_id, None)
            if node_out_grad is None:
                continue

            grads = node.bwd_fn(grad=node_out_grad)

            # Cleanup: free references from graph
            node.output = None
            node.bwd_fn = None

            if grads is None:
                node.parents = None
                continue

            if not isinstance(grads, tuple):
                grads = (grads,)
            
            # Pad missing grads with None (e.g., unused inputs)
            if len(grads) < len(node.parents):
                grads = grads + (None,) * (len(node.parents) - len(grads))

            for parent, grad in zip(node.parents, grads):
                if grad is None:
                    continue

                if grad.is_cuda:
                    any_cuda = True

                pid = id(parent)
                # Accumulate gradients. `safe` controls whether to clone before modifying.
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
                input_grads[arg] = grad

        if any_cuda:
            torch.cuda.empty_cache()

        grads_list = list(input_grads.values())
        grad_out = grads_list[0] if len(grads_list) == 1 else grads_list

        return _out, grad_out

    return wrapped_function

def value_and_grad(fn: Callable, safe=False, end_node:int=-1):
    return _compute(fn, safe=safe, end_node=end_node)

def grad(fn: Callable, safe=False, end_node:int=-1):
    def wrapper(args:list|dict|tuple):
        _, grads = _compute(fn, safe=safe, end_node=end_node)(args)
        return grads
    return wrapper
