# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.

"""
This file implements the internal backward-mode automatic differentiation utility
used by NeoNet. It constructs the computation graph using `Tape`, then performs
reverse-mode gradient propagation with optional memory safety.

Note: The gradient engine expects all user-defined operations to return scalar-like
LiteTensors. Non-scalar outputs are not currently supported for autodiff. If violated,
execution will raise immediately to avoid silent miscomputations.

"""

from neo._src.autograd import Tape, TapeContext
from typing import Callable, List, Any
from neo._torch import neolib
from neo._torch.lite_tensor import LiteTensor
from neo._torch.user_functions import lite

def check_dict(x):
    if isinstance(x, dict):
        return x.values()
    else:
        return x
    
def is_scalar(x):
    try:
        return x.reshape([])
    except:
        raise RuntimeError(f"object {lite(x)} shape={x.cpu().numpy().shape} is not a scaler")

def _compute(fn: Callable, safe=False, end_node:int=-1):
    """
    Builds the computation graph and runs backward pass to compute gradients
    with respect to inputs.

    Args:
        fn (Callable): The user-defined function to differentiate. It must return
                       a scalar-like `LiteTensor`. Any non-scalar output will trigger
                       a runtime error.
        safe (bool): If True, clones intermediate gradients before accumulation
                     to avoid in-place side effects during backward. This is slower
                     but useful for debugging numerical instabilities.

    Returns:
        Callable: A wrapped function that takes (list | tuple | dict) inputs and returns
                  (output, gradients). Gradients are returned as a single LiteTensor
                  if there's one input, else a list of LiteTensors.

    Design notes:
        - Only tensors participating in `function(...)` are recorded in the Tape.
        - Gradient accumulation is in-place by default for performance.
        - The backward pass skips null gradients and safely releases memory references.
        - CUDA memory is explicitly cleared at the end if any gradients touched GPU.

    Known Limitations:
        - Does not support higher-order gradients (yet).
        - Assumes the forward pass returns a single output node.
        - Inputs must be wrapped as `LiteTensor`s and must appear in the function call.
    """

    def wrapped_function(args:list|tuple|dict):
        import torch, neo
        torch.set_grad_enabled(False)  # Disables PyTorch autograd to avoid interference

        if not neo.record_tape.is_enabled():
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

        out_grad = neolib.ones_like(is_scalar(out.data))
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
                input_grads[arg] = LiteTensor(grad)

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

class build_computation_graph:
    """
    Low-level context manager for computing gradients via NeoNet's autograd engine.

    Usage:
        def loss(x): ...
        g = build_computation_graph(loss, inputs=[x])
        g.backward()
        output, grads = g.out, g.grad

    Args:
        function (Callable): Optional function to bind immediately.
        inputs (list | tuple | dict): Inputs to the function. Must be LiteTensors.
        safe (bool): Whether to clone gradients before accumulation.

    Call Behavior:
        You can also use `g = build_computation_graph(...)(fn)` to bind after construction.

    Note:
        This is not the public API. Use `neo.grad(...)` or `neo.value_and_grad(...)`
        for standard gradient use cases. This exists for internal control over tracing,
        memory, and side-effect handling.

    WARNING:
        Gradient correctness is your responsibility if you bypass `function(...)`
        or manually manipulate the Tape. If you're seeing silent NaNs, check for:
        - In-place ops during forward
        - Non-scalar outputs
        - Mixed device types in ops
        - Implicit detach during `with torch.no_grad()`
    """

    def __init__(self, function:Callable=None, inputs:list|tuple|dict=None, safe=False, end_node:int=-1): #type: ignore
        self._function = function
        self._variables = inputs
        self.safe = safe
        self.end_node = end_node
        self.out, self.grad = None, None

    def backward(self):
        self.out, self.grad = _compute(self._function, safe=self.safe, end_node=self.end_node)(self._variables)

    def __call__(self, fn):
        self._function = fn
        self.out, self.grad = _compute(fn, safe=self.safe, end_node=self.end_node)(self._variables)
        return self
    
class Curves:
    def __init__(self, persistent=False):
        self.persistent = persistent
        self._grads = None
        self._func = None
        self._inputs = None
        self._used = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.persistent and self._used:
            self._grads = None
            self._func = None
            self._inputs = None

    def lift_them(self, func, *inputs):
        self._func = func
        self._inputs = inputs

    def gradient(self, end_node, wrt, safe=False):
        if self._func is None:
            raise RuntimeError("No function was watched. Did you call tape.watch(fn, inputs)?")
        grad_fn = _compute(lambda *x: self._func(*x), safe=safe, end_node=end_node)

        inputs = self._inputs if self._inputs is not None else (wrt,)
        if not isinstance(inputs, (list, tuple, dict)):
            inputs = (inputs,)

        _, self._grads = grad_fn(inputs)
        self._used = True
        return self._grads