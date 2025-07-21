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
            return g[None, :]  # e.g. (D,) → (1, D)
        return g

    if isinstance(grads, tuple):
        return tuple(_fix(g) for g in grads)
    else:
        return _fix(grads)



def value_and_grad(fn: Callable, debug=False):
    def wrapped_function(*args):
        # Record original shapes
        original_shapes = [arg.shape if hasattr(arg, 'shape') else () for arg in args]
        
        tape = Tape()
        TapeContext.push(tape.nodes)
        out = fn(*args)
        TapeContext.pop()

        device = define_device(out.value)
        xp = get_xp(device=device)

        # Create gradient with proper shape
        out_grad = xp.ones_like(out.value, dtype=out.value.dtype)
        grad_dict = {id(out): out_grad}
        
        if debug:
            print("Initial grad_dict:", grad_dict)

        # Reverse-mode autodiff
        for node in reversed(tape.nodes):
            node_out_grad = grad_dict.get(id(node.output))
            if node_out_grad is None:
                continue

            grad_inputs = node.bwd_fn(grad=node_out_grad)
            if grad_inputs is None:
                continue

            grad_inputs = if_xnary(grad_inputs)

            for parent, grad in zip(node.parents, grad_inputs):
                pid = id(parent)
                existing_grad = grad_dict.get(pid)
                
                if existing_grad is not None:
                    # Shape-aware gradient accumulation
                    try:
                        # Try to add directly if shapes match
                        grad_dict[pid] = existing_grad + grad
                    except ValueError:
                        # Broadcast gradients to match original shape
                        target_shape = existing_grad.shape
                        grad_dict[pid] = existing_grad + grad.reshape(target_shape)
                else:
                    grad_dict[pid] = grad

        # Restore original shapes for gradients
        arg_grads = []
        for i, arg in enumerate(args):
            grad = grad_dict.get(id(arg))
            if grad is not None:
                # Reshape to match original input shape
                if grad.shape != original_shapes[i]:
                    try:
                        grad = grad.reshape(original_shapes[i])
                    except:
                        if debug:
                            print(f"Shape mismatch: grad {grad.shape} vs original {original_shapes[i]}")
            else:
                grad = 0
                
            arg_grads.append(grad)

        return out, tuple(arg_grads)

    return wrapped_function
