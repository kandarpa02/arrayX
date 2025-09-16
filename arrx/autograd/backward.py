from typing import Callable, Set
from arrx import lib
from arrx.core.Array import ArrayImpl, shift

def backward(out, initial_grad=None):
    # Step 0: initialize gradient dict
    grads = {id(out): initial_grad if initial_grad is not None else out.ones_like()}

    # Step 1: collect all nodes via DFS (topo sort)
    visited = set()
    topo_nodes = []

    def dfs(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, ArrayImpl):
            for parent in node.parents:
                dfs(parent)
        topo_nodes.append(node)

    dfs(out)

    # Step 2: propagate gradients in reverse topo order
    for node in reversed(topo_nodes):
        if not isinstance(node, ArrayImpl) or node.bwd_fn is None:
            continue
        grad = grads.get(id(node))
        parent_grads = node.bwd_fn(grad)
        for parent, parent_grad in zip(node.parents, parent_grads):
            if parent is None:
                continue
            pid = id(parent)
            if pid in grads:
                grads[pid] = grads[pid] + parent_grad
            else:
                grads[pid] = parent_grad

    return grads


def is_float_buffer(buf):
    if isinstance(buf, float):
        return True
    try:
        return buf.dtype.kind == 'f'
    except AttributeError:
        return False
    

def check_raw_tensor(a):
    if isinstance(a, lib.ndarray|int|float):
        return a
    elif isinstance(a, ArrayImpl):
        return a._rawbuffer
    else:
        raise ValueError(f"given {a} of type {type(a)} is not supported")
    

def shift_vals(inp):
    if isinstance(inp, dict):
        for i, j in inp.items():
            inp[i] = shift(j)
        return inp
    else:
        return [shift(_in) for _in in inp]


def grad(fn, order=1, last_node=-1):
    def wrapper(args: list|tuple|dict):
        _args = args if isinstance(args, list|tuple) else list(args.values())
        for x in _args:
            buf = check_raw_tensor(x)
            if is_float_buffer(buf):
                continue
            raise TypeError(f"grad requires only float inputs, found {buf.dtype if hasattr(buf, 'dtype') else type(buf)}")

        args = shift_vals(args)

        _out = fn(args)
        out = _out[last_node] if isinstance(_out, tuple) else _out
        grads = backward(out)
        out_grads = [shift(grads.get(id(arg), shift(arg.zero_like()))) for arg in _args]
        return out_grads[0] if len(out_grads) == 1 else out_grads

    if order == 1:
        return wrapper
    else:
        for _ in range(order-1):
            wrapper = grad(wrapper)
        return wrapper


def value_and_grad(fn, last_node=-1):
    def wrapper(args: list|tuple|dict):
        _args = args if isinstance(args, list|tuple) else list(args.values())
        for x in _args:
            buf = check_raw_tensor(x)
            if is_float_buffer(buf):
                continue
            raise TypeError(f"grad requires only float inputs, found {buf.dtype if hasattr(buf, 'dtype') else type(buf)}")

        args = shift_vals(args)

        _out = fn(args)
        out = _out[last_node] if isinstance(_out, tuple) else _out

        grads = backward(out)
        # Return gradients for each ilibut argument
        out_grads = [shift(grads.get(id(arg), shift(arg.zero_like()))) for arg in _args]
        return _out, out_grads[0] if len(out_grads) == 1 else out_grads
    
    return wrapper