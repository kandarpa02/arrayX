from typing import Callable
import numpy as np
from xtorch.core.Array import ArrayImpl, shift

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


def grad(fn, order=1):
    def wrapper(*args):
        for x in args:
            buf = x._rawbuffer
            if isinstance(buf, float):
                continue
            if isinstance(buf, np.ndarray) and np.issubdtype(buf.dtype, np.floating):
                continue
            if isinstance(buf, np.generic) and np.issubdtype(buf.dtype, np.floating):
                continue
            raise TypeError(f"grad requires only float inputs, found {buf.dtype if hasattr(buf, 'dtype') else type(buf)}")

        args = [shift(arg) for arg in args]
        out = fn(*args)
        grads = backward(out)
        # Return gradients for each input argument
        out_grads = [shift(grads.get(id(arg), shift(arg.zero_like()))) for arg in args]
        return out_grads[0] if len(out_grads) == 1 else out_grads
    if order == 1:
        return wrapper
    else:
        for _ in range(order-1):
            wrapper = grad(wrapper)
        return wrapper
