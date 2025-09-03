# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport intptr_t
from nexnet._src.autograd.GRAPH_MANAGER cimport Node, Tape 

cpdef dict run_backward(Tape tape, object out, object out_grad, bint safe):
    """
    Optimized backward traversal loop (still uses torch.Tensor as object).
    """
    cdef dict grad_dict = {}
    cdef intptr_t pid, nid
    cdef Node node
    cdef object node_out_grad, grads, grad, parent

    grad_dict[id(out)] = out_grad

    # Traverse backwards
    for node in tape.nodes[::-1]:
        nid = id(node.output)
        node_out_grad = grad_dict.pop(nid, None)
        if node_out_grad is None:
            continue

        grads = node.bwd_fn(node_out_grad)

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

            pid = id(parent)

            grad = grad.data.clone() if safe else grad.data

            if pid in grad_dict:
                grad_dict[pid].add_(grad)
            else:
                grad_dict[pid] = grad

        node.parents = None

    return grad_dict
