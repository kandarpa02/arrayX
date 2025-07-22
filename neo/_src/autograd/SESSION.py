from neo._src.autograd import Node, Tape, TapeContext
from neo._src.autograd._graph_cache import GRAPH_CACHE
from neo.backend import get_xp
from typing import Callable
import numpy as np
import hashlib
import pickle

def define_device(x):
    return 'cpu' if isinstance(x, np.ndarray) else 'cuda'

def get_cache_key(fn: Callable, args: tuple):
    try:
        key_data = (
            fn.__name__,
            tuple((id(a), getattr(a, "value", None).shape) for a in args),
        )
        return hashlib.sha1(pickle.dumps(key_data)).hexdigest()
    except Exception as e:
        return None

def value_and_grad(fn: Callable):
    def wrapped(*args):
        device = define_device(args[0].value)
        xp = get_xp(device=device)

        key = get_cache_key(fn, args)
        use_cache = key in GRAPH_CACHE if key else False

        if use_cache:
            tape_nodes = GRAPH_CACHE[key]
            # You must run a fresh forward to get actual outputs
            TapeContext.push([])
            out = fn(*args)
            tape_nodes = TapeContext.pop()
        else:
            tape = Tape()
            TapeContext.push(tape.nodes)
            out = fn(*args)
            TapeContext.pop()

            # Cache a sanitized copy
            if key:
                GRAPH_CACHE[key] = [
                    Node(output=None, parents=node.parents, bwd_fn=node.bwd_fn)
                    for node in tape.nodes
                ]

            tape_nodes = tape.nodes  # use real tape for this run

                    # if key:
                    #     GRAPH_CACHE[key] = sanitized_nodes
                    # tape_nodes = tape.nodes  # still use the original (fresh) one for this run

        # Always run forward again for fresh values
        TapeContext.push([])
        out = fn(*args)
        TapeContext.pop()

        out_grad = xp.ones_like(out.value, dtype=out.value.dtype)
        grad_dict = {id(out): out_grad}

        for node in reversed(tape_nodes):
            node_out_grad = grad_dict.get(id(node.output))
            if node_out_grad is None:
                continue

            grad_inputs = node.bwd_fn(grad=node_out_grad)
            if grad_inputs is None:
                continue

            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)

            for parent, grad in zip(node.parents, grad_inputs):
                if grad is None:
                    continue
                pid = id(parent)
                grad_dict[pid] = grad_dict.get(pid, 0) + grad

        input_grads = {
            arg: grad_dict[id(arg)]
            for arg in args
            if id(arg) in grad_dict
        }

        return out, input_grads

    return wrapped
