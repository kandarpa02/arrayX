from typing import Callable, Set
from collections.abc import Mapping, Iterable
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


def flatten_args(args):
    flat = []
    if isinstance(args, Mapping):  # dict-like
        for v in args.values():
            flat.extend(flatten_args(v))
    elif isinstance(args, (list, tuple)):
        for item in args:
            flat.extend(flatten_args(item))
    else:
        flat.append(args)
    return flat

def _is_mapping(obj):
    return isinstance(obj, Mapping)

def _is_sequence(obj):
    # treat tuples/lists as sequences; don't treat str/bytes as sequence
    return isinstance(obj, (list, tuple))

def shift_structure(obj):
    """Recursively apply shift to leaves and preserve structure."""
    if _is_mapping(obj):
        return {k: shift_structure(v) for k, v in obj.items()}
    if _is_sequence(obj):
        seq = [shift_structure(v) for v in obj]
        return tuple(seq) if isinstance(obj, tuple) else seq
    # leaf: apply shift (wrap raw arrays / scalars / ArrayImpl -> ArrayImpl)
    return shift(obj)

def grad_structure_from_shifted(obj, grads_dict):
    """
    Build parallel structure of gradients for the *shifted* obj.
    `obj` here should be the shifted structure (so leaves are ArrayImpl).
    `grads_dict` is the dict returned by backward(): keys are ids of nodes.
    """
    if _is_mapping(obj):
        return {k: grad_structure_from_shifted(v, grads_dict) for k, v in obj.items()}
    if _is_sequence(obj):
        seq = [grad_structure_from_shifted(v, grads_dict) for v in obj]
        return tuple(seq) if isinstance(obj, tuple) else seq

    # leaf: expected to be ArrayImpl (shift returned ArrayImpl)
    try:
        # get gradient ArrayImpl from grads_dict using id; fallback to zero-like
        g = grads_dict.get(id(obj))
    except Exception:
        g = None

    if g is None:
        # if no gradient found, return a zero-like wrapped ArrayImpl
        return shift(obj.zero_like())
    else:
        # ensure returned object is an ArrayImpl (shift is safe if g already ArrayImpl)
        return shift(g)
    

def grad(fn, order=1, last_node=-1):
    def wrapper(*args):
        # Validate all leaf args (flatten without changing structure)
        flat_args = flatten_args(args)
        for x in flat_args:
            buf = check_raw_tensor(x)
            if is_float_buffer(buf):
                continue
            raise TypeError(f"grad requires only float inputs, found {buf.dtype if hasattr(buf, 'dtype') else type(buf)}")

        # Shift only leaves, preserve top-level structure
        shifted_args_list = [shift_structure(arg) for arg in args]

        # Call function with shifted structures
        _out = fn(*shifted_args_list)
        out = _out[last_node] if isinstance(_out, tuple) else _out

        # Backprop
        grads = backward(out)

        # Flatten the shifted args into leaves (these are ArrayImpl nodes)
        shifted_leaves = flatten_args(shifted_args_list)

        # Collect gradients for each leaf in order (fallback to zero-like)
        out_grads = []
        for leaf in shifted_leaves:
            g = grads.get(id(leaf))
            if g is None:
                out_grads.append(shift(leaf.zero_like()))
            else:
                out_grads.append(shift(g))

        # Always return a flattened list of grads
        return out_grads

    # Support higher-order grads by wrapping repeatedly (keeps prior pattern)
    if order == 1:
        return wrapper
    else:
        w = wrapper
        for _ in range(order - 1):
            w = grad(w)  # repeated wrapping (same convention as before)
        return w



def value_and_grad(fn, last_node=-1):
    def wrapper(args, *more_args):
        # Combine args and more_args into a single structure for validation
        all_args = (args,) + more_args if more_args else args

        # Flatten args only for validation (keep original structure for calling)
        flat_args = flatten_args(all_args)
        for x in flat_args:
            buf = check_raw_tensor(x)
            if is_float_buffer(buf):
                continue
            raise TypeError(f"grad requires only float inputs, found {buf.dtype if hasattr(buf, 'dtype') else type(buf)}")

        # Prepare the argument list preserving structure, but with leaves shifted
        if more_args:
            orig_args_list = [args] + list(more_args)
        else:
            orig_args_list = [args]

        shifted_args_list = [shift_structure(arg) for arg in orig_args_list]

        # Call the function with shifted structures (so arrays are ArrayImpl leaves)
        _out = fn(*shifted_args_list)
        out = _out[last_node] if isinstance(_out, tuple) else _out

        # Run backward and collect grads as a single flattened list
        grads = backward(out)

        # Flatten the shifted args into leaves (these are the ArrayImpl nodes used in the graph)
        shifted_leaves = flatten_args(shifted_args_list)

        # For each leaf, fetch gradient by id; fallback to zero-like if not found.
        # Use shift(...) to ensure return values are ArrayImpl (or consistent wrapper).
        out_grads = []
        for leaf in shifted_leaves:
            g = grads.get(id(leaf))
            if g is None:
                out_grads.append(shift(leaf.zero_like()))
            else:
                out_grads.append(shift(g))

        # always return a list of grads (flattened), per your request. 
        return _out, out_grads

    return wrapper
