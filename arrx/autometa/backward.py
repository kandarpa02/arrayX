from typing import List, Tuple, Optional, Callable, Any
from collections.abc import Mapping
from itertools import chain
from arrx import lib
from arrx.Core.Array import ArrayImpl, shift

# Localize for speed
_lib_ndarray = lib.ndarray
_ArrayImpl = ArrayImpl
_is = isinstance


# flatten_leaves (same as before)
def flatten_leaves(obj):
    leaves = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Mapping):
            vals = list(cur.values())
            for v in reversed(vals):
                stack.append(v)
        elif isinstance(cur, (list, tuple)):
            for v in reversed(cur):
                stack.append(v)
        else:
            leaves.append(cur)
    return leaves


# check helpers (same)
def is_float_buffer(buf):
    if isinstance(buf, float):
        return True
    try:
        return buf.dtype.kind == 'f'
    except Exception:
        return False


def check_raw_tensor(a):
    if _is(a, (_lib_ndarray, int, float)):
        return a
    if _is(a, _ArrayImpl):
        return a._rawbuffer
    raise ValueError(f"given {a} of type {type(a)} is not supported")


# Indexed graph builder (produces topo list: parents before children)
def build_indexed_graph(out) -> Tuple[List[Any], dict, List[int], List[int], List[Optional[Callable]]]:
    """
    Build index-based representation for the graph that ends at `out`.
    Returns:
      nodes: list of node objects in topo order (parents before child)
      id_to_idx: mapping id(node) -> index
      parents_flat: flat parent indices with -1 for None
      parents_offset: start index (into parents_flat) for each node
      bwd_fns: list of backward functions (or None) per node index
    """
    # Iterative DFS that appends node after processing parents -> postorder (parents first)
    seen = set()
    stack = [(out, False)]
    topo = []

    while stack:
        node, done = stack.pop()
        if node is None:
            continue
        nid = id(node)
        if done:
            topo.append(node)
            continue
        if nid in seen:
            continue
        seen.add(nid)
        # mark node to append after its parents
        stack.append((node, True))
        # push parents so they are processed before node
        if _is(node, _ArrayImpl):
            parents = node.parents
            for p in reversed(parents):
                if p is not None and id(p) not in seen:
                    stack.append((p, False))

    # topo now has parents before children; use it as nodes list
    nodes = list(topo)
    id_to_idx = {id(n): i for i, n in enumerate(nodes)}

    parents_flat: List[int] = []
    parents_offset: List[int] = [0] * len(nodes)
    bwd_fns: List[Optional[Callable]] = [None] * len(nodes)

    for i, node in enumerate(nodes):
        parents_offset[i] = len(parents_flat)
        if _is(node, _ArrayImpl):
            bwd_fns[i] = getattr(node, "bwd_fn", None)
            for p in node.parents:
                parents_flat.append(-1 if p is None else id_to_idx.get(id(p), -1))
        else:
            bwd_fns[i] = None

    return nodes, id_to_idx, parents_flat, parents_offset, bwd_fns


# -------------------------
# Fast indexed backward (drop-in replacement for backward)
# -------------------------
def backward(out, initial_grad=None):
    """
    Fast backward that returns a dict mapping id(node) -> gradient (same API as before).
    Uses an index-based propagation to avoid dict lookups in hot inner loops.
    """
    if out is None:
        return {}

    nodes, id_to_idx, parents_flat, parents_offset, bwd_fns = build_indexed_graph(out)
    n = len(nodes)
    grads_list: List[Optional[Any]] = [None] * n

    # seed output gradient
    out_idx = id_to_idx[id(out)]
    grads_list[out_idx] = initial_grad if initial_grad is not None else out.ones_like()

    # compute counts for each node (fast)
    counts: List[int] = [0] * n
    plen = len(parents_flat)
    for i in range(n):
        start = parents_offset[i]
        end = parents_offset[i + 1] if (i + 1) < n else plen
        counts[i] = end - start

    # propagate in reverse topo order (children -> parents)
    for i in range(n - 1, -1, -1):
        bwd = bwd_fns[i]
        if bwd is None:
            continue
        g = grads_list[i]
        if g is None:
            continue
        start = parents_offset[i]
        cnt = counts[i]
        # call user backward
        parent_grads = bwd(g)
        # assume parent_grads is an iterable matching cnt
        j = start
        for pg in parent_grads:
            parent_idx = parents_flat[j]
            j += 1
            if parent_idx == -1:
                continue
            existing = grads_list[parent_idx]
            if existing is None:
                grads_list[parent_idx] = pg
            else:
                # prefer in-place if ArrayImpl supports it; fall back to + (creates new object)
                try:
                    # try in-place add if available (safe if semantics permit)
                    existing += pg
                    grads_list[parent_idx] = existing
                except Exception:
                    grads_list[parent_idx] = existing + pg

    # return dict mapping id(node) -> grad (only for nodes that have a grad)
    id_grads = {id(nodes[i]): grads_list[i] for i in range(n) if grads_list[i] is not None}
    return id_grads


# shift_structure, grad_structure_from_shifted, grad, value_and_grad
def _is_mapping(obj):
    return isinstance(obj, Mapping)


def _is_sequence(obj):
    return isinstance(obj, (list, tuple))


def shift_structure(obj):
    if _is_mapping(obj):
        return {k: shift_structure(v) for k, v in obj.items()}
    if _is_sequence(obj):
        seq = [shift_structure(v) for v in obj]
        return tuple(seq) if isinstance(obj, tuple) else seq
    return shift(obj)


def grad_structure_from_shifted(obj, grads_dict):
    if _is_mapping(obj):
        return {k: grad_structure_from_shifted(v, grads_dict) for k, v in obj.items()}
    if _is_sequence(obj):
        seq = [grad_structure_from_shifted(v, grads_dict) for v in obj]
        return tuple(seq) if isinstance(obj, tuple) else seq
    g = grads_dict.get(id(obj))
    if g is None:
        return shift(obj.zero_like())
    return shift(g)


def grad(fn: Callable, order: int = 1, last_node: int = -1):
    def make_wrapper(f):
        def wrapper(*args):
            flat_args = flatten_leaves(args)
            for x in flat_args:
                buf = check_raw_tensor(x)
                if is_float_buffer(buf):
                    continue
                raise TypeError(f"grad requires only float inputs, found {getattr(buf, 'dtype', type(buf))}")

            shifted_args = [shift_structure(arg) for arg in args]
            _out = f(*shifted_args)
            out = _out[last_node] if isinstance(_out, tuple) else _out
            grads = backward(out)
            shifted_leaves = flatten_leaves(shifted_args)
            out_grads = []
            for leaf in shifted_leaves:
                g = grads.get(id(leaf))
                if g is None:
                    out_grads.append(shift(leaf.zero_like()))
                else:
                    out_grads.append(shift(g))
            if len(out_grads) < 2:
                return out_grads[0]
            return out_grads
        return wrapper

    if order == 1:
        return make_wrapper(fn)
    w = make_wrapper(fn)
    for _ in range(order - 1):
        w = grad(w)
    return w


def value_and_grad(fn: Callable, last_node: int = -1):
    def wrapper(*args):
        flat_args = flatten_leaves(args)
        for x in flat_args:
            buf = check_raw_tensor(x)
            if is_float_buffer(buf):
                continue
            raise TypeError(f"grad requires only float inputs, found {getattr(buf, 'dtype', type(buf))}")

        shifted_args = [shift_structure(arg) for arg in args]
        _out = fn(*shifted_args)
        out = _out[last_node] if isinstance(_out, tuple) else _out
        grads = backward(out)
        shifted_leaves = flatten_leaves(shifted_args)
        out_grads = []
        for leaf in shifted_leaves:
            g = grads.get(id(leaf))
            if g is None:
                out_grads.append(shift(leaf.zero_like()))
            else:
                out_grads.append(shift(g))
        g_out = out_grads[0] if len(out_grads) < 2 else out_grads
        return _out, g_out
    return wrapper
