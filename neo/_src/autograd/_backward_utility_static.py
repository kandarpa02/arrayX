from typing import Any, Callable, Dict, List, Union
from neo._src.autograd import Tape, TapeContext
from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
import torch

# Helper to flatten/unflatten nested structures (list/dict/tuple)
def tree_flatten(x):
    flat = []
    def _flatten(val):
        if isinstance(val, (list, tuple)):
            return type(val)(_flatten(v) for v in val)
        elif isinstance(val, dict):
            return {k: _flatten(v) for k, v in val.items()}
        else:
            flat.append(val)
            return None
    spec = _flatten(x)
    return flat, spec

def tree_unflatten(flat: List[Any], spec):
    flat_iter = iter(flat)
    def _unflatten(s):
        if isinstance(s, (list, tuple)):
            return type(s)(_unflatten(v) for v in s)
        elif isinstance(s, dict):
            return {k: _unflatten(v) for k, v in s.items()}
        elif s is None:
            return next(flat_iter)
        else:
            raise RuntimeError("Invalid spec node")
    result = _unflatten(spec)
    try:
        next(flat_iter)
        raise RuntimeError("Flat list has extra elements")
    except StopIteration:
        pass
    return result


class StaticPlaceholder:
    __slots__ = ("original_lite", "id")
    def __init__(self, lite: LiteTensor):
        if not isinstance(lite, LiteTensor):
            raise TypeError("StaticPlaceholder expects a LiteTensor")
        self.original_lite = lite
        self.id = id(lite)


class StaticOp:
    __slots__ = ("out_id", "parent_ids", "fwd_callable", "bwd_callable", "node_repr")
    def __init__(self, node):
        self.node_repr = repr(node)
        self.out_id = id(node.output)
        self.parent_ids = tuple(id(p) for p in (node.parents or ()))
        self.bwd_callable = node.bwd_fn
        owner = getattr(node.bwd_fn, "__self__", None)
        self.fwd_callable = getattr(owner, "forward", None) if owner is not None else None


class StaticGraph:
    def __init__(
        self,
        ops: List[StaticOp],
        input_placeholders: List[StaticPlaceholder],
        output_op: StaticOp,
        flat_input_spec: Any,
        flat_input_keys: List[LiteTensor],
    ):
        self.ops = ops
        self.placeholders_list = input_placeholders
        self.placeholders_by_id = {ph.id: ph for ph in input_placeholders}
        self.output_op = output_op
        self._runtime_values: Dict[int, torch.Tensor] = {}

        self.flat_input_spec = flat_input_spec
        self.flat_input_keys = flat_input_keys

    def forward(self, inputs: Any) -> LiteTensor:
        self._runtime_values.clear()

        flat_inputs, input_spec = tree_flatten(inputs)
        if len(flat_inputs) != len(self.flat_input_keys):
            raise ValueError(
                f"Input leaf count mismatch. Expected {len(self.flat_input_keys)}, got {len(flat_inputs)}"
            )
        for i, leaf in enumerate(flat_inputs):
            if not isinstance(leaf, LiteTensor):
                raise TypeError(f"Input leaf {i} is not a LiteTensor, got {type(leaf)}")

        for ph, leaf in zip(self.placeholders_list, flat_inputs):
            self._runtime_values[ph.id] = leaf.data

        for idx, op in enumerate(self.ops):
            args = []
            missing_parent = None
            for pid in op.parent_ids:
                if pid not in self._runtime_values:
                    missing_parent = pid
                    break
                args.append(self._runtime_values[pid])
            if missing_parent is not None:
                raise RuntimeError(
                    f"During static forward, missing runtime value for parent id {missing_parent} "
                    f"needed by op[{idx}] (node={op.node_repr})."
                )

            if op.fwd_callable is None:
                raise RuntimeError(
                    f"No forward callable available for op (node={op.node_repr})."
                )

            out = op.fwd_callable(*args)

            if isinstance(out, LiteTensor):
                out_tensor = out.data
            elif isinstance(out, torch.Tensor):
                out_tensor = out
            else:
                raise TypeError(
                    f"forward callable returned unsupported type {type(out)} for node {op.node_repr}"
                )

            self._runtime_values[op.out_id] = out_tensor

        final = self._runtime_values.get(self.output_op.out_id)
        if final is None:
            raise RuntimeError("StaticGraph forward completed but output value is missing.")
        return LiteTensor(final)

    def backward(self, safe: bool = False) -> Any:
        final_out_id = self.output_op.out_id
        grad_map: Dict[int, torch.Tensor] = {}
        final_val = self._runtime_values.get(final_out_id)
        if final_val is None:
            raise RuntimeError("Cannot run backward: forward must be executed before backward.")

        grad_map[final_out_id] = neolib.ones_like(final_val)

        for op in reversed(self.ops):
            out_grad = grad_map.pop(op.out_id, None)
            if out_grad is None:
                continue
            try:
                parent_grads = op.bwd_callable(grad=out_grad)
            except TypeError:
                parent_grads = op.bwd_callable(out_grad)

            if parent_grads is None:
                continue
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)

            if len(parent_grads) < len(op.parent_ids):
                parent_grads = parent_grads + (None,) * (len(op.parent_ids) - len(parent_grads))

            for pid, g in zip(op.parent_ids, parent_grads):
                if g is None:
                    continue
                if pid in grad_map:
                    grad_map[pid] = grad_map[pid].add_(g.clone() if safe else g)
                else:
                    grad_map[pid] = g.clone() if safe else g

        grad_leaves = []
        for ph in self.placeholders_list:
            g = grad_map.get(ph.id)
            grad_leaves.append(LiteTensor(g) if g is not None else None)

        return tree_unflatten(grad_leaves, self.flat_input_spec)


class StaticGraphBuilder:
    def build(self, fn: Callable, input_lite: Any) -> StaticGraph:
        flat_inputs, flat_spec = tree_flatten(input_lite)
        if not all(isinstance(x, LiteTensor) for x in flat_inputs):
            raise TypeError("StaticGraphBuilder.build expects all leaves in input to be LiteTensor")

        placeholders = [StaticPlaceholder(lt) for lt in flat_inputs]

        TapeContext.push(Tape())
        try:
            if isinstance(input_lite, (list, tuple)):
                out = fn(*input_lite)
            elif isinstance(input_lite, dict):
                out = fn(input_lite)
            else:
                out = fn(input_lite)
        finally:
            tape = TapeContext.pop()

        ops = []
        last_op = None
        for node in tape:
            s = StaticOp(node)
            ops.append(s)
            last_op = s

        if last_op is None:
            raise RuntimeError("empty tape recorded; nothing to build")

        return StaticGraph(ops, placeholders, last_op, flat_spec, flat_inputs)
