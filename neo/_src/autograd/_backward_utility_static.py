from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from neo._src.autograd import Tape, TapeContext
from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
import torch

# Helper to flatten/unflatten nested structures (list/dict/tuple)
def tree_flatten(x):
    flat: List[Any] = []

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
    """
    StaticOp stores:
      - out_id: id(node.output)
      - args_template / kwargs_template: exactly as recorded (constants preserved)
      - tensor_arg_ids: mapping pos -> parent_id (only for tensor slots)
      - tensor_kwarg_ids: mapping key -> parent_id (only for tensor slots)
      - parent_ids: tuple of parent ids (order from node.parents)
      - fwd_callable / bwd_callable
    """

    __slots__ = (
        "out_id",
        "parent_ids",
        "args_template",
        "kwargs_template",
        "tensor_arg_ids",
        "tensor_kwarg_ids",
        "fwd_callable",
        "bwd_callable",
        "node_repr",
    )

    def __init__(self, node):
        self.node_repr = repr(node)
        self.out_id = id(node.output)

        # parents and their ids
        parents = list(node.parents or ())
        self.parent_ids = tuple(id(p) for p in parents)

        # capture exact arg/kw templates if available, else defaults
        node_args = getattr(node, "args", ())
        node_kwargs = getattr(node, "kwargs", {})

        self.args_template: List[Any] = list(node_args)
        self.kwargs_template: Dict[str, Any] = dict(node_kwargs)

        # maps (only for tensor positions)
        self.tensor_arg_ids: Dict[int, int] = {}
        self.tensor_kwarg_ids: Dict[str, int] = {}

        # mapping algorithm 
        # For each parent (in tape order), try:
        #  1 identity match with args_template entries (a is p)
        #  2 identity match with kwargs_template values (v is p)
        #  3 fill an explicit None placeholder in args_template
        #  4 append a new positional None slot and map to it
        for p in parents:
            pid = id(p)
            placed = False

            # 1) identity match in positional args
            for i, a in enumerate(self.args_template):
                if a is p:
                    # exact identity: map this position to this parent id
                    self.tensor_arg_ids[i] = pid
                    placed = True
                    break
            if placed:
                continue

            # 2) identity match in kwargs
            for k, v in self.kwargs_template.items():
                if v is p:
                    self.tensor_kwarg_ids[k] = pid
                    placed = True
                    break
            if placed:
                continue

            # 3) fill an explicit None placeholder slot in args_template (prefer these)
            for i, a in enumerate(self.args_template):
                if a is None and i not in self.tensor_arg_ids:
                    self.tensor_arg_ids[i] = pid
                    placed = True
                    break
            if placed:
                continue

            # 4) last resort: append a new positional slot
            idx = len(self.args_template)
            self.args_template.append(None)
            self.tensor_arg_ids[idx] = pid
            # placed = True  # implied

        # store callables
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
        var_leaf_indices: Optional[List[int]] = None,
    ):
        self.ops = ops
        self.placeholders_list = input_placeholders
        self.placeholders_by_id = {ph.id: ph for ph in input_placeholders}
        self.output_op = output_op
        self._runtime_values: Dict[int, torch.Tensor] = {}

        self.flat_input_spec = flat_input_spec
        self.flat_input_keys = flat_input_keys
        self.var_leaf_indices = var_leaf_indices

    
    def forward(self, inputs: Any) -> LiteTensor:
        self._runtime_values.clear()

        flat_inputs, _input_spec = tree_flatten(inputs)

        if len(flat_inputs) != len(self.flat_input_keys):
            raise ValueError(
                f"Input leaf count mismatch. Expected {len(self.flat_input_keys)}, got {len(flat_inputs)}"
            )

        for i, leaf in enumerate(flat_inputs):
            if not isinstance(leaf, LiteTensor):
                raise TypeError(f"Input leaf {i} is not a LiteTensor, got {type(leaf)}")

        # Bind placeholders -> runtime torch.Tensor
        for ph, leaf in zip(self.placeholders_list, flat_inputs):
            self._runtime_values[ph.id] = leaf.data

        # Execute ops in recorded order
        for idx, op in enumerate(self.ops):
            # Start from templates (constants preserved)
            args = list(op.args_template)
            kwargs = dict(op.kwargs_template)

            # Replace mapped tensor positional args
            for pos, pid in op.tensor_arg_ids.items():
                if pid not in self._runtime_values:
                    raise RuntimeError(
                        f"[Op {idx}] Missing runtime value for parent id {pid} "
                        f"needed by op (node={op.node_repr})."
                    )
                runtime_val = self._runtime_values[pid]
                if runtime_val is None:
                    raise RuntimeError(
                        f"[Op {idx}] Runtime value for parent id {pid} is None "
                        f"in op (node={op.node_repr})."
                    )
                args[pos] = runtime_val

            # Replace mapped tensor kwargs
            for k, pid in op.tensor_kwarg_ids.items():
                if pid not in self._runtime_values:
                    raise RuntimeError(
                        f"[Op {idx}] Missing runtime value for parent id {pid} "
                        f"needed by op (node={op.node_repr}) for kwarg '{k}'."
                    )
                runtime_val = self._runtime_values[pid]
                if runtime_val is None:
                    raise RuntimeError(
                        f"[Op {idx}] Runtime value for parent id {pid} is None "
                        f"in op (node={op.node_repr}) for kwarg '{k}'."
                    )
                kwargs[k] = runtime_val

            # Debug print: show args and kwargs types before call
            print(f"[Op {idx}] Calling {op.fwd_callable} with args types: {[type(a) for a in args]} and kwargs keys: {list(kwargs.keys())}")

            if op.fwd_callable is None:
                raise RuntimeError(f"No forward callable available for op (node={op.node_repr}).")

            out = op.fwd_callable(*args, **kwargs)

            if isinstance(out, LiteTensor):
                out_tensor = out.data
            elif isinstance(out, torch.Tensor):
                out_tensor = out
            else:
                raise TypeError(
                    f"Forward callable returned unsupported type {type(out)} for node {op.node_repr}"
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

            # pad to match number of recorded parents
            if len(parent_grads) < len(op.parent_ids):
                parent_grads = parent_grads + (None,) * (len(op.parent_ids) - len(parent_grads))

            for pid, g in zip(op.parent_ids, parent_grads):
                if g is None:
                    continue
                if pid in grad_map:
                    grad_map[pid] = grad_map[pid].add_(g.clone() if safe else g)
                else:
                    grad_map[pid] = g.clone() if safe else g

        # Build grad leaves in full input order
        grad_leaves: List[Optional[LiteTensor]] = []
        for ph in self.placeholders_list:
            g = grad_map.get(ph.id)
            grad_leaves.append(LiteTensor(g) if g is not None else None)

        if self.var_leaf_indices is not None:
            return [grad_leaves[i] for i in self.var_leaf_indices]

        return tree_unflatten(grad_leaves, self.flat_input_spec)


class StaticGraphBuilder:
    def build(
        self,
        fn: Callable,
        input_lite: Any = None,
        *,
        constants: Any = None,
        variables: Any = None,
    ) -> StaticGraph:
        # choose API mode
        if constants is None and variables is None:
            if input_lite is None:
                raise TypeError("Either input_lite or (constants, variables) must be provided")
            combined_input = input_lite
            var_leaf_indices: Optional[List[int]] = None
        else:
            if variables is None:
                raise TypeError("If you pass constants, you must pass variables")
            if constants is None:
                constants_tuple: Tuple = tuple()
            elif isinstance(constants, (list, tuple)):
                constants_tuple = tuple(constants)
            else:
                constants_tuple = (constants,)
            combined_input = (*constants_tuple, variables)
            var_leaf_indices = []

        flat_inputs, flat_spec = tree_flatten(combined_input)

        if not all(isinstance(x, LiteTensor) for x in flat_inputs):
            raise TypeError("StaticGraphBuilder.build expects all leaves in input to be LiteTensor")

        placeholders = [StaticPlaceholder(lt) for lt in flat_inputs]

        # compute variable leaf indices if needed
        if var_leaf_indices is not None:
            var_leaves, _ = tree_flatten(variables)
            remaining = list(var_leaves)
            indices: List[int] = []
            for idx, leaf in enumerate(flat_inputs):
                if remaining and leaf is remaining[0]:
                    indices.append(idx)
                    remaining.pop(0)
            if remaining:
                raise RuntimeError("Failed to identify variable leaves inside combined input")
            var_leaf_indices = indices
        else:
            var_leaf_indices = None

        tape = Tape()
        TapeContext.push(tape)
        try:
            if isinstance(combined_input, (list, tuple)):
                out = fn(*combined_input)
            elif isinstance(combined_input, dict):
                out = fn(combined_input)
            else:
                out = fn(combined_input)
        finally:
            popped = TapeContext.pop()
            if popped is None:
                import warnings
                warnings.warn("TapeContext.pop() returned None; using local Tape instance.")

        ops: List[StaticOp] = []
        last_op: Optional[StaticOp] = None
        for node in tape:
            s = StaticOp(node)
            ops.append(s)
            last_op = s

        if last_op is None:
            raise RuntimeError("empty tape recorded; nothing to build")

        return StaticGraph(ops, placeholders, last_op, flat_spec, flat_inputs, var_leaf_indices)
