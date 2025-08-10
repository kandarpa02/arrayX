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
    Robust StaticOp that preserves original args/kwargs (constants) and
    maps recorded parent LiteTensors to the exact arg/kw slot that used them.
    """

    __slots__ = (
        "out_id",
        "parent_ids",
        "args_template",
        "kwargs_template",
        "tensor_arg_ids",    # pos -> parent_id
        "tensor_kwarg_ids",  # key -> parent_id
        "fwd_callable",
        "bwd_callable",
        "node_repr",
    )

    def __init__(self, node):
        self.node_repr = repr(node)
        self.out_id = id(node.output)
        # ordered list of parent LiteTensors (may be empty)
        parents = list(node.parents or ())
        self.parent_ids = tuple(id(p) for p in parents)

        # Capture exact templates (positional and keyword), default to empty if missing
        node_args = getattr(node, "args", ())
        node_kwargs = getattr(node, "kwargs", {})

        self.args_template: List[Any] = list(node_args)
        self.kwargs_template: Dict[str, Any] = dict(node_kwargs)

        # Prepare maps: which arg positions/kw keys correspond to which parent ids
        self.tensor_arg_ids: Dict[int, int] = {}
        self.tensor_kwarg_ids: Dict[str, int] = {}

        # Map parents -> positions/keys by identity
        # For each parent, try to find it in args_template (identity match) first,
        # then in kwargs_template values; if not found, assign to next free positional slot.
        used_positions = set()
        used_keys = set()

        for p in parents:
            pid = id(p)
            placed = False
            # search positional args for identity match
            for i, a in enumerate(self.args_template):
                if a is p:
                    self.tensor_arg_ids[i] = pid
                    used_positions.add(i)
                    placed = True
                    break
            if placed:
                continue
            # search kwargs for identity match
            for k, v in self.kwargs_template.items():
                if v is p:
                    self.tensor_kwarg_ids[k] = pid
                    used_keys.add(k)
                    placed = True
                    break
            if placed:
                continue
            # fallback: find next unused positional index (extend template if needed)
            # prefer filling `None` slots first
            for i, a in enumerate(self.args_template):
                if i in used_positions:
                    continue
                # if slot already holds a constant not equal to parent, skip it
                if a is not None:
                    continue
                self.tensor_arg_ids[i] = pid
                used_positions.add(i)
                placed = True
                break
            if placed:
                continue
            # otherwise append a new positional slot
            idx = len(self.args_template)
            self.args_template.append(None)
            self.tensor_arg_ids[idx] = pid
            used_positions.add(idx)

        # Identify tensor kwargs by identity too (if any parents remain - already handled)
        # (Note: above loop already captured kwarg placements where identity matched)

        # save bwd/fwd callables
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

        # Execute ops in recorded order, preserving constants and replacing tensor slots
        for idx, op in enumerate(self.ops):
            args = list(op.args_template)  # copy template so we don't mutate original
            kwargs = dict(op.kwargs_template)

            # Replace positional tensor args from runtime map
            for pos, pid in op.tensor_arg_ids.items():
                if pid not in self._runtime_values:
                    raise RuntimeError(
                        f"During static forward, missing runtime value for parent id {pid} "
                        f"needed by op[{idx}] (node={op.node_repr})."
                    )
                runtime_val = self._runtime_values[pid]
                if pos >= len(args):
                    args.extend([None] * (pos + 1 - len(args)))
                args[pos] = runtime_val

            # Replace keyword tensor args
            for k, pid in op.tensor_kwarg_ids.items():
                if pid not in self._runtime_values:
                    raise RuntimeError(
                        f"During static forward, missing runtime value for parent id {pid} "
                        f"needed by op[{idx}] (node={op.node_repr}) for kwarg '{k}'."
                    )
                kwargs[k] = self._runtime_values[pid]

            if op.fwd_callable is None:
                raise RuntimeError(f"No forward callable available for op (node={op.node_repr}).")

            out = op.fwd_callable(*args, **kwargs)

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

            # pad parents to match parent_ids length
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
