from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from neo._src.autograd import Tape, TapeContext
from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
import torch

# Helper to flatten/unflatten nested structures (list/dict/tuple)
def tree_flatten(x):
    """
    Flattens nested structure into a flat list of leaves and a spec.
    Spec is a nested container with same structure but leaves replaced by None.
    Supports lists, tuples, dicts, and combinations.
    """
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
    """
    Reconstruct nested structure from flat list and spec.
    """
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
    Stores:
      - out_id: id(node.output)
      - args_template: original positional args (with constants preserved)
      - kwargs_template: original keyword args (with constants preserved)
      - tensor_arg_ids: {pos_idx: parent_id} for tensor positional args
      - tensor_kwarg_ids: {kw_name: parent_id} for tensor keyword args
      - bwd_callable, fwd_callable, node_repr
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

        # parent_ids: ordered list of ids for all parents
        self.parent_ids = tuple(id(p) for p in (node.parents or ()))

        # Capture templates exactly as originally recorded
        self.args_template = list(getattr(node, "args", []))
        self.kwargs_template = dict(getattr(node, "kwargs", {}))

        # Identify which args/kwargs are tensors that map to parents
        self.tensor_arg_ids = {}
        self.tensor_kwarg_ids = {}

        # Map positional tensor args
        for i, a in enumerate(self.args_template):
            if isinstance(a, LiteTensor):
                self.tensor_arg_ids[i] = id(a)

        # Map keyword tensor args
        for k, v in self.kwargs_template.items():
            if isinstance(v, LiteTensor):
                self.tensor_kwarg_ids[k] = id(v)

        # Store forward/backward callables
        self.bwd_callable = node.bwd_fn
        owner = getattr(node.bwd_fn, "__self__", None)
        self.fwd_callable = getattr(owner, "forward", None) if owner else None



class StaticGraph:
    """
    StaticGraph stores:
      - ops: list of StaticOp (topological order)
      - placeholders_list: list of StaticPlaceholder for every input leaf (flattened order)
      - output_op: last op (used to find final out id)
      - flat_input_spec: spec to reconstruct nested inputs from leaves
      - var_leaf_indices: Optional[list[int]] indices into placeholders_list that are variables
          - If None -> old behaviour: backward() returns full nested grad structure
          - If not None -> backward() returns flat list of grads corresponding to these indices
    """

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
        self.flat_input_keys = flat_input_keys  # flattened input leaves (LiteTensor objects)
        self.var_leaf_indices = var_leaf_indices  # indices into placeholders_list for variables (or None)

    def forward(self, inputs: Any) -> LiteTensor:
        """
        Run forward with nested inputs matching the building-time structure.
        Returns LiteTensor result.
        """
        self._runtime_values.clear()

        flat_inputs, _input_spec = tree_flatten(inputs)

        if len(flat_inputs) != len(self.flat_input_keys):
            raise ValueError(
                f"Input leaf count mismatch. Expected {len(self.flat_input_keys)}, got {len(flat_inputs)}"
            )

        for i, leaf in enumerate(flat_inputs):
            if not isinstance(leaf, LiteTensor):
                raise TypeError(f"Input leaf {i} is not a LiteTensor, got {type(leaf)}")

        # Bind placeholder id -> torch.Tensor
        for ph, leaf in zip(self.placeholders_list, flat_inputs):
            self._runtime_values[ph.id] = leaf.data

        # Execute ops in recorded order
        for idx, op in enumerate(self.ops):
            # Start from templates so constants remain unchanged
            args = list(op.args_template)  # shallow copy
            kwargs = dict(op.kwargs_template)

            # Replace positional tensor arguments with runtime tensors
            for pos, pid in op.tensor_arg_ids.items():
                # pid is an id of the LiteTensor recorded earlier
                if pid not in self._runtime_values:
                    raise RuntimeError(
                        f"During static forward, missing runtime value for parent id {pid} "
                        f"needed by op[{idx}] (node={op.node_repr})."
                    )
                runtime_val = self._runtime_values[pid]
                # Ensure args list is large enough
                if pos >= len(args):
                    # extend with None until pos
                    args.extend([None] * (pos + 1 - len(args)))
                args[pos] = runtime_val

            # Replace kwarg tensor arguments
            for k, pid in op.tensor_kwarg_ids.items():
                if pid not in self._runtime_values:
                    raise RuntimeError(
                        f"During static forward, missing runtime value for parent id {pid} "
                        f"needed by op[{idx}] (node={op.node_repr}) for kwarg '{k}'."
                    )
                kwargs[k] = self._runtime_values[pid]

            # fwd callable must exist
            if op.fwd_callable is None:
                raise RuntimeError(f"No forward callable available for op (node={op.node_repr}).")

            # Call forward with preserved constants and runtime tensors in-place
            out = op.fwd_callable(*args, **kwargs)

            # Accept LiteTensor or torch.Tensor
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
        """
        Compute backward pass.

        - If var_leaf_indices is None: return nested structure matching inputs (old behaviour).
        - If var_leaf_indices is a list of indices: return a flat list of grads for the variables
          in the same order the user provided the `variables` leaves during build.
        """
        final_out_id = self.output_op.out_id
        grad_map: Dict[int, torch.Tensor] = {}
        final_val = self._runtime_values.get(final_out_id)
        if final_val is None:
            raise RuntimeError("Cannot run backward: forward must be executed before backward.")

        grad_map[final_out_id] = neolib.ones_like(final_val)

        # Reverse-mode accumulation
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

            # pad
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

        # If user asked for variables-only mode, return flat list of grads for those indices
        if self.var_leaf_indices is not None:
            var_grads: List[Optional[LiteTensor]] = [grad_leaves[i] for i in self.var_leaf_indices]
            return var_grads

        # Old behaviour: return nested structure matching inputs
        return tree_unflatten(grad_leaves, self.flat_input_spec)


class StaticGraphBuilder:
    """
    build(fn, input_lite=None, *, constants=None, variables=None)

    - Backwards-compatible: if input_lite is provided (and constants/variables are None),
      old behaviour is used: all leaves are treated as variables and backward() returns nested grads.

    - New API: pass constants (tuple/list or single) and variables (nested structure).
      The function will be invoked as fn(*constants, variables). The builder will track all input leaves
      but will mark only variable leaves as the user-requested trainable leaves.
      backward() will return a flat list of grads for variables (matching the flattened order).
    """
    def build(
        self,
        fn: Callable,
        input_lite: Any = None,
        *,
        constants: Any = None,
        variables: Any = None,
    ) -> StaticGraph:
        # Determine which API branch to use
        if constants is None and variables is None:
            # old API path: user passed `input_lite` as single nested structure or sequence
            if input_lite is None:
                raise TypeError("Either input_lite or (constants, variables) must be provided")
            combined_input = input_lite
            # var indices None -> old behaviour
            var_leaf_indices: Optional[List[int]] = None
        else:
            # new API path: build combined input structure where fn will be called as fn(*constants, variables)
            if variables is None:
                raise TypeError("If you pass constants, you must pass variables (the 'trainable' structure)")
            # Normalize constants to tuple for unpacking
            if constants is None:
                constants_tuple: Tuple = tuple()
            elif isinstance(constants, tuple):
                constants_tuple = constants
            elif isinstance(constants, list):
                constants_tuple = tuple(constants)
            else:
                # single constant value (e.g. x) -> wrap
                constants_tuple = (constants,)

            # The combined input is a tuple: (*constants_tuple, variables)
            # This matches the typical function signature like fn(x, y, params)
            combined_input = (*constants_tuple, variables)

            # We will compute var_leaf_indices below
            var_leaf_indices = []

        # Flatten combined input, get spec and list of leaves
        flat_inputs, flat_spec = tree_flatten(combined_input)

        # Validate leaves are LiteTensors
        if not all(isinstance(x, LiteTensor) for x in flat_inputs):
            raise TypeError("StaticGraphBuilder.build expects all leaves in input to be LiteTensor")

        # Make placeholders for every leaf (full order)
        placeholders = [StaticPlaceholder(lt) for lt in flat_inputs]

        # If using new API, compute which placeholder indices correspond to variable leaves
        if var_leaf_indices is not None:
            # flatten variables separately and locate their identities in flat_inputs
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

        # Create and push a fresh tape
        tape = Tape()
        TapeContext.push(tape)
        try:
            # Call function with appropriate unpacking:
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

        # Build StaticOp list from recorded nodes in our tape
        ops: List[StaticOp] = []
        last_op: Optional[StaticOp] = None
        for node in tape:
            s = StaticOp(node)
            ops.append(s)
            last_op = s

        if last_op is None:
            raise RuntimeError("empty tape recorded; nothing to build")

        return StaticGraph(ops, placeholders, last_op, flat_spec, flat_inputs, var_leaf_indices)
