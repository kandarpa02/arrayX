# neo/_src/static_no_change.py
from typing import List, Dict, Any, Callable
from neo._src.autograd import Tape, TapeContext
from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
import torch


class StaticPlaceholder:
    """Wrapper around an original LiteTensor used at build-time.
    At runtime we map it to a concrete torch.Tensor (LiteTensor.data).
    """
    __slots__ = ("original_lite", "id")
    def __init__(self, lite: LiteTensor):
        self.original_lite = lite
        self.id = id(lite)


class StaticOp:
    """Holds callables captured from the recorded Node."""
    __slots__ = ("out_id", "parent_ids", "fwd_callable", "bwd_callable", "node")
    def __init__(self, node):
        self.node = node
        self.out_id = id(node.output)
        # parents are LiteTensors at record time; store ids for lookup
        self.parent_ids = tuple(id(p) for p in node.parents)
        # bwd is stored on node already (bound method)
        self.bwd_callable = node.bwd_fn
        # Try to locate forward callable from the bound bwd method's __self__.
        # If bwd_fn is a bound method, bwd_fn.__self__ is the Policy instance
        # and Policy.forward is the forward function we want to call.
        try:
            owner = getattr(node.bwd_fn, "__self__", None)
            if owner is None:
                raise AttributeError("bwd_fn has no __self__; cannot find forward")
            fwd = getattr(owner, "forward", None)
            if fwd is None:
                raise AttributeError("policy object has no forward method")
            # fwd is a bound method: owner.forward
            self.fwd_callable = fwd
        except Exception as e:
            # fail early if a node doesn't follow expected shape
            raise RuntimeError(f"Failed to extract forward callable for node {node}: {e}") from e



class StaticGraph:
    """
    Static graph created from a single tape recording. Reusable for multiple forward/backward runs.
    """
    def __init__(self, ops: List[StaticOp], input_placeholders: List[StaticPlaceholder], output_op: StaticOp):
        self.ops = ops
        # placeholder id -> Placeholder
        self.placeholders: Dict[int, StaticPlaceholder] = {ph.id: ph for ph in input_placeholders}
        self.output_op = output_op
        # runtime map: original_lite_id -> torch.Tensor (the runtime value for that recorded id)
        self._runtime_values: Dict[int, torch.Tensor] = {}

    def forward(self, *input_lite: LiteTensor) -> LiteTensor:
        """Run forward with new LiteTensor inputs (preserves LiteTensor API)."""
        # bind inputs: order must match placeholders order used at build
        if len(input_lite) != len(self.placeholders):
            raise ValueError(f"expected {len(self.placeholders)} inputs, got {len(input_lite)}")
        # clear runtime map
        self._runtime_values.clear()
        # deterministic ordering: use the same order as placeholders were created when building (insertion order)
        for ph_id, lt in zip(list(self.placeholders.keys()), input_lite):
            if not isinstance(lt, LiteTensor):
                raise TypeError("forward expects LiteTensor objects")
            self._runtime_values[ph_id] = lt.data  # store torch.Tensor

        # execute all ops in recorded order
        for op in self.ops:
            # gather parent tensors for this op from runtime map
            args = []
            for pid in op.parent_ids:
                if pid not in self._runtime_values:
                    raise RuntimeError(f"parent id {pid} not bound before use in static forward")
                args.append(self._runtime_values[pid])
            # call the captured forward callable (bound Policy.forward). It expects raw tensor args.
            out_tensor = op.fwd_callable(*args)
            # store output in runtime map keyed by recorded output id
            self._runtime_values[op.out_id] = out_tensor

        # wrap final output into LiteTensor (preserve old API)
        final = self._runtime_values[self.output_op.out_id]
        return LiteTensor(final)

    def backward(self, safe: bool = False):
        """Run reverse-mode using captured bwd callables.
        Returns gradients for inputs (ordering same as build inputs).
        """
        # Start with ones_like on final output
        final_out_id = self.output_op.out_id
        grad_map: Dict[int, torch.Tensor] = {}
        grad_map[final_out_id] = neolib.ones_like(self._runtime_values[final_out_id])

        # traverse ops in reverse
        for op in reversed(self.ops):
            out_grad = grad_map.pop(op.out_id, None)
            if out_grad is None:
                continue
            # call bwd callable. Most of your bwd methods accept grad=... keyword; try that first.
            try:
                parent_grads = op.bwd_callable(grad=out_grad)
            except TypeError:
                parent_grads = op.bwd_callable(out_grad)
            # normalize to tuple
            if parent_grads is None:
                continue
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
            # distribute to recorded parent ids
            for pid, g in zip(op.parent_ids, parent_grads):
                if g is None:
                    continue
                if pid in grad_map:
                    grad_map[pid] = grad_map[pid].add_(g.clone() if safe else g)
                else:
                    grad_map[pid] = g.clone() if safe else g

        # collect input grads in the same order as placeholders
        input_grads = []
        for ph_id in self.placeholders.keys():
            g = grad_map.get(ph_id)
            input_grads.append(LiteTensor(g) if g is not None else None)
        return input_grads if len(input_grads) != 1 else input_grads[0]



class StaticGraphBuilder:
    """Builds a StaticGraph from a single call to fn(*lite_inputs)."""
    def build(self, fn: Callable, *input_lite: LiteTensor) -> StaticGraph:
        if not all(isinstance(x, LiteTensor) for x in input_lite):
            raise TypeError("StaticGraphBuilder.build expects LiteTensor inputs")

        # build placeholders from the given input LiteTensors (preserve order)
        placeholders = [StaticPlaceholder(lt) for lt in input_lite]

        # record tape during one forward pass
        tape = Tape()
        TapeContext.push(tape)
        out = fn(*input_lite)  # this run will create Nodes on the Tape
        TapeContext.pop()

        # convert recorded nodes into StaticOps
        ops: List[StaticOp] = []
        last_op = None
        for node in tape:   # tape supports iteration via __getitem__ / __len__
            s = StaticOp(node)   # captures fwd and bwd callables from node
            ops.append(s)
            last_op = s

        if last_op is None:
            raise RuntimeError("empty tape recorded; nothing to build")

        return StaticGraph(ops, placeholders, last_op)