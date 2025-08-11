from typing import Any, List, Dict, Optional, Tuple, Union
import itertools
import torch

from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo._torch.lite_tensor import LiteTensor

from neo._src.arithmetic_ import *
from neo._src.log_ import *
from neo._src.unary_ import *
from neo._src.reduct_ import *


class Node:
    _ids = itertools.count()

    def __init__(self, op: Optional[Policy], inputs: List["Node"], name: Optional[str] = None):
        self.id = next(Node._ids)
        # `op` here should be a Policy *instance* created with desired init args,
        # so we can later re-create a fresh instance of the same class with the
        # same init args if needed.
        self.op = op
        self.inputs = inputs
        self.name = name or f"node_{self.id}"
        self.output_cache = None   # forward-pass result (LiteTensor or list)
        self.grad_cache = None     # backward-pass gradient (torch.Tensor or list of torch.Tensor)
        self.shape = None          # optional shape for validation

        # helper to recreate a fresh policy instance per run:
        # store init args/kwargs on the policy instance if created with them
        # e.g. policy = policy_cls(*args, **kwargs); policy._init_args = args; policy._init_kwargs = kwargs
        # If not present, we will try policy.__class__() with no args.
        self._policy_instance_used = None

    def __repr__(self):
        return f"<Node {self.name} op={self.op.__class__.__name__ if self.op else None}>"


class Variable(Node):
    def __init__(self, shape: Optional[Tuple[int, ...]] = None, d_type: str = '', device: str = '', name: str = None): #type:ignore
        super().__init__(op=None, inputs=[], name=name or "var")
        self.shape = shape
        self.d_type = d_type
        self.device = device
        self.is_variable = True
        self.is_constant = False


class Constant(Node):
    def __init__(self, value: LiteTensor, name: str = None): #type:ignore
        super().__init__(op=None, inputs=[], name=name or "const")
        self.output_cache = value
        self.shape = tuple(value.data.shape)
        self.is_variable = False
        self.is_constant = True


class Placeholder(Node):
    def __init__(self, shape: Optional[Tuple[int, ...]] = None, d_type: str = '', device: str = '', name: str = None): #type:ignore
        super().__init__(op=None, inputs=[], name=name or "placeholder")
        self.shape = shape
        self.d_type = d_type
        self.device = device
        self.is_variable = False
        self.is_constant = False


class Symbol:
    def __init__(self, node: Node):
        self.node = node
    def _binary_op(self, other, policy_cls, *args, **kwargs):
        other_sym = other if isinstance(other, Symbol) else Symbol(Constant(LiteTensor(other)))
        policy_inst = policy_cls()              # zero-arg ctor
        policy_inst._init_args = args
        policy_inst._init_kwargs = kwargs
        new_node = Node(op=policy_inst, inputs=[self.node, other_sym.node])
        return Symbol(new_node)

    def _unary_op(self, policy_cls, *args, **kwargs):
        policy_inst = policy_cls()              # zero-arg ctor
        policy_inst._init_args = args
        policy_inst._init_kwargs = kwargs
        new_node = Node(op=policy_inst, inputs=[self.node])
        return Symbol(new_node)

    # Binary operators
    def __add__(self, other): return self._binary_op(other, addition)
    def __sub__(self, other): return self._binary_op(other, subtraction)
    def __mul__(self, other): return self._binary_op(other, multiplication)
    def __truediv__(self, other): return self._binary_op(other, division)
    def __pow__(self, other): return self._binary_op(other, power_op)
    def __matmul__(self, other): return self._binary_op(other, matmul_op)

    # Unary ops (no extra params)
    def log(self): return self._unary_op(log)
    def log10(self): return self._unary_op(log10)
    def abs(self): return self._unary_op(abs)
    def sign(self): return self._unary_op(sign)
    def exp(self): return self._unary_op(exp)
    def sqrt(self): return self._unary_op(sqrt)

    # Reductions (with optional params)
    def sum(self, dim=None, keepdim=False):
        return self._unary_op(sum_op, dim=dim, keepdim=keepdim)

    def mean(self, dim=None, keepdim=False):
        return self._unary_op(mean_op, dim=dim, keepdim=keepdim)

    def max(self, dim=None, keepdim=False):
        return self._unary_op(max_op, dim=dim, keepdim=keepdim)


def topological_sort(output_nodes: List[Node]) -> List[Node]:
    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for inp in node.inputs:
            dfs(inp)
        order.append(node)

    for out in output_nodes:
        dfs(out)

    return order


def _ensure_torch_tensor(x):
    # Accept LiteTensor, torch.Tensor, Python scalar, numpy scalar/array
    if isinstance(x, LiteTensor):
        return x.data
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        raise TypeError("Unable to convert to torch.Tensor: %r" % (x,))


def _wrap_as_litetensor(x):
    # x may be torch.Tensor, LiteTensor, scalar, tuple/list of same
    if isinstance(x, LiteTensor):
        return x
    if isinstance(x, torch.Tensor):
        return LiteTensor(x)
    if isinstance(x, (tuple, list)):
        return [ _wrap_as_litetensor(e) for e in x ]
    # fallback
    return LiteTensor(torch.as_tensor(x))


def _is_sequence(obj):
    return isinstance(obj, (tuple, list))


def run_graph(outputs: List[Node],
              vars: List[Variable],
              feed_dict: Dict[Node, LiteTensor]) -> Tuple[List[LiteTensor], List[LiteTensor]]:
    """
    Execute forward + backward for given outputs.
    - outputs: list of output Nodes (can be single-element list)
    - vars: list of Variable nodes for which grads will be returned (in same order)
    - feed_dict: mapping Node -> LiteTensor for Variables & Placeholders & optional Constants override

    Returns:
      (outputs_list_of_LiteTensor_or_list, grads_for_vars_as_list_of_LiteTensor)
    """
    order = topological_sort(outputs)

    for node in order:
        # constants already have output_cache set to LiteTensor
        if isinstance(node, Constant):
            continue

        if isinstance(node, Variable):
            if node not in feed_dict:
                raise ValueError(f"No value provided for Variable {node.name}")
            val = feed_dict[node]
            if not isinstance(val, LiteTensor):
                raise TypeError(f"Feed value for {node.name} must be LiteTensor")
            node.output_cache = val
            # optional shape check
            if node.shape is not None and tuple(val.data.shape) != tuple(node.shape):
                raise ValueError(f"Shape mismatch for Variable {node.name}: expected {node.shape} got {tuple(val.data.shape)}")
            continue

        if isinstance(node, Placeholder):
            if node not in feed_dict:
                raise ValueError(f"No value provided for Placeholder {node.name}")
            val = feed_dict[node]
            if not isinstance(val, LiteTensor):
                raise TypeError(f"Feed value for placeholder {node.name} must be LiteTensor")
            node.output_cache = val
            if node.shape is not None and tuple(val.data.shape) != tuple(node.shape):
                raise ValueError(f"Shape mismatch for Placeholder {node.name}: expected {node.shape} got {tuple(val.data.shape)}")
            continue

        # Create a fresh policy instance from stored op instance's class & init args if provided.
        policy_template = node.op
        if policy_template is None:
            raise RuntimeError(f"Op node {node.name} has no policy instance")

        init_args = getattr(policy_template, "_init_args", ())
        init_kwargs = getattr(policy_template, "_init_kwargs", {})
        try:
            policy = policy_template.__class__(*init_args, **init_kwargs)
        except Exception:
            # fallback to trying no-arg constructor
            policy = policy_template.__class__()

        # Save the instance to use in backward
        node._policy_instance_used = policy

        # unwrap inputs to raw torch tensors (or sequences of tensors)
        raw_inputs = []
        for inp in node.inputs:
            val = inp.output_cache
            if _is_sequence(val):
                # If the input is a list/tuple of LiteTensors, convert each
                raw_inputs.append([ _ensure_torch_tensor(e) for e in val ])
            else:
                raw_inputs.append(_ensure_torch_tensor(val))

        # Call forward with raw torch.Tensor arguments.
        raw_out = policy.forward(*raw_inputs)

        # Normalize forward outputs to either a single LiteTensor or list of LiteTensors
        if _is_sequence(raw_out):
            wrapped = []
            for item in raw_out:
                if isinstance(item, LiteTensor):
                    wrapped.append(item)
                else:
                    wrapped.append(_wrap_as_litetensor(item))
            node.output_cache = wrapped
        else:
            node.output_cache = _wrap_as_litetensor(raw_out)

        # optional shape check if set on node
        if node.shape is not None:
            # support list outputs? assume shape is for first output
            out_data = node.output_cache[0].data if _is_sequence(node.output_cache) else node.output_cache.data
            if tuple(out_data.shape) != tuple(node.shape):
                raise ValueError(f"Node {node.name} produced shape {tuple(out_data.shape)} but expected {tuple(node.shape)}")

    # ---------------- Collect final outputs (wrap consistently) ----------------
    def _collect_output_value(node: Node):
        return node.output_cache
    output_vals = [_collect_output_value(n) for n in outputs]

    # Initialize grad_cache for outputs: match shape and type (torch.Tensor).
    # If an output is a sequence, we seed each element with ones_like.
    for out_node in outputs:
        out_val = out_node.output_cache
        if _is_sequence(out_val):
            # create same-structure grads
            grads = [ torch.ones_like(elem.data) for elem in out_val ]
            out_node.grad_cache = grads
        else:
            out_node.grad_cache = torch.ones_like(out_val.data)

    # Walk in reverse topo order
    for node in reversed(order):
        if node.op is None:
            continue
        grad_in = node.grad_cache
        if grad_in is None:
            # nothing to backprop through this node
            continue

        policy = getattr(node, "_policy_instance_used", None)
        if policy is None:
            raise RuntimeError(f"No policy instance recorded for node {node.name} (forward didn't run?)")

        # ensure grad_in is raw torch.Tensor or list of torch.Tensor when calling backward
        if _is_sequence(grad_in):
            raw_grad_in = [ _ensure_torch_tensor(g) for g in grad_in ]
        else:
            raw_grad_in = _ensure_torch_tensor(grad_in)

        # Call backward -> should return grad(s) for each input (torch.Tensor or sequence)
        raw_grads_for_inputs = policy.backward(raw_grad_in)

        # normalize to tuple
        if not isinstance(raw_grads_for_inputs, tuple):
            raw_grads_for_inputs = (raw_grads_for_inputs,)

        # accumulate gradients into inputs (store as raw torch.Tensor or lists)
        for inp_node, raw_g in zip(node.inputs, raw_grads_for_inputs):
            if raw_g is None:
                continue
            # If raw_g is LiteTensor, extract .data
            if isinstance(raw_g, LiteTensor):
                raw_g = raw_g.data
            if _is_sequence(raw_g):
                # store lists as lists of torch.Tensors
                to_add = [ _ensure_torch_tensor(x) for x in raw_g ]
            else:
                to_add = _ensure_torch_tensor(raw_g)

            # accumulate
            if inp_node.grad_cache is None:
                inp_node.grad_cache = to_add
            else:
                # both caches must be same structure; add elementwise or tensor add
                if _is_sequence(inp_node.grad_cache) and _is_sequence(to_add):
                    # element-wise add of lists
                    inp_node.grad_cache = [ a + b for a, b in zip(inp_node.grad_cache, to_add) ]
                elif _is_sequence(inp_node.grad_cache) != _is_sequence(to_add):
                    raise RuntimeError("Gradient structure mismatch during accumulation")
                else:
                    inp_node.grad_cache = inp_node.grad_cache + to_add


    var_grads: List[LiteTensor] = []
    for v in vars:
        raw_g = v.grad_cache
        if raw_g is None:
            # return zeros of same shape as variable if available, else None
            if v.output_cache is not None:
                zero = torch.zeros_like(v.output_cache.data)
                var_grads.append(LiteTensor(zero))
            else:
                var_grads.append(None)
            continue

        # raw_g should be torch.Tensor (or list) -> wrap into LiteTensor
        if _is_sequence(raw_g):
            # return first element if variable expected scalar? but variables are leaf nodes — sequence unlikely
            # we'll wrap sequence as list of LiteTensors
            var_grads.append([ LiteTensor(x) for x in raw_g ])
        else:
            if isinstance(raw_g, LiteTensor):
                var_grads.append(raw_g)
            else:
                var_grads.append(LiteTensor(raw_g))

    # ---------------- Cleanup: free policy instances to release ctx references ----------------
    for node in order:
        if hasattr(node, "_policy_instance_used"):
            node._policy_instance_used = None

    return output_vals, var_grads
