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
        self.op = op
        self.inputs = inputs
        self.name = name or f"node_{self.id}"
        self.output_cache = None   # LiteTensor, list, or dict of LiteTensors
        self.grad_cache = None     # torch.Tensor, list, or dict of torch.Tensor
        self.shape = None

        self._policy_instance_used = None

    def __repr__(self):
        return f"<Node {self.name} op={self.op.__class__.__name__ if self.op else None}>"


class Variable(Node):
    def __init__(self, shape: Optional[Tuple[int, ...]] = None, d_type: str = '', device: str = '',
                 name: str = None, is_dict: bool = False):  # added is_dict flag
        super().__init__(op=None, inputs=[], name=name or "var")
        self.shape = shape
        self.d_type = d_type
        self.device = device
        self.is_variable = True
        self.is_constant = False
        self.is_dict = is_dict  # True if this variable holds dict of LiteTensors


class Constant(Node):
    def __init__(self, value: Union[LiteTensor, Dict[str, LiteTensor]], name: str = None):
        super().__init__(op=None, inputs=[], name=name or "const")
        self.output_cache = value
        if isinstance(value, LiteTensor):
            self.shape = tuple(value.data.shape)
        else:
            # dict case: no single shape
            self.shape = None
        self.is_variable = False
        self.is_constant = True


class Placeholder(Node):
    def __init__(self, shape: Optional[Tuple[int, ...]] = None, d_type: str = '', device: str = '', name: str = None):
        super().__init__(op=None, inputs=[], name=name or "placeholder")
        self.shape = shape
        self.d_type = d_type
        self.device = device
        self.is_variable = False
        self.is_constant = False


class Symbol:
    def __init__(self, node: Node, key: Optional[Union[str, Tuple[str, ...]]] = None):
        self.node = node
        self.key = key  # None or string key or tuple of keys for nested dicts

    def _binary_op(self, other, policy_cls, *args, **kwargs):
        other_sym = other if isinstance(other, Symbol) else Symbol(Constant(LiteTensor(other)))
        policy_inst = policy_cls()  # zero-arg ctor
        policy_inst._init_args = args
        policy_inst._init_kwargs = kwargs
        new_node = Node(op=policy_inst, inputs=[self.node, other_sym.node])
        return Symbol(new_node)

    def __getitem__(self, idx):
        # Compose keys for nested dict indexing
        if self.key is None:
            new_key = idx
        else:
            if isinstance(self.key, tuple):
                new_key = self.key + (idx,)
            else:
                new_key = (self.key, idx)
        return Symbol(self.node, key=new_key)

    def _unary_op(self, policy_cls, *args, **kwargs):
        policy_inst = policy_cls()  # zero-arg ctor
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

    # Unary ops
    def log(self): return self._unary_op(log)
    def log10(self): return self._unary_op(log10)
    def abs(self): return self._unary_op(abs)
    def sign(self): return self._unary_op(sign)
    def exp(self): return self._unary_op(exp)
    def sqrt(self): return self._unary_op(sqrt)

    # Reductions
    def sum(self, dim=None, keepdim=False):
        return self._unary_op(sum_op, dim=dim, keepdim=keepdim)

    def mean(self, dim=None, keepdim=False):
        return self._unary_op(mean_op, dim=dim, keepdim=keepdim)

    def max(self, dim=None, keepdim=False):
        return self._unary_op(max_op, dim=dim, keepdim=keepdim)

    def get_value(self):
        val = self.node.output_cache
        if self.key is None:
            return val
        keys = self.key if isinstance(self.key, tuple) else (self.key,)
        for k in keys:
            val = val[k]
        return val


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
    if isinstance(x, LiteTensor):
        return x.data
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        raise TypeError(f"Unable to convert to torch.Tensor: {x!r}")


def _wrap_as_litetensor(x):
    if isinstance(x, LiteTensor):
        return x
    if isinstance(x, torch.Tensor):
        return LiteTensor(x)
    if isinstance(x, (tuple, list)):
        return [_wrap_as_litetensor(e) for e in x]
    if isinstance(x, dict):
        return {k: _wrap_as_litetensor(v) for k, v in x.items()}
    return LiteTensor(torch.as_tensor(x))


def _is_sequence(obj):
    return isinstance(obj, (tuple, list))


def run_graph(outputs: List[Node],
              vars: List[Variable],
              feed_dict: Dict[Node, Union[LiteTensor, Dict[str, LiteTensor]]]
              ) -> Tuple[List[Union[LiteTensor, List[LiteTensor], Dict[str, LiteTensor]]],
                         List[Union[LiteTensor, List[LiteTensor], Dict[str, LiteTensor]]]]:
    order = topological_sort(outputs)

    for node in order:
        if isinstance(node, Constant):
            continue

        if isinstance(node, Variable):
            if node not in feed_dict:
                raise ValueError(f"No value provided for Variable {node.name}")
            val = feed_dict[node]
            # Allow dict-valued variable or single LiteTensor
            if node.is_dict:
                if not isinstance(val, dict):
                    raise TypeError(f"Feed value for dict Variable {node.name} must be dict of LiteTensor")
                for k, v in val.items():
                    if not isinstance(v, LiteTensor):
                        raise TypeError(f"Dict feed for {node.name}[{k}] must be LiteTensor")
                node.output_cache = val
            else:
                if not isinstance(val, LiteTensor):
                    raise TypeError(f"Feed value for Variable {node.name} must be LiteTensor")
                node.output_cache = val
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

        # Create fresh policy instance
        policy_template = node.op
        if policy_template is None:
            raise RuntimeError(f"Op node {node.name} has no policy instance")

        init_args = getattr(policy_template, "_init_args", ())
        init_kwargs = getattr(policy_template, "_init_kwargs", {})
        try:
            policy = policy_template.__class__(*init_args, **init_kwargs)
        except Exception:
            policy = policy_template.__class__()

        node._policy_instance_used = policy

        # unwrap inputs (handle dict-valued inputs)
        raw_inputs = []
        for inp in node.inputs:
            val = inp.output_cache
            if isinstance(val, dict):
                # unwrap dict values to torch.Tensor
                raw_inputs.append({k: _ensure_torch_tensor(v) for k, v in val.items()})
            elif _is_sequence(val):
                raw_inputs.append([_ensure_torch_tensor(e) for e in val])
            else:
                raw_inputs.append(_ensure_torch_tensor(val))

        raw_out = policy.forward(*raw_inputs)

        # Normalize outputs (handle dict outputs)
        if isinstance(raw_out, dict):
            node.output_cache = {k: _wrap_as_litetensor(v) for k, v in raw_out.items()}
        elif _is_sequence(raw_out):
            node.output_cache = [_wrap_as_litetensor(v) for v in raw_out]
        else:
            node.output_cache = _wrap_as_litetensor(raw_out)

        # shape check (for first output in list or dict)
        if node.shape is not None:
            if isinstance(node.output_cache, dict):
                first_val = next(iter(node.output_cache.values()))
                out_data = first_val.data
            elif _is_sequence(node.output_cache):
                out_data = node.output_cache[0].data
            else:
                out_data = node.output_cache.data

            if tuple(out_data.shape) != tuple(node.shape):
                raise ValueError(f"Node {node.name} produced shape {tuple(out_data.shape)} but expected {tuple(node.shape)}")

    # Collect final outputs
    output_vals = [n.output_cache for n in outputs]

    # Initialize grad_cache for outputs
    for out_node in outputs:
        out_val = out_node.output_cache
        if isinstance(out_val, dict):
            grads = {k: torch.ones_like(v.data) for k, v in out_val.items()}
            out_node.grad_cache = grads
        elif _is_sequence(out_val):
            grads = [torch.ones_like(elem.data) for elem in out_val]
            out_node.grad_cache = grads
        else:
            out_node.grad_cache = torch.ones_like(out_val.data)

    # Backward pass
    for node in reversed(order):
        if node.op is None:
            continue
        grad_in = node.grad_cache
        if grad_in is None:
            continue

        policy = getattr(node, "_policy_instance_used", None)
        if policy is None:
            raise RuntimeError(f"No policy instance recorded for node {node.name} (forward didn't run?)")

        # Prepare raw grad input
        if isinstance(grad_in, dict):
            raw_grad_in = {k: _ensure_torch_tensor(v) for k, v in grad_in.items()}
        elif _is_sequence(grad_in):
            raw_grad_in = [_ensure_torch_tensor(g) for g in grad_in]
        else:
            raw_grad_in = _ensure_torch_tensor(grad_in)

        raw_grads_for_inputs = policy.backward(raw_grad_in)
        if not isinstance(raw_grads_for_inputs, tuple):
            raw_grads_for_inputs = (raw_grads_for_inputs,)

        for inp_node, raw_g in zip(node.inputs, raw_grads_for_inputs):
            if raw_g is None:
                continue
            if isinstance(raw_g, LiteTensor):
                raw_g = raw_g.data

            if isinstance(raw_g, dict):
                to_add = {k: _ensure_torch_tensor(v) for k, v in raw_g.items()}
            elif _is_sequence(raw_g):
                to_add = [_ensure_torch_tensor(x) for x in raw_g]
            else:
                to_add = _ensure_torch_tensor(raw_g)

            if inp_node.grad_cache is None:
                inp_node.grad_cache = to_add
            else:
                # Accumulate gradients properly (handle dict and sequences)
                if isinstance(inp_node.grad_cache, dict) and isinstance(to_add, dict):
                    for k in to_add:
                        inp_node.grad_cache[k] = inp_node.grad_cache.get(k, 0) + to_add[k]
                elif _is_sequence(inp_node.grad_cache) and _is_sequence(to_add):
                    inp_node.grad_cache = [a + b for a, b in zip(inp_node.grad_cache, to_add)]
                elif type(inp_node.grad_cache) != type(to_add):
                    raise RuntimeError("Gradient structure mismatch during accumulation")
                else:
                    inp_node.grad_cache = inp_node.grad_cache + to_add

    # Gather gradients for variables
    var_grads: List[Union[LiteTensor, List[LiteTensor], Dict[str, LiteTensor]]] = []
    for v in vars:
        raw_g = v.grad_cache
        if raw_g is None:
            if v.output_cache is not None:
                if isinstance(v.output_cache, dict):
                    zero_dict = {k: torch.zeros_like(t.data) for k, t in v.output_cache.items()}
                    var_grads.append({k: LiteTensor(z) for k, z in zero_dict.items()})
                else:
                    zero = torch.zeros_like(v.output_cache.data)
                    var_grads.append(LiteTensor(zero))
            else:
                var_grads.append(None)
            continue

        if isinstance(raw_g, dict):
            var_grads.append({k: LiteTensor(t) if not isinstance(t, LiteTensor) else t for k, t in raw_g.items()})
        elif _is_sequence(raw_g):
            var_grads.append([LiteTensor(t) if not isinstance(t, LiteTensor) else t for t in raw_g])
        else:
            if isinstance(raw_g, LiteTensor):
                var_grads.append(raw_g)
            else:
                var_grads.append(LiteTensor(raw_g))

    # Cleanup
    for node in order:
        if hasattr(node, "_policy_instance_used"):
            node._policy_instance_used = None

    return output_vals, var_grads
