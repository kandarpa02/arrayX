from typing import Any, NamedTuple
from neo._src.autograd.define_then_run import Symbol, Variable, Constant, Node
from neo._torch.lite_tensor import LiteTensor
import torch

from typing import Any, Tuple, List, Dict, Union

def make_dict_vars(params: Any, prefix: str = "") -> Tuple[Any, List[Variable]]:
    """
    Recursively convert a nested dict/list/tuple of values into
    the same nested structure but with Variable nodes.
    Returns:
      - The nested structure of Variables,
      - A flat list of all Variable nodes created.
    """
    vars_list = []

    if isinstance(params, dict):
        out = {}
        for k, v in params.items():
            full_name = f"{prefix}.{k}" if prefix else k
            var_subtree, vars_sublist = make_dict_vars(v, full_name)
            out[k] = var_subtree
            vars_list.extend(vars_sublist)
        return out, vars_list

    elif isinstance(params, (list, tuple)):
        out = []
        for i, v in enumerate(params):
            full_name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            var_subtree, vars_sublist = make_dict_vars(v, full_name)
            out.append(var_subtree)
            vars_list.extend(vars_sublist)
        if isinstance(params, tuple):
            out = tuple(out)
        return out, vars_list

    else:
        var = Variable(name=prefix)
        vars_list.append(var)
        return var, vars_list

def make_symbol(obj):
    if isinstance(obj, Variable):
        return Symbol(obj)
    if isinstance(obj, dict):
        return {k: make_symbol(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        res = [make_symbol(v) for v in obj]
        return tuple(res) if isinstance(obj, tuple) else res
    return obj


def build_feed_dict(vars_struct, values_struct, feed=None):
    if feed is None:
        feed = {}
    if isinstance(vars_struct, Variable):
        feed[vars_struct] = values_struct
    elif isinstance(vars_struct, dict):
        for k in vars_struct:
            build_feed_dict(vars_struct[k], values_struct[k], feed)
    elif isinstance(vars_struct, (list, tuple)):
        for v, val in zip(vars_struct, values_struct):
            build_feed_dict(v, val, feed)
    return feed

import secrets
class define_dict_map:
    def __init__(self, input:dict, prefix:str= None #type:ignore
    ): 
        self.input = input
        self.prefix = secrets.token_hex(8) if prefix is None else prefix
        self.function_input, self.vars, self.feed_dict = self.values()

    def values(self):
        variable_dict, variables_list = make_dict_vars(self.input, self.prefix)
        variable_dict_symbol = make_symbol(variable_dict)
        feed_dict = build_feed_dict(variable_dict, self.input)
        return variable_dict_symbol, variables_list, feed_dict
        

def _to_constant_symbol(x: Any) -> Symbol:
    """Wrap a raw value into a Constant Node and return its Symbol."""
    if isinstance(x, LiteTensor):
        return Symbol(Constant(x))
    if isinstance(x, torch.Tensor):
        return Symbol(Constant(LiteTensor(x)))
    # fallback: Python scalar, numpy, etc.
    try:
        return Symbol(Constant(LiteTensor(x)))
    except TypeError as e:
        raise TypeError(f"Cannot convert {type(x)} to Constant Symbol: {e}")


def wrap_symbol(x: Any) -> Any:
    """
    Recursively wrap inputs into Symbol or container-of-Symbols.
    - If x is a Symbol or Node subclass -> wrap or return Symbol.
    - If x is a dict/list/tuple -> recursively wrap items.
    - If x is a LiteTensor/torch.Tensor/scalar -> wrap as Constant Symbol.
    """
    if isinstance(x, Symbol):
        return x

    # Node subclasses include Variable, Constant
    if isinstance(x, Node):
        return Symbol(x)

    if isinstance(x, dict):
        return {k: wrap_symbol(v) for k, v in x.items()}

    if isinstance(x, list):
        return [wrap_symbol(v) for v in x]

    if isinstance(x, tuple):
        return tuple(wrap_symbol(v) for v in x)

    return _to_constant_symbol(x)

def function(fn):
    """
    Decorator that wraps args and kwargs into Symbols recursively.
    Non-graph kwargs (int, float, bool, str, None) are left untouched.
    """
    def wrapper(*args, **kwargs):
        sym_args = [wrap_symbol(a) for a in args]

        sym_kwargs = {}
        for k, v in kwargs.items():
            # Wrap containers and graph nodes; leave literals untouched
            if isinstance(v, (Variable, Constant, Symbol, dict, list, tuple, LiteTensor, torch.Tensor)):
                sym_kwargs[k] = wrap_symbol(v)
            else:
                sym_kwargs[k] = v

        return fn(*sym_args, **sym_kwargs)

    return wrapper
