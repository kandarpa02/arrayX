from typing import Dict, Tuple, Callable, Any
from contextlib import contextmanager
from neo._torch.lite_tensor import LiteTensor
from neo._torch.random import RNGKey

ParamType = Dict[str, LiteTensor]


class Module:
    _current_params: ParamType | None = None  
    _name_stack: list[str] = []  

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @contextmanager
    def param_context(self, params: ParamType):
        old_params, old_stack = Module._current_params, Module._name_stack
        Module._current_params = params
        Module._name_stack = []
        try:
            yield
        finally:
            Module._current_params, Module._name_stack = old_params, old_stack

    @staticmethod
    def param(name: str, shape, dtype, device, init_fn, rng=None) -> LiteTensor:
        if Module._current_params is None:
            raise RuntimeError("No param context active.")
        # Build hierarchical name
        prefix = "/".join(Module._name_stack) if Module._name_stack else ""
        full_name = f"{prefix}/{name}" if prefix else name
        if full_name not in Module._current_params:
            Module._current_params[full_name] = (
                init_fn(shape, key=rng, dtype=dtype, device=device)
                if rng is not None else
                init_fn(shape, dtype=dtype, device=device)
            )
        return Module._current_params[full_name]

    def __call__(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        raise NotImplementedError

    def init(self, x: LiteTensor, rng: RNGKey) -> ParamType:
        params: ParamType = {}
        with self.param_context(params):
            _ = self(x, rng)
        return params

    def apply(self, params: ParamType, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        with self.param_context(params):
            return self(x, rng)

class Layer(Module):
    def __init__(self, name: str = "", is_leaf: bool = False):
        super().__init__(name)
        self.is_leaf = is_leaf

    def __call__(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        if not self.is_leaf: 
            Module._name_stack.append(self.name)
        try:
            return self.__call__(x, rng)
        finally:
            if not self.is_leaf:
                Module._name_stack.pop()

