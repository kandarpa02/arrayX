from typing import Dict
from contextlib import contextmanager
from nexnet._torch.lite_tensor import LiteTensor
from nexnet._torch.random import RNGKey

ParamType = Dict[str, LiteTensor]


class Module:
    _current_params: ParamType | None = None
    _name_stack: list[str] = []
    _name_counters: dict[str, int] = {}

    def __init__(self, name: str = ""):
        cls_name = self.__class__.__name__
        if not name:
            if not Module._name_stack:  # root module -> stable name
                name = cls_name
            else:  # submodule -> auto-increment
                count = Module._name_counters.get(cls_name, 0)
                name = f"{cls_name}_{count}"
                Module._name_counters[cls_name] = count + 1
        self.name = name


    @contextmanager
    def param_context(self, params: ParamType):
        old_params = Module._current_params
        Module._current_params = params
        try:
            yield
        finally:
            Module._current_params = old_params

    @contextmanager
    def name_context(self):
        Module._name_stack.append(self.name)
        try:
            yield
        finally:
            Module._name_stack.pop()

    @staticmethod
    def param(name: str, shape, dtype, init_fn, rng=None) -> LiteTensor:
        if Module._current_params is None:
            raise RuntimeError("No param context active.")
        prefix = "/".join(Module._name_stack) if Module._name_stack else ""
        full_name = f"{prefix}/{name}" if prefix else name
        if full_name not in Module._current_params:
            Module._current_params[full_name] = (
                init_fn(shape, key=rng, dtype=dtype)
                if rng is not None else
                init_fn(shape, dtype=dtype)
            )
        return Module._current_params[full_name]

    def __call__(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        raise NotImplementedError

    def init(self, x: LiteTensor, rng: RNGKey) -> ParamType:
        # Save current state
        old_stack = Module._name_stack.copy()
        old_counters = Module._name_counters.copy()
        
        # Reset for this initialization
        Module._name_stack = []
        Module._name_counters = {}
        
        params: ParamType = {}
        with self.param_context(params):
            with self.name_context():
                _ = self(x, rng)
        
        # Restore previous state
        Module._name_stack = old_stack
        Module._name_counters = old_counters
        
        return params

    def apply(self, params: ParamType, x: LiteTensor, rng: RNGKey, *args, **kwargs) -> LiteTensor:
        # Save current state
        old_stack = Module._name_stack.copy()
        old_counters = Module._name_counters.copy()
        
        # Reset for this application
        Module._name_stack = []
        Module._name_counters = {}
        
        with self.param_context(params):
            with self.name_context():
                result = self(x, rng, *args, **kwargs)
        
        # Restore previous state
        Module._name_stack = old_stack
        Module._name_counters = old_counters
        
        return result