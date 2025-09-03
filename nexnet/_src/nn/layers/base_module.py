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
            count = Module._name_counters.get(cls_name, 0)
            name = f"{cls_name}_{count}"
            Module._name_counters[cls_name] = count + 1
        self.name = name

    @contextmanager
    def param_context(self, params: ParamType):
        old_params, old_stack, old_counters = (
            Module._current_params, Module._name_stack, Module._name_counters
        )
        Module._current_params = params
        Module._name_stack = []
        Module._name_counters = {}  # reset counters for each init/apply
        try:
            yield
        finally:
            Module._current_params, Module._name_stack, Module._name_counters = (
                old_params, old_stack, old_counters
            )

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
        params: ParamType = {}
        with self.param_context(params):
            _ = self(x, rng)
        return params

    def apply(self, params: ParamType, x: LiteTensor, rng: RNGKey, *args, **kwargs) -> LiteTensor:
        with self.param_context(params):
            return self(x, rng, *args, **kwargs)

