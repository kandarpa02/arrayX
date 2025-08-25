from typing import Dict, Tuple, Callable, Any
from contextlib import contextmanager
from neo._torch.lite_tensor import LiteTensor
from neo._torch.random import RNGKey

ParamType = Dict[str, LiteTensor]

class Module:
    """
    Base Module class with @compact-style param management.
    """
    _current_params: ParamType|Any = None  

    @contextmanager
    def param_context(self, params: ParamType):
        old = Module._current_params
        Module._current_params = params
        try:
            yield
        finally:
            Module._current_params = old

    @staticmethod
    def param(name: str, shape: Tuple[int, ...], dtype:Any, device:Any, init_fn: Callable, rng: RNGKey|Any=None) -> LiteTensor:
        if Module._current_params is None:
            raise RuntimeError("No param context active. Use `with module.param_context(params)`")
        if name not in Module._current_params:
            Module._current_params[name] = init_fn(shape, key=rng, dtype=dtype, device=device) if rng is not None else init_fn(shape, dtype=dtype, device=device)
        return Module._current_params[name]

    def forward(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        raise NotImplementedError

    def init(self, x: LiteTensor, rng: RNGKey) -> ParamType:
        """
        Stateless init: returns a param dict.
        """
        params: ParamType = {}
        with self.param_context(params):
            _ = self.forward(x, rng)
        return params

    def __call__(self, params: ParamType, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        """
        Forward pass using given params.
        """
        with self.param_context(params):
            return self.forward(x, rng)

class Layer(Module):
    """
    Base class for single layers (Linear, Conv, etc.).
    Handles param initialization inside forward automatically.
    """
    def __init__(self, name: str = ''):
        super().__init__()
        self.name = name or self.__class__.__name__

    def forward(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        raise NotImplementedError

    def __call__(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        return self.forward(x, rng)