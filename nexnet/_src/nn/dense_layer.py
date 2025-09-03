import nexnet
from typing import Any, Callable
from nexnet._torch.random import RNGKey
from nexnet._torch.lite_tensor import LiteTensor
from .initializers import *
from ..nn.layers.base_module import Module


class Dense(Module):
    def __init__(self, out_features: int, nonlin: str | Any = None, initializer: Callable | Any = None, name: str = ''):
        super().__init__(name)
        self.out_features = out_features
        self.nonlin = nonlin
        self.init_fn = xavier_uniform if initializer is None else initializer

    def __call__(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        in_features = x.shape[-1]

        w = self.param(
            f"{self.name}/weight",
            (self.out_features, in_features),
            x.dtype,
            self.init_fn,
            rng
        )

        b = self.param(
            f"{self.name}/bias",
            (self.out_features,),
            x.dtype,
            nexnet.zeros,
            None
        )

        return nexnet.nn.linear(x, w, b, nonlin=self.nonlin)
