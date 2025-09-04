import nexnet as nx
from typing import Any, Callable
from nexnet._torch.random import RNGKey
from .initializers import *
from ..nn.layers.base_module import Module
import torch


class Dense(Module):
    def __init__(
        self,
        out_features: int,
        nonlin: str | Any = None,
        initializer: Callable | Any = None,
        name: str = ''
    ):
        super().__init__(name)
        self.out_features = out_features
        self.nonlin = nonlin
        self.init_fn = xavier_uniform if initializer is None else initializer

    def __call__(self, x: torch.Tensor, rng: RNGKey) -> torch.Tensor:
        in_features = x.shape[-1]

        with self.name_context():
            w = self.param(
                "weight",
                (self.out_features, in_features),
                x.dtype,
                self.init_fn,
                rng
            )

            b = self.param(
                "bias",
                (self.out_features,),
                x.dtype,
                zero_init,
                None
            )

        return nx.nn.linear(x, w, b, nonlin=self.nonlin)
