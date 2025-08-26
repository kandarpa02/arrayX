from typing import Any, Callable
from neo._torch.random import RNGKey
from neo._torch.lite_tensor import LiteTensor
from .initializers import *
from ..nn.layers.base_module import Layer

class Dense(Layer):
    def __init__(self, out_features: int, nonlin:str|Any=None, initializer:Callable|Any=None, name: str = ''):
        super().__init__(name, is_leaf=True)
        self.out_features = out_features
        self.nonlin = nonlin
        self.init_fn = xavier_uniform if initializer is None else initializer

    def forward(self, x: LiteTensor, rng: RNGKey) -> LiteTensor:
        in_features = x.shape[-1]

        w = self.param(
            f"{self.name}/weight", 
            (self.out_features, in_features), 
            x.dtype, 
            x.device, 
            self.init_fn, 
            rng
            )
        
        b = self.param(
            
            f"{self.name}/bias", 
            (self.out_features,), 
            x.dtype, x.device, 
            neo.zeros, 
            None
            )

        return neo.nn.linear(x, w, b, nonlin=self.nonlin)