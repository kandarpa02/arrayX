from neo._src.nn.layers.base_module import Layer
from neo._src.nn.batchnorm_functional import batchnorm2d
import neo

class BatchNorm2D(Layer):
    def __init__(self, num_features: int, momentum=0.0, eps: float = 1e-5, train=True, name: str = ""):
        super().__init__(name, is_leaf=True)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.train = train

    def forward(self, x, rng):
        gamma = self.param(
            f"{self.name}/gamma",
            (self.num_features,),
            x.dtype,
            neo.ones,
            None,
        )
        beta = self.param(
            f"{self.name}/beta",
            (self.num_features,),
            x.dtype,
            neo.zeros,
            None,
        )

        return batchnorm2d(
            x, 
            gamma, 
            beta, 
            momentum=self.momentum
            eps=self.eps, 
            train=self.train
            )