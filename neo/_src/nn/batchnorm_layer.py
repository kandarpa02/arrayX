from neo._src.nn.layers.base_module import Layer
from neo._src.nn.batchnorm_functional import batchnorm2d
import neo
from typing import Optional, Dict, Tuple

from neo._src.nn.layers.base_module import Layer
from neo._src.nn.batchnorm_functional import batchnorm2d
import neo
from typing import Optional, Dict, Tuple


class BatchNorm2D(Layer):
    def __init__(self, num_features: int, momentum=0.1, eps: float = 1e-5, name: str = ""):
        super().__init__(name, is_leaf=True)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Initialize running statistics as parameters (not trainable)
        self.running_mean = neo.zeros((num_features,))
        self.running_var = neo.ones((num_features,))

    def forward(
        self,
        x: neo.LiteTensor,
        rng,
        state: Optional[Dict[str, neo.LiteTensor]] = None,
        train: bool = True
    ) -> Tuple[neo.LiteTensor, Dict[str, neo.LiteTensor]]:
        """
        BatchNorm forward.
        """

        gamma = self.param(f"{self.name}/gamma", (self.num_features,), x.dtype, neo.ones, None)
        beta = self.param(f"{self.name}/beta", (self.num_features,), x.dtype, neo.zeros, None)

        # Call functional batchnorm with current running stats
        out, updated_mean, updated_var = batchnorm2d(
            x,
            gamma,
            beta,
            running_mean=self.running_mean,
            running_var=self.running_var,
            momentum=self.momentum,
            eps=self.eps,
            train=train
        )

        # Update running statistics if in training mode
        if train:
            self.running_mean = updated_mean
            self.running_var = updated_var

        # Return empty state since we're storing running stats internally
        return out, {}