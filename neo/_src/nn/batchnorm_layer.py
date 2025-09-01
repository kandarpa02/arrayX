from neo._src.nn.layers.base_module import Layer
from neo._src.nn.batchnorm_functional import batchnorm2d
from neo import LiteTensor, zeros, ones


class BatchNorm2D(Layer):
    def __init__(self, num_features: int, momentum=0.1, eps: float = 1e-5, name: str = ""):
        super().__init__(name, is_leaf=True)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: LiteTensor, rng, state=None, train: bool = True):
        gamma = self.param(f"{self.name}/gamma", (self.num_features,), x.dtype, ones, None)
        beta = self.param(f"{self.name}/beta", (self.num_features,), x.dtype, zeros, None)

        if state is None:
            state = {}

        running_mean = state.get(f"{self.name}/mean")
        running_var = state.get(f"{self.name}/var")

        if running_mean is None:
            running_mean = zeros((self.num_features,), dtype=x.dtype)
        if running_var is None:
            running_var = ones((self.num_features,), dtype=x.dtype)

        out, updated_mean, updated_var = batchnorm2d(
            x, gamma, beta,
            running_mean=running_mean,
            running_var=running_var,
            momentum=self.momentum,
            eps=self.eps,
            train=train
        )

        new_state = {
            f"{self.name}/mean": updated_mean,
            f"{self.name}/var": updated_var,
        }
        return out, new_state
