from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo._src.nn._linear_fused_fn import *
from neo._torch.lite_tensor import LiteTensor
from typing import Any


# raw linear policy
class _linear(Policy):
    def forward(self, X, w, b):
        out, X, w = linear_fwd(X, w, b)
        self.ctx.save(X, w)
        return out

    def backward(self, grad):
        X, w = self.ctx.release
        return linear_bwd(grad, X, w)
    
# linear relu policy
class _linear_relu(Policy):
    def forward(self, X, w, b):
        out, X, w = linear_relu_fwd(X, w, b)
        self.ctx.save(out, X, w)
        return out

    def backward(self, grad):
        out, X, w = self.ctx.release
        return linear_relu_bwd(grad, out, X, w)
    

# linear tanh policy
class _linear_tanh(Policy):
    def forward(self, X, w, b):
        out, X, w = linear_tanh_fwd(X, w, b)
        self.ctx.save(out, X, w)
        return out

    def backward(self, grad):
        out, X, w = self.ctx.release
        return linear_tanh_bwd(grad, out, X, w)


# linear + sigmoid
class _linear_sigmoid(Policy):
    def forward(self, X, w, b):
        out, out_, X_, w_ = linear_sigmoid_fwd(X, w, b)
        self.ctx.save(out_, X_, w_)
        return out
    def backward(self, grad):
        out, X, w = self.ctx.release
        return linear_sigmoid_bwd(grad, out, X, w)


# linear + softmax
class _linear_softmax(Policy):
    def forward(self, X, w, b):
        out, out_, X_, w_ = linear_softmax_fwd(X, w, b)
        self.ctx.save(out_, X_, w_)
        return out
    def backward(self, grad):
        out, X, w = self.ctx.release
        return linear_softmax_bwd(grad, out, X, w)


# linear + leakyrelu
class _linear_leakyrelu(Policy):
    def forward(self, X, w, b):
        out, out_, X_, w_, slope = linear_leakyrelu_fwd(X, w, b)
        self.ctx.save(out_, X_, w_, slope)
        return out
    def backward(self, grad):
        out, X, w, slope = self.ctx.release
        return linear_leakyrelu_bwd(grad, out, X, w, slope)


# linear + relu6
class _linear_relu6(Policy):
    def forward(self, X, w, b):
        out, out_, X_, w_ = linear_relu6_fwd(X, w, b)
        self.ctx.save(out_, X_, w_)
        return out
    def backward(self, grad):
        out, X, w = self.ctx.release
        return linear_relu6_bwd(grad, out, X, w)


NONLIN_DICT = {
    None: _linear,
    "relu": _linear_relu,
    "tanh": _linear_tanh,
    "sigmoid": _linear_sigmoid,
    "softmax": _linear_softmax,
    "leaky_relu": _linear_leakyrelu,
    "relu6": _linear_relu6,
}


def linear(x: LiteTensor, w: LiteTensor, b: LiteTensor, nonlin:Any|str=None):
    """
    Applies a linear transformation to the incoming data: ( y = xW^T + b )

    This operation is equivalent to a fully-connected layer without activation.
    Automatically supports autograd via Neo's function tracing mechanism.

    Args:
        x (LiteTensor): Input tensor of shape `(N, in_features)` or `(in_features,)`.
        w (LiteTensor): Weight tensor of shape `(out_features, in_features)`.
        b (LiteTensor): Bias tensor of shape `(out_features,)`.

    Returns:
        LiteTensor: Output tensor of shape `(N, out_features)`, where:
            - `N` is the batch size if `x` is 2D
            - `1` if `x` is 1D (internally reshaped)

    Notes:
        - Automatically reshapes 1D input to 2D for compatibility.
        - Supports gradient propagation through all inputs.
        - In-place bias addition is performed for performance.

    Example:
        >>> x = neo.lite([[1.0, 2.0]])
        >>> w = neo.lite([[0.5, 0.5]])
        >>> b = neo.lite([0.1])
        >>> y = linear(x, w, b)  # returns [[1.6]]
    """
    try:
        _fn = NONLIN_DICT[nonlin]
    except KeyError:
        raise KeyError(f"[{nonlin}] is not a valid function")

    if _fn is None:
        raise NotImplementedError(f"[{nonlin}] has not been implemented yet")

    return function(_fn)(x, w, b)


from typing import Dict, Any, List, Optional

def MLP(x: LiteTensor, params: Dict[str, LiteTensor], nonlins: Optional[List[Optional[str]]] = None):
    """
    N-layer MLP using neo.nn.linear with autograd support.
    
    Args:
        x (LiteTensor): Input tensor of shape (N, in_features).
        params (dict): Dictionary of weights & biases:
            {
              "w1": Tensor, "b1": Tensor,
              "w2": Tensor, "b2": Tensor,
              ...
              "wN": Tensor, "bN": Tensor
            }
        nonlins (list[str|None]): List of nonlinearities per layer.
                                  If None, defaults to ReLU for all but last.
                                  Example: ["relu", "relu", None]
    
    Returns:
        LiteTensor: Output tensor of shape (N, out_features).
    """
    num_layers = len(params) // 2
    if nonlins is None:
        # Default: ReLU for all but last
        nonlins = ["relu"] * (num_layers - 1) + [None]
    
    for i in range(1, num_layers + 1):
        w, b = params[f"w{i}"], params[f"b{i}"]
        x = linear(x, w, b, nonlin=nonlins[i - 1])
    return x
