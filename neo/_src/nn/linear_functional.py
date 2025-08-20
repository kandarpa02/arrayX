from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib
from neo._torch.lite_tensor import LiteTensor
from typing import Any

def linear_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b), X, w

def linear_bwd(grad: neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    return grad @ w, grad.T @ X, grad.sum(0)

def linear_relu_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b).relu_(), X, w

def linear_relu_bwd(grad: neolib.Tensor, out:neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    mask = out > 0
    grad = grad.mul_(mask)
    return grad @ w, grad.T @ X, grad.sum(0)


class _linear(Policy):
    def forward(self, X, w, b):
        out, X, w = linear_fwd(X, w, b)
        self.ctx.save(X, w)
        return out

    def backward(self, grad):
        X, w = self.ctx.release
        return linear_bwd(grad, X, w)
    

class _linear_relu(Policy):
    def forward(self, X, w, b):
        out, X, w = linear_relu_fwd(X, w, b)
        self.ctx.save(out, X, w)
        return out

    def backward(self, grad):
        out, X, w = self.ctx.release
        return linear_relu_bwd(grad, out, X, w)
    

NONLIN_DICT = {
    None:_linear,
    "relu":_linear_relu,
    "tanh": None
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
