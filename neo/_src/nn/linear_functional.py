from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib
from neo._torch.lite_tensor import LiteTensor
from typing import Any

# raw linear
def linear_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b), X, w

def linear_bwd(grad: neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    return grad @ w, grad.T @ X, grad.sum(0)

# linear relu
def linear_relu_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b).relu_(), X, w

def linear_relu_bwd(grad: neolib.Tensor, out:neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    mask = out > 0
    grad = grad.mul_(mask)
    return grad @ w, grad.T @ X, grad.sum(0)

# linear tanh
def linear_tanh_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b).tanh_(), X, w

def linear_tanh_bwd(grad: neolib.Tensor, out:neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    grad = grad.mul_(1-out**2)
    return grad @ w, grad.T @ X, grad.sum(0)


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

NONLIN_DICT = {
    None:_linear,
    "relu":_linear_relu,
    "tanh":_linear_tanh
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
