from nexnet.functions import function
from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet, custom_grad
from nexnet._src.nn._linear_fused_fn import *
from typing import Any
import torch

# plain linear
@custom_grad
def _linear(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, X_, w_ = linear_fwd(X, w, b)

    def linear_backward(grad: torch.Tensor):
        return linear_bwd(grad, X_, w_)
    
    return out_t, (X, w, b), linear_backward

# linear + relu
@custom_grad
def _linear_relu(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, X_, w_ = linear_relu_fwd(X, w, b)

    def linear_relu_backward(grad: torch.Tensor):
        return linear_relu_bwd(grad, out_t, X_, w_)

    return out_t, (X, w, b), linear_relu_backward

# linear + tanh
@custom_grad
def _linear_tanh(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, X_, w_ = linear_tanh_fwd(X, w, b)

    def linear_tanh_backward(grad: torch.Tensor):
        return linear_tanh_bwd(grad, out_t, X_, w_)

    return out_t, (X, w, b), linear_tanh_backward

# linear + sigmoid
@custom_grad
def _linear_sigmoid(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, out_act, X_, w_ = linear_sigmoid_fwd(X, w, b)

    def linear_sigmoid_backward(grad: torch.Tensor):
        return linear_sigmoid_bwd(grad, out_act, X_, w_)

    return out_t, (X, w, b), linear_sigmoid_backward

# linear + softmax
@custom_grad
def _linear_softmax(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, out_act, X_, w_ = linear_softmax_fwd(X, w, b)

    def linear_softmax_backward(grad: torch.Tensor):
        return linear_softmax_bwd(grad, out_act, X_, w_)

    return out_t, (X, w, b), linear_softmax_backward

# linear + leakyrelu
@custom_grad
def _linear_leakyrelu(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, out_act, X_, w_, slope = linear_leakyrelu_fwd(X, w, b)

    def linear_leakyrelu_backward(grad: torch.Tensor):
        return linear_leakyrelu_bwd(grad, out_act, X_, w_, slope)

    return out_t, (X, w, b), linear_leakyrelu_backward

# linear + relu6
@custom_grad
def _linear_relu6(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    out_t, out_act, X_, w_ = linear_relu6_fwd(X, w, b)

    def linear_relu6_backward(grad: torch.Tensor):
        return linear_relu6_bwd(grad, out_act, X_, w_)

    return out_t, (X, w, b), linear_relu6_backward


NONLIN_DICT = {
    None: _linear,
    "relu": _linear_relu,
    "tanh": _linear_tanh,
    "sigmoid": _linear_sigmoid,
    "softmax": _linear_softmax,
    "leaky_relu": _linear_leakyrelu,
    "relu6": _linear_relu6,
}


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, nonlin: Any | str = None):
    """
    Applies a linear transformation to the incoming data: ( y = xW^T + b )

    This operation is equivalent to a fully-connected layer without activation.
    Automatically supports autograd via Neo's function tracing mechanism.

    Args:
        x (torch.Tensor): Input tensor of shape `(N, in_features)` or `(in_features,)`.
        w (torch.Tensor): Weight tensor of shape `(out_features, in_features)`.
        b (torch.Tensor): Bias tensor of shape `(out_features,)`.

    Returns:
        torch.Tensor: Output tensor of shape `(N, out_features)`, where:
            - `N` is the batch size if `x` is 2D
            - `1` if `x` is 1D (internally reshaped)
    """
    try:
        _fn = NONLIN_DICT[nonlin]
    except KeyError:
        raise KeyError(f"[{nonlin}] is not a valid function")

    if _fn is None:
        raise NotImplementedError(f"[{nonlin}] has not been implemented yet")

    return _fn(x, w, b)
