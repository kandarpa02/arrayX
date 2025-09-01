from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Tracelet, custom_grad
from neo._src.nn._linear_fused_fn import *
from neo._torch.lite_tensor import LiteTensor
from typing import Any

# plain linear

@custom_grad
def _linear(X, w, b):
    out_t, X_, w_ = linear_fwd(X.data, w.data, b.data)

    def linear_backward(grad):
        return linear_bwd(grad, X_, w_)
    
    out = LiteTensor(out_t)
    return out, (X, w, b), linear_backward

# linear + relu
@custom_grad
def _linear_relu(X, w, b):
    out_t, X_, w_ = linear_relu_fwd(X.data, w.data, b.data)

    def linear_relu_backward(grad):
        return linear_relu_bwd(grad, out_t, X_, w_)

    out = LiteTensor(out_t)
    return out, (X, w, b), linear_relu_backward


# linear + tanh
@custom_grad
def _linear_tanh(X, w, b):
    out_t, X_, w_ = linear_tanh_fwd(X.data, w.data, b.data)

    def linear_tanh_backward(grad):
        return linear_tanh_bwd(grad, out_t, X_, w_)

    out = LiteTensor(out_t)
    return out, (X, w, b), linear_tanh_backward


# linear + sigmoid
@custom_grad
def _linear_sigmoid(X, w, b):
    out_t, out_act, X_, w_ = linear_sigmoid_fwd(X.data, w.data, b.data)

    def linear_sigmoid_backward(grad):
        return linear_sigmoid_bwd(grad, out_act, X_, w_)

    out = LiteTensor(out_t)
    return out, (X, w, b), linear_sigmoid_backward


# linear + softmax
@custom_grad
def _linear_softmax(X, w, b):
    out_t, out_act, X_, w_ = linear_softmax_fwd(X.data, w.data, b.data)

    def linear_softmax_backward(grad):
        return linear_softmax_bwd(grad, out_act, X_, w_)

    out = LiteTensor(out_t)
    return out, (X, w, b), linear_softmax_backward


# linear + leakyrelu
@custom_grad
def _linear_leakyrelu(X, w, b):
    out_t, out_act, X_, w_, slope = linear_leakyrelu_fwd(X.data, w.data, b.data)

    def linear_leakyrelu_backward(grad):
        return linear_leakyrelu_bwd(grad, out_act, X_, w_, slope)

    out = LiteTensor(out_t)
    return out, (X, w, b), linear_leakyrelu_backward


# linear + relu6
@custom_grad
def _linear_relu6(X, w, b):
    out_t, out_act, X_, w_ = linear_relu6_fwd(X.data, w.data, b.data)

    def linear_relu6_backward(grad):
        return linear_relu6_bwd(grad, out_act, X_, w_)

    out = LiteTensor(out_t)
    return out, (X, w, b), linear_relu6_backward



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

    return _fn(x, w, b)

