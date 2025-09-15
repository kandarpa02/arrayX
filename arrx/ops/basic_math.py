from ..typing import Array
from ..core.Array import ArrayImpl, ArrayStorage
from arrx import lib

# helper to wrap scalars/arrays into ArrayImpl
def shift(data):
    if isinstance(data, ArrayImpl):
        return data
    if isinstance(data, ArrayStorage):
        return ArrayImpl(data)
    return ArrayImpl(lib.array(data), parents=(), bwd_fn=None)


# -------------------
# Logarithmic functions
# -------------------

def log(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.log(x._rawbuffer), (x,), None)

    def _bwd_log(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad * (1 / x)
        return (g,)

    out.bwd_fn = _bwd_log
    return out


def log10(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.log10(x._rawbuffer), (x,), None)

    ln10 = shift(lib.log(10.0))

    def _bwd_log10(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad / (x * ln10)
        return (g,)

    out.bwd_fn = _bwd_log10
    return out


def exp(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.exp(x._rawbuffer), (x,), None)

    def _bwd_exp(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad * out
        return (g,)

    out.bwd_fn = _bwd_exp
    return out


# -------------------
# Trigonometric functions
# -------------------

def sin(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.sin(x._rawbuffer), (x,), None)

    def _bwd_sin(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad * lib.cos(x._rawbuffer)
        return (g,)

    out.bwd_fn = _bwd_sin
    return out


def cos(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.cos(x._rawbuffer), (x,), None)

    def _bwd_cos(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad * -lib.sin(x._rawbuffer)
        return (g,)

    out.bwd_fn = _bwd_cos
    return out


def tan(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.tan(x._rawbuffer), (x,), None)

    def _bwd_tan(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        # derivative: 1 / cos(x)^2
        cos_x = lib.cos(x._rawbuffer)
        g = grad / (cos_x * cos_x)
        return (g,)

    out.bwd_fn = _bwd_tan
    return out


# -------------------
# Common activation functions
# -------------------

def tanh(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.tanh(x._rawbuffer), (x,), None)

    def _bwd_tanh(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad * (1 - out * out)
        return (g,)

    out.bwd_fn = _bwd_tanh
    return out


def relu(x: Array):
    x = shift(x)
    out = ArrayImpl(lib.maximum(0, x._rawbuffer), (x,), None)

    def _bwd_relu(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        mask = lib.where(x._rawbuffer > 0, 1.0, 0.0)
        g = grad * mask
        return (g,)

    out.bwd_fn = _bwd_relu
    return out


def sigmoid(x: Array):
    x = shift(x)
    exp_neg_x = lib.exp(-x._rawbuffer)
    sig = 1 / (1 + exp_neg_x)
    out = ArrayImpl(sig, (x,), None)

    def _bwd_sigmoid(grad):
        if not isinstance(grad, ArrayImpl):
            grad = shift(grad)
        g = grad * out * (1 - out)
        return (g,)

    out.bwd_fn = _bwd_sigmoid
    return out
