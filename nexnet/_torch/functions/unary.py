from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch


def abs(x):
    out = x.abs()

    def abs_backward(grad):
        return grad * x.sign()

    with Tracelet() as t:
        t.register(out, (x,), abs_backward)

    return out


def sign(x):
    out = x.sign()

    def sign_backward(grad):
        # sign has zero derivative almost everywhere
        return torch.zeros_like(x)

    with Tracelet() as t:
        t.register(out, (x,), sign_backward)

    return out


def exp(x):
    out = x.exp()

    def exp_backward(grad):
        return grad * out  # derivative of exp is exp(x), already computed in out

    with Tracelet() as t:
        t.register(out, (x,), exp_backward)

    return out


def sqrt(x):
    out = x.sqrt()

    def sqrt_backward(grad):
        return grad * 0.5 / out  # d/dx sqrt(x) = 1 / (2 * sqrt(x))

    with Tracelet() as t:
        t.register(out, (x,), sqrt_backward)

    return out
