from neo._src.autograd.FUNCTION_REGISTER import Tracelet
import torch

def abs(x):
    out = x.unary_op(lambda x: x.abs())

    def abs_backward(grad, x=x.data):
        grad = torch.as_tensor(grad, dtype=x.dtype, device=x.device)
        return grad * x.sign()

    with Tracelet() as t:
        t.register(out, (x,), abs_backward)

    return out


def sign(x):
    out = x.unary_op(lambda x: x.sign())

    def sign_backward(grad, x=x.data):
        # sign has zero derivative almost everywhere
        return torch.zeros_like(x)

    with Tracelet() as t:
        t.register(out, (x,), sign_backward)

    return out


def exp(x):
    out = x.unary_op(lambda x: x.exp())

    def exp_backward(grad, out=out.data):
        grad = torch.as_tensor(grad, dtype=out.dtype, device=out.device)
        return grad * out   # derivative of exp is exp(x), already computed in out

    with Tracelet() as t:
        t.register(out, (x,), exp_backward)

    return out


def sqrt(x):
    out = x.unary_op(lambda x: x.sqrt())

    def sqrt_backward(grad, x=x.data, out=out.data):
        grad = torch.as_tensor(grad, dtype=out.dtype, device=out.device)
        return grad * 0.5 / out  # d/dx sqrt(x) = 1 / (2 * sqrt(x))

    with Tracelet() as t:
        t.register(out, (x,), sqrt_backward)

    return out
