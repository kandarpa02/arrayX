from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch

def log(x):
    out = x.unary_op(lambda x: x.log())

    def log_backward(grad, x=x.data):
        grad = torch.as_tensor(grad, dtype=x.dtype, device=x.device)
        return grad / x  

    with Tracelet() as t:
        t.register(out, (x,), log_backward)

    return out


def log10(x):
    out = x.unary_op(lambda x: x.log10())

    def log10_backward(grad, x=x.data):
        grad = torch.as_tensor(grad, dtype=x.dtype, device=x.device)
        return grad / (x * torch.log(torch.tensor(10, dtype=x.dtype, device=x.device)))

    with Tracelet() as t:
        t.register(out, (x,), log10_backward)

    return out
