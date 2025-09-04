from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch

def log(x):
    out = torch.log(x)

    def log_backward(grad):
        return grad / x

    with Tracelet() as t:
        t.register(out, (x,), log_backward)

    return out


def log10(x):
    out = torch.log10(x)

    def log10_backward(grad):
        return grad / (x * torch.log(torch.tensor(10.0, dtype=x.dtype, device=x.device)))

    with Tracelet() as t:
        t.register(out, (x,), log10_backward)

    return out
