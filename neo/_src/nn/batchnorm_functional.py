import neo
from neo._torch.lite_tensor import LiteTensor
from neo._src.autograd.FUNCTION_REGISTER import Tracelet
import torch
import torch.nn.functional as F

def batchnorm2d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor, momentum=0.0, eps=1e-5, train=True):
    def bn2d_forward(x_t, gamma, beta, momentum=momentum, eps=eps, train=train):
        return F.batch_norm(
            x_t,
            running_mean=None,
            running_var=None,
            weight=gamma,
            bias=beta,
            training=train,  
            momentum=0.0,
            eps=eps
        )

    out = x.nary_op((gamma, beta), bn2d_forward)

    def bn2d_backward(grad, x=x.data, gamma=gamma.data, beta=beta.data):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        N = x.shape[0] * x.shape[2] * x.shape[3]
        var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        x_norm = (x - mean) / (var + eps).sqrt()

        dgamma = (grad * x_norm).sum(dim=(0, 2, 3), keepdim=True)
        dbeta = grad.sum(dim=(0, 2, 3), keepdim=True)

        dx_norm = grad * gamma.reshape(1, -1, 1, 1)
        dvar = (dx_norm * (x - mean) * -0.5 * (var + eps).pow(-1.5)).sum(dim=(0, 2, 3), keepdim=True)
        dmean = (dx_norm * -1.0 / (var + eps).sqrt()).sum(dim=(0, 2, 3), keepdim=True) + dvar * ((-2.0 * (x - mean)).mean(dim=(0, 2, 3), keepdim=True))

        dx = dx_norm / (var + eps).sqrt() + dvar * 2.0 * (x - mean) / N + dmean / N
        return dx, dgamma.reshape(gamma.shape), dbeta.reshape(beta.shape)

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), bn2d_backward)

    return out
