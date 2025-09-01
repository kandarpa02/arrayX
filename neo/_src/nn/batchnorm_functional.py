from typing import Optional, Tuple
import neo
from neo._src.autograd.FUNCTION_REGISTER import Tracelet
from neo._torch.lite_tensor import LiteTensor

def batchnorm2d(
    x: LiteTensor,
    gamma: LiteTensor,
    beta: LiteTensor,
    running_mean: Optional[LiteTensor] = None,
    running_var: Optional[LiteTensor] = None,
    momentum: float = 0.1,
    eps: float = 1e-5,
    train: bool = True
) -> Tuple[LiteTensor, Optional[LiteTensor], Optional[LiteTensor]]:
    """
    Neo BatchNorm2d (Flax/Haiku style).
    """

    C = x.data.shape[1]

    if running_mean is None:
        running_mean = neo.zeros((C,), dtype=x.dtype)
    if running_var is None:
        running_var = neo.ones((C,), dtype=x.dtype)

    if train:
        mean = x.data.mean(dim=(0,2,3), keepdim=True)   # (1,C,1,1)
        var  = x.data.var(dim=(0,2,3), unbiased=False, keepdim=True)

        updated_running_mean = LiteTensor(
            (1 - momentum) * running_mean.data + momentum * mean.view(-1),
            d_type=x.dtype
        )
        updated_running_var = LiteTensor(
            (1 - momentum) * running_var.data + momentum * var.view(-1),
            d_type=x.dtype
        )
    else:
        mean = running_mean.data.view(1, -1, 1, 1)
        var  = running_var.data.view(1, -1, 1, 1)
        updated_running_mean = running_mean
        updated_running_var = running_var

    # normalize
    x_norm = (x.data - mean) / (var + eps).sqrt()
    out_data = gamma.data.view(1, -1, 1, 1) * x_norm + beta.data.view(1, -1, 1, 1)
    out = LiteTensor(out_data, d_type=x.dtype)

    # backward
    def bn2d_backward(grad, x=x, gamma=gamma, beta=beta, x_norm=x_norm, var=var, eps=eps):
        N = x.data.shape[0] * x.data.shape[2] * x.data.shape[3]

        gamma_b = gamma.data.view(1, -1, 1, 1)

        dx_norm = grad * gamma_b
        dx = (1. / N) * (1. / (var + eps).sqrt()) * (
            N * dx_norm
            - dx_norm.sum(dim=(0,2,3), keepdim=True)
            - x_norm * (dx_norm * x_norm).sum(dim=(0,2,3), keepdim=True)
        )

        dgamma = (grad * x_norm).sum(dim=(0,2,3))
        dbeta  = grad.sum(dim=(0,2,3))

        return (
            dx,
            dgamma,
            dbeta
        )

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), bn2d_backward)

    return out, updated_running_mean, updated_running_var
