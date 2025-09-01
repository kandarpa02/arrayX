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
    Neo BatchNorm2d .
    """

    C = x.data.shape[1]

    if running_mean is None:
        running_mean = neo.zeros((C,), dtype=x.dtype)
    if running_var is None:
        running_var = neo.ones((C,), dtype=x.dtype)

    if train:
        mean = x.data.mean(dim=(0,2,3), keepdim=True)   # (1,C,1,1)
        var = x.data.var(dim=(0,2,3), unbiased=False, keepdim=True)

        updated_running_mean = LiteTensor(
            (1 - momentum) * running_mean.data + momentum * mean.squeeze(),
            d_type=x.dtype
        )
        updated_running_var = LiteTensor(
            (1 - momentum) * running_var.data + momentum * var.squeeze(),
            d_type=x.dtype
        )
    else:
        mean = running_mean.data.view(1, -1, 1, 1)
        var = running_var.data.view(1, -1, 1, 1)
        updated_running_mean = running_mean
        updated_running_var = running_var

    # normalize
    inv_std = 1.0 / (var + eps).sqrt()
    x_norm = (x.data - mean) * inv_std
    out_data = gamma.data.view(1, -1, 1, 1) * x_norm + beta.data.view(1, -1, 1, 1)
    out = LiteTensor(out_data, d_type=x.dtype)

    # backward
    def bn2d_backward(grad, x=x, gamma=gamma, mean=mean, inv_std=inv_std, x_norm=x_norm):
        xd = x.data
        gd = gamma.data

        # Ensure grad is same device/dtype as x
        grad = grad.to(device=xd.device, dtype=xd.dtype)

        # N = number of elements per channel
        N = xd.shape[0] * xd.shape[2] * xd.shape[3]

        # dbeta / dgamma
        dbeta = grad.sum(dim=(0, 2, 3))                 # shape (C,)
        dgamma = (grad * x_norm).sum(dim=(0, 2, 3))    # shape (C,)

        # input gradient using the compact correct formula
        # keepdim=True for the reductions so broadcasting works cleanly
        sum_grad = grad.sum(dim=(0, 2, 3), keepdim=True)                    # shape (1, C, 1, 1)
        sum_grad_xnorm = (grad * x_norm).sum(dim=(0, 2, 3), keepdim=True)   # shape (1, C, 1, 1)

        # gd view for broadcasting
        g_view = gd.view(1, -1, 1, 1)

        dx = (1.0 / N) * g_view * inv_std * (
            N * grad
            - sum_grad
            - x_norm * sum_grad_xnorm
        )

        # return shapes: dx (N,C,H,W), dgamma (C,), dbeta (C,)
        return dx, dgamma, dbeta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), bn2d_backward)

    return out, updated_running_mean, updated_running_var