from typing import Optional, Tuple
import neo
from neo._src.autograd.FUNCTION_REGISTER import Tracelet
from neo._torch.lite_tensor import LiteTensor

def batchnorm2d_fused(
    x: LiteTensor,
    gamma: LiteTensor,
    beta: LiteTensor,
    running_mean: Optional[LiteTensor] = None,
    running_var: Optional[LiteTensor] = None,
    momentum: float = 0.1,
    eps: float = 1e-5,
    train: bool = True
) -> Tuple[LiteTensor, Optional[LiteTensor], Optional[LiteTensor]]:

    C = x.data.shape[1]

    # init running stats if None
    if running_mean is None:
        running_mean = neo.zeros((C,), dtype=x.dtype)
    if running_var is None:
        running_var = neo.ones((C,), dtype=x.dtype)

    # forward stats
    if train:
        mean = x.data.mean(dim=(0,2,3), keepdim=True)
        var  = x.data.var(dim=(0,2,3), unbiased=False, keepdim=True)
        updated_running_mean = LiteTensor((1-momentum)*running_mean.data + momentum*mean.squeeze(), d_type=x.dtype)
        updated_running_var  = LiteTensor((1-momentum)*running_var.data  + momentum*var.squeeze(), d_type=x.dtype)
    else:
        mean = running_mean.data.view(1,-1,1,1)
        var  = running_var.data.view(1,-1,1,1)
        updated_running_mean = running_mean
        updated_running_var  = running_var

    # compute output (forward)
    inv_std = 1.0 / (var + eps).sqrt()
    out_data = gamma.data.view(1,-1,1,1)*(x.data - mean)*inv_std + beta.data.view(1,-1,1,1)
    out = LiteTensor(out_data, d_type=x.dtype)

    # backward: everything recomputed in the closure
    def bn2d_backward(grad, x=x, gamma=gamma):
        xd = x.data
        gd = gamma.data
        grad = grad.to(device=xd.device, dtype=xd.dtype)

        N = xd.shape[0] * xd.shape[2] * xd.shape[3]

        # recompute mean/var/inv_std/x_norm per channel
        mean = xd.mean(dim=(0,2,3), keepdim=True)
        var  = xd.var(dim=(0,2,3), unbiased=False, keepdim=True)
        inv_std = 1.0 / (var + eps).sqrt()
        x_norm = (xd - mean) * inv_std

        # compute param gradients
        dbeta  = grad.sum(dim=(0,2,3))
        dgamma = (grad * x_norm).sum(dim=(0,2,3))

        # compute input gradient in-place style (minimal temporaries)
        sum_grad = grad.sum(dim=(0,2,3), keepdim=True)
        sum_grad_xnorm = (grad * x_norm).sum(dim=(0,2,3), keepdim=True)
        g_view = gd.view(1, -1, 1, 1)
        dx = (1.0 / N) * g_view * inv_std * (N*grad - sum_grad - x_norm*sum_grad_xnorm)

        # return minimal outputs
        return dx, dgamma, dbeta

    # register only minimal things: x and gamma
    with Tracelet() as t:
        t.register(out, (x, gamma), bn2d_backward)

    return out, updated_running_mean, updated_running_var
