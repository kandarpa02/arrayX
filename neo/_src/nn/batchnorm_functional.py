import neo
from typing import Optional
from neo._torch.lite_tensor import LiteTensor
from neo._src.autograd.FUNCTION_REGISTER import Tracelet
import torch
import torch.nn.functional as F


import neo
from neo._torch.lite_tensor import LiteTensor
from neo._src.autograd.FUNCTION_REGISTER import Tracelet
from typing import Optional
import torch

def batchnorm2d(
    x: LiteTensor,
    gamma: LiteTensor,
    beta: LiteTensor,
    running_mean: Optional[LiteTensor] = None,
    running_var: Optional[LiteTensor] = None,
    momentum=0.0,
    eps=1e-5,
    train=True
) -> tuple[LiteTensor, Optional[LiteTensor], Optional[LiteTensor]]:
    """
    Returns:
        out: normalized tensor
        updated_running_mean: updated mean (same as input if train=False)
        updated_running_var: updated var (same as input if train=False)
    """

    # Pre-broadcast gamma/beta
    gamma_b = gamma.data.view(1, -1, 1, 1)
    beta_b = beta.data.view(1, -1, 1, 1)

    if train:
        # Compute batch statistics
        mean = x.data.mean(dim=(0,2,3), keepdim=True)
        var = x.data.var(dim=(0,2,3), unbiased=False, keepdim=True)

        # Update running stats if provided
        if running_mean is not None and running_var is not None:
            updated_running_mean = (1 - momentum) * running_mean.data + momentum * mean
            updated_running_var  = (1 - momentum) * running_var.data  + momentum * var
        else:
            updated_running_mean = None
            updated_running_var  = None
    else:
        # Use running stats for inference
        mean = running_mean.data.view(1, -1, 1, 1)
        var  = running_var.data.view(1, -1, 1, 1)
        updated_running_mean = running_mean
        updated_running_var  = running_var

    # Forward normalization
    x_norm = (x.data - mean) / (var + eps).sqrt()
    out_data = gamma_b * x_norm + beta_b
    out = LiteTensor(out_data, d_type=x.dtype)

    # Fused backward formula
    def bn2d_backward(grad, x=x.data, gamma_b=gamma_b, x_norm=x_norm, var=var, eps=eps):
        N = x.shape[0] * x.shape[2] * x.shape[3]  # elements per channel
        dx_norm = grad * gamma_b
        # Fused: dx = (dx_norm - mean(dx_norm) - x_norm*mean(dx_norm*x_norm)) / sqrt(var+eps)
        dx = (1.0 / (var + eps).sqrt()) * (
            dx_norm
            - dx_norm.mean(dim=(0,2,3), keepdim=True)
            - x_norm * (dx_norm * x_norm).mean(dim=(0,2,3), keepdim=True)
        )
        dgamma = (grad * x_norm).sum(dim=(0,2,3))
        dbeta = grad.sum(dim=(0,2,3))
        return dx, dgamma, dbeta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), bn2d_backward)

    return out, updated_running_mean, updated_running_var
