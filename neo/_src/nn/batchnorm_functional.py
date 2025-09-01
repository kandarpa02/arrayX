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
    Flax/Haiku-style 2D BatchNorm.

    Args:
        x: input LiteTensor of shape (N, C, H, W)
        gamma: scale parameter (C,)
        beta: shift parameter (C,)
        running_mean: running mean (C,), updated only in train mode
        running_var: running var (C,), updated only in train mode
        momentum: for running stats
        eps: numerical stability
        train: True for training, False for inference

    Returns:
        out: normalized tensor
        updated_running_mean: updated running mean (C,) or same as input if train=False
        updated_running_var: updated running var (C,) or same as input if train=False
    """

    # Pre-broadcast gamma/beta for channel-wise scaling
    gamma_b = gamma.data.view(1, -1, 1, 1)
    beta_b = beta.data.view(1, -1, 1, 1)

    # Initialize running stats if missing
    if running_mean is None:
        running_mean = neo.zeros((x.data.shape[1],), dtype=x.dtype)
    if running_var is None:
        running_var = neo.ones((x.data.shape[1],), dtype=x.dtype)

    if train:
        # Compute batch stats
        mean = x.data.mean(dim=(0, 2, 3), keepdim=True)   # shape (1, C, 1, 1)
        var  = x.data.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        # Update running stats (keep 1D shape)
        updated_running_mean = neo.Tensor(
            (1 - momentum) * running_mean.data + momentum * mean.view(-1),
            dtype=x.dtype
        )
        updated_running_var = neo.Tensor(
            (1 - momentum) * running_var.data + momentum * var.view(-1),
            dtype=x.dtype
        )
    else:
        # Use running stats for inference, expand for broadcasting
        mean = running_mean.data.view(1, -1, 1, 1)
        var  = running_var.data.view(1, -1, 1, 1)
        updated_running_mean = running_mean
        updated_running_var = running_var

    # Normalize
    x_norm = (x.data - mean) / (var + eps).sqrt()
    out_data = gamma_b * x_norm + beta_b
    out = LiteTensor(out_data, d_type=x.dtype)

    # Backward (fused formula)
    def bn2d_backward(grad, x=x.data, gamma_b=gamma_b, x_norm=x_norm, var=var, eps=eps):
        N = x.shape[0]*x.shape[2]*x.shape[3]  # total elements per channel
        dx_norm = grad * gamma_b

        # Corrected fused formula
        dx = (1. / N) * (1. / (var + eps).sqrt()) * (
            N * dx_norm
            - dx_norm.sum(dim=(0,2,3), keepdim=True)
            - x_norm * (dx_norm * x_norm).sum(dim=(0,2,3), keepdim=True)
        )

        dgamma = (grad * x_norm).sum(dim=(0,2,3))
        dbeta  = grad.sum(dim=(0,2,3))
        return dx, dgamma, dbeta


    with Tracelet() as t:
        t.register(out, (x, gamma, beta), bn2d_backward)

    return out, updated_running_mean, updated_running_var
