from typing import Optional, Tuple
import neo
from neo._src.autograd.FUNCTION_REGISTER import Tracelet, custom_grad
from neo._torch.lite_tensor import LiteTensor
import torch.nn.functional as F
import torch

def _batchnorm2d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                running_mean: LiteTensor, running_var: LiteTensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    # Forward: use native_batch_norm (updates running_mean / running_var in-place)
    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x.data, gamma.data, beta.data,
        running_mean.data, running_var.data,
        training=train, momentum=momentum, eps=eps
    )

    # Backward closure
    def backward(grad, x=x, gamma=gamma, beta=beta,
                 save_mean=save_mean, save_invstd=save_invstd):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad, x.data, save_mean, save_invstd,
            gamma.data, training=train, eps=eps,
            output_mask=(True, True, True)
        )
        # only grads for x, gamma, beta
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(LiteTensor(out), (x, gamma, beta), backward)
        
    return (
        LiteTensor(out),
        running_mean,
        running_var,
    )

    
def batchnorm2d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                running_mean: LiteTensor, running_var: LiteTensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):
    
    return _batchnorm2d(
        x, 
        gamma, 
        beta,
        running_mean, 
        running_var,
        momentum, 
        eps, 
        train)
    