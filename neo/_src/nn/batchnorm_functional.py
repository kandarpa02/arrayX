from typing import Optional, Tuple
import neo
from neo._src.autograd.FUNCTION_REGISTER import Tracelet, custom_grad
from neo._torch.lite_tensor import LiteTensor
import torch.nn.functional as F
import torch


def _batchnorm1d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                 running_mean: LiteTensor, running_var: LiteTensor,
                 momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x.data, gamma.data, beta.data,
        running_mean.data, running_var.data,
        training=train, momentum=momentum, eps=eps
    )
    out = LiteTensor(out)

    def backward(grad, x=x, gamma=gamma, beta=beta,
                 save_mean=save_mean, save_invstd=save_invstd):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad,
            x.data,
            gamma.data,
            running_mean.data,
            running_var.data,
            save_mean,
            save_invstd,
            train,
            eps,
            (True, True, True)
        )
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), backward)

    return out, running_mean, running_var


def batchnorm1d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                running_mean: LiteTensor, running_var: LiteTensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):
    return _batchnorm1d(x, gamma, beta, running_mean, running_var, momentum, eps, train)


def _batchnorm2d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                 running_mean: LiteTensor, running_var: LiteTensor,
                 momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    # Forward
    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x.data, gamma.data, beta.data,
        running_mean.data, running_var.data,
        training=train, momentum=momentum, eps=eps
    )
    out = LiteTensor(out)

    # Backward
    def backward(grad, x=x, gamma=gamma, beta=beta,
                 save_mean=save_mean, save_invstd=save_invstd):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad,
            x.data,
            gamma.data,
            running_mean.data,
            running_var.data,
            save_mean,
            save_invstd,
            train,
            eps,
            (True, True, True)
        )
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), backward)

    return out, running_mean, running_var

    
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
    

def _batchnorm3d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                 running_mean: LiteTensor, running_var: LiteTensor,
                 momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x.data, gamma.data, beta.data,
        running_mean.data, running_var.data,
        training=train, momentum=momentum, eps=eps
    )
    out = LiteTensor(out)

    def backward(grad, x=x, gamma=gamma, beta=beta,
                 save_mean=save_mean, save_invstd=save_invstd):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad,
            x.data,
            gamma.data,
            running_mean.data,
            running_var.data,
            save_mean,
            save_invstd,
            train,
            eps,
            (True, True, True)
        )
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), backward)

    return out, running_mean, running_var


def batchnorm3d(x: LiteTensor, gamma: LiteTensor, beta: LiteTensor,
                running_mean: LiteTensor, running_var: LiteTensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):
    return _batchnorm3d(x, gamma, beta, running_mean, running_var, momentum, eps, train)
