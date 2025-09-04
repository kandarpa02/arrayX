from typing import Optional, Tuple
from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch


def _batchnorm1d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                 running_mean: torch.Tensor, running_var: torch.Tensor,
                 momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x, gamma, beta, running_mean, running_var,
        training=train, momentum=momentum, eps=eps
    )

    def backward(grad):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad, x, gamma, running_mean, running_var,
            save_mean, save_invstd, train, eps,
            (True, True, True)
        )
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), backward)

    return out, running_mean, running_var


def batchnorm1d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                running_mean: torch.Tensor, running_var: torch.Tensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):
    return _batchnorm1d(x, gamma, beta, running_mean, running_var, momentum, eps, train)


def _batchnorm2d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                 running_mean: torch.Tensor, running_var: torch.Tensor,
                 momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x, gamma, beta, running_mean, running_var,
        training=train, momentum=momentum, eps=eps
    )

    def backward(grad):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad, x, gamma, running_mean, running_var,
            save_mean, save_invstd, train, eps,
            (True, True, True)
        )
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), backward)

    return out, running_mean, running_var


def batchnorm2d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                running_mean: torch.Tensor, running_var: torch.Tensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):
    return _batchnorm2d(x, gamma, beta, running_mean, running_var, momentum, eps, train)


def _batchnorm3d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                 running_mean: torch.Tensor, running_var: torch.Tensor,
                 momentum: float = 0.1, eps: float = 1e-5, train: bool = True):

    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x, gamma, beta, running_mean, running_var,
        training=train, momentum=momentum, eps=eps
    )

    def backward(grad):
        grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
            grad, x, gamma, running_mean, running_var,
            save_mean, save_invstd, train, eps,
            (True, True, True)
        )
        return grad_input, grad_gamma, grad_beta

    with Tracelet() as t:
        t.register(out, (x, gamma, beta), backward)

    return out, running_mean, running_var


def batchnorm3d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                running_mean: torch.Tensor, running_var: torch.Tensor,
                momentum: float = 0.1, eps: float = 1e-5, train: bool = True):
    return _batchnorm3d(x, gamma, beta, running_mean, running_var, momentum, eps, train)
