from typing import Optional, Union, Tuple
import torch
from torch.nn.grad import (
    conv1d_input, conv1d_weight,
    conv2d_input, conv2d_weight,
    conv3d_input, conv3d_weight,
)
from nexnet._src.autograd.FUNCTION_REGISTER import custom_grad


@custom_grad
def conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1
):
    stride = (stride,) if isinstance(stride, int) else stride
    padding = (padding,) if isinstance(padding, int) else padding
    dilation = (dilation,) if isinstance(dilation, int) else dilation

    out = torch.nn.functional.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    def backward(grad):
        grad_input = conv1d_input(
            input.shape, weight, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        grad_weight = conv1d_weight(
            input, weight.shape, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        grad_bias = grad.sum(dim=(0, 2)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None

    return out, (input, weight, bias), backward


@custom_grad
def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1
):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    out = torch.nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    def backward(grad):
        grad_input = conv2d_input(
            input.shape, weight, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        grad_weight = conv2d_weight(
            input, weight.shape, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        grad_bias = grad.sum(dim=(0, 2, 3)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None

    return out, (input, weight, bias), backward


@custom_grad
def conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1
):
    stride = (stride, stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation

    out = torch.nn.functional.conv3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    def backward(grad):
        grad_input = conv3d_input(
            input.shape, weight, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        grad_weight = conv3d_weight(
            input, weight.shape, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        grad_bias = grad.sum(dim=(0, 2, 3, 4)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None

    return out, (input, weight, bias), backward


__all__ = ['conv1d', 'conv2d', 'conv3d']
