from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy

from neo._torch.lite_tensor import LiteTensor
from typing import Optional, Union, Tuple

import torch
from torch.nn.grad import (
    conv1d_input, conv1d_weight,
    conv2d_input, conv2d_weight,
    conv3d_input, conv3d_weight,
)

__all__ = ['conv1d', 'conv2d', 'conv3d']


class _conv1d(Policy):
    def forward(
        self, input, weight, bias,
        stride=1, padding=0, dilation=1, groups=1
    ):
        self.ctx.save(
            input, weight, bias,
            stride, padding, dilation, groups
        )
        return torch.nn.functional.conv1d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

    def backward(self, grad):
        input, weight, bias, stride, padding, dilation, groups = (
            self.ctx.release
        )

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

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )


def conv1d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
):
    """
    Performs a 1D convolution over an input LiteTensor composed of several input planes.

    Parameters
    ----------
    input : LiteTensor
        Input LiteTensor of shape (N, C_in, L_in), where
        - N is the batch size,
        - C_in is the number of input channels,
        - L_in is the length of the input sequence.
    weight : LiteTensor
        Convolution weights of shape (C_out, C_in // groups, K),
        where K is the kernel size.
    bias : LiteTensor, optional
        Optional bias LiteTensor of shape (C_out,). If None, no bias is added.
    stride : int, optional
        Stride of the convolution. Default: 1.
    padding : int, optional
        Zero-padding added to both sides of the input. Default: 0.
    dilation : int, optional
        Spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1.

    Returns
    -------
    LiteTensor
        Output LiteTensor of shape (N, C_out, L_out), where L_out is determined by the convolution parameters.

    Notes
    -----
    - Supports autograd with exact gradient computation via `neo`'s custom backward policy.
    - Gradient computation is based on PyTorch's `conv1d_input`, `conv1d_weight` utilities.
    """
    return function(_conv1d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )


class _conv2d(Policy):
    def forward(
        self, input, weight, bias,
        stride=1, padding=0, dilation=1, groups=1
    ):
        self.ctx.save(
            input, weight, bias,
            stride, padding, dilation, groups
        )
        return torch.nn.functional.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

    def backward(self, grad):
        input, weight, bias, stride, padding, dilation, groups = (
            self.ctx.release
        )

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

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )

def conv2d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
):
    """
    Performs a 2D convolution over an input LiteTensor composed of several input planes.

    Parameters
    ----------
    input : LiteTensor
        Input LiteTensor of shape (N, C_in, H_in, W_in), where
        - N is the batch size,
        - C_in is the number of input channels,
        - H_in, W_in are the height and width of the input image.
    weight : LiteTensor
        Convolution weights of shape (C_out, C_in // groups, K_H, K_W),
        where K_H and K_W are the kernel height and width.
    bias : LiteTensor, optional
        Optional bias LiteTensor of shape (C_out,). If None, no bias is added.
    stride : int or tuple, optional
        Stride of the convolution. Default: 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1.

    Returns
    -------
    LiteTensor
        Output LiteTensor of shape (N, C_out, H_out, W_out), where output dimensions are determined by the convolution parameters.

    Notes
    -----
    - Implements full support for autograd via `neo`'s functional interface.
    - Backward gradients use native PyTorch ops for accuracy and speed.
    """
    return function(_conv2d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )



class _conv3d(Policy):
    def forward(
        self, input, weight, bias,
        stride=1, padding=0, dilation=1, groups=1
    ):
        self.ctx.save(
            input, weight, bias,
            stride, padding, dilation, groups
        )
        return torch.nn.functional.conv3d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

    def backward(self, grad):
        input, weight, bias, stride, padding, dilation, groups = (
            self.ctx.release
        )

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

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )


def conv3d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
):
    """
    Performs a 3D convolution over an input LiteTensor composed of several input planes.

    Parameters
    ----------
    input : LiteTensor
        Input LiteTensor of shape (N, C_in, D_in, H_in, W_in), where
        - N is the batch size,
        - C_in is the number of input channels,
        - D_in, H_in, W_in are the depth, height, and width of the input volume.
    weight : LiteTensor
        Convolution weights of shape (C_out, C_in // groups, K_D, K_H, K_W),
        where K_D, K_H, and K_W are the depth, height, and width of the kernel.
    bias : LiteTensor, optional
        Optional bias LiteTensor of shape (C_out,). If None, no bias is added.
    stride : int or tuple, optional
        Stride of the convolution. Default: 1.
    padding : int or tuple, optional
        Zero-padding added to all three sides of the input. Default: 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1.

    Returns
    -------
    LiteTensor
        Output LiteTensor of shape (N, C_out, D_out, H_out, W_out), where output dimensions are computed based on the convolution parameters.

    Notes
    -----
    - Integrated with `neo`'s custom autograd backend.
    - Backpropagation leverages PyTorch’s internal `conv3d_input` and `conv3d_weight` operations.
    """
    return function(_conv3d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )
