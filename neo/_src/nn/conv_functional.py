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

        def to_tuple(x):
            return (x,) if isinstance(x, int) else x

        stride = to_tuple(stride)
        padding = to_tuple(padding)
        dilation = to_tuple(dilation)

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
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1
):
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

        def to_tuple(x):
            return (x, x) if isinstance(x, int) else x

        stride = to_tuple(stride)
        padding = to_tuple(padding)
        dilation = to_tuple(dilation)

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

        def to_tuple(x):
            return (x, x, x) if isinstance(x, int) else x

        stride = to_tuple(stride)
        padding = to_tuple(padding)
        dilation = to_tuple(dilation)

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
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1
):
    return function(_conv3d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )
