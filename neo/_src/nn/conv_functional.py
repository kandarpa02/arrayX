from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
import torch
from torch.nn.grad import (
    conv1d_input, conv1d_weight,
    conv2d_input, conv2d_weight,
    conv3d_input, conv3d_weight,
)

__all__ = ['conv1d', 'conv2d', 'conv3d']


class _conv1d(Policy):
    def forward(
        self, input, weight, bias=None,
        stride=1, padding=0, dilation=1, groups=1
    ):
        self.ctx.save(
            input, weight, bias,
            stride, padding, dilation, groups
        )
        return torch.nn.functional.conv1d(
            input, weight, bias,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

    def backward(self, grad_output):
        input, weight, bias, stride, padding, dilation, groups = (
            self.ctx.release
        )

        grad_input = conv1d_input(
            input.shape, weight, grad_output,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_weight = conv1d_weight(
            input, weight.shape, grad_output,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_bias = (
            grad_output.sum(dim=(0, 2)) if bias is not None else None
        )

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )


def conv1d(
    input, weight, bias=None,
    stride=1, padding=0, dilation=1, groups=1
):
    return function(_conv1d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )


class _conv2d(Policy):
    def forward(
        self, input, weight, bias=None,
        stride=1, padding=0, dilation=1, groups=1
    ):
        self.ctx.save(
            input, weight, bias,
            stride, padding, dilation, groups
        )
        return torch.nn.functional.conv2d(
            input, weight, bias,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

    def backward(self, grad_output):
        input, weight, bias, stride, padding, dilation, groups = (
            self.ctx.release
        )

        grad_input = conv2d_input(
            input.shape, weight, grad_output,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_weight = conv2d_weight(
            input, weight.shape, grad_output,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_bias = (
            grad_output.sum(dim=(0, 2, 3))
            if bias is not None else None
        )

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )


def conv2d(
    input, weight, bias=None,
    stride=1, padding=0, dilation=1, groups=1
):
    return function(_conv2d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )


class _conv3d(Policy):
    def forward(
        self, input, weight, bias=None,
        stride=1, padding=0, dilation=1, groups=1
    ):
        self.ctx.save(
            input, weight, bias,
            stride, padding, dilation, groups
        )
        return torch.nn.functional.conv3d(
            input, weight, bias,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

    def backward(self, grad_output):
        input, weight, bias, stride, padding, dilation, groups = (
            self.ctx.release
        )

        grad_input = conv3d_input(
            input.shape, weight, grad_output,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_weight = conv3d_weight(
            input, weight.shape, grad_output,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_bias = (
            grad_output.sum(dim=(0, 2, 3, 4))
            if bias is not None else None
        )

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )


def conv3d(
    input, weight, bias=None,
    stride=1, padding=0, dilation=1, groups=1
):
    return function(_conv3d)(
        input, weight, bias,
        stride, padding, dilation, groups
    )
