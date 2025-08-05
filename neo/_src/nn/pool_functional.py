from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy

import torch
import torch.nn.functional as F

__all__ = [
    'max_pool1d', 'max_pool2d', 'max_pool3d',
    'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
]


class _max_pool1d(Policy):
    def forward(
        self, input, kernel_size, stride=None,
        padding=0, dilation=1, ceil_mode=False
    ):
        output, indices = F.max_pool1d(
            input, kernel_size, stride, padding,
            dilation, ceil_mode, return_indices=True
        )
        self.ctx.save(
            indices, input.shape, kernel_size,
            stride, padding, output.shape
        )
        return output

    def backward(self, grad):
        indices, input_shape, kernel_size, stride, padding, output_shape = (
            self.ctx.release
        )
        return (
            F.max_unpool1d(
                grad, indices, kernel_size,
                stride, padding, output_size=output_shape
            ),
            None, None, None, None, None
        )


def max_pool1d(
    input, kernel_size, stride=None,
    padding=0, dilation=1, ceil_mode=False
):
    return function(_max_pool1d)(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


class _max_pool2d(Policy):
    def forward(
        self, input, kernel_size, stride=None,
        padding=0, dilation=1, ceil_mode=False
    ):
        output, indices = F.max_pool2d(
            input, kernel_size, stride, padding,
            dilation, ceil_mode, return_indices=True
        )
        self.ctx.save(
            indices, input.shape, kernel_size,
            stride, padding, output.shape
        )
        return output

    def backward(self, grad):
        indices, input_shape, kernel_size, stride, padding, output_shape = (
            self.ctx.release
        )
        return (
            F.max_unpool2d(
                grad, indices, kernel_size,
                stride, padding, output_size=output_shape
            ),
            None, None, None, None, None
        )


def max_pool2d(
    input, kernel_size, stride=None,
    padding=0, dilation=1, ceil_mode=False
):
    return function(_max_pool2d)(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


class _max_pool3d(Policy):
    def forward(
        self, input, kernel_size, stride=None,
        padding=0, dilation=1, ceil_mode=False
    ):
        output, indices = F.max_pool3d(
            input, kernel_size, stride, padding,
            dilation, ceil_mode, return_indices=True
        )
        self.ctx.save(
            indices, input.shape, kernel_size,
            stride, padding, output.shape
        )
        return output

    def backward(self, grad):
        indices, input_shape, kernel_size, stride, padding, output_shape = (
            self.ctx.release
        )
        return (
            F.max_unpool3d(
                grad, indices, kernel_size,
                stride, padding, output_size=output_shape
            ),
            None, None, None, None, None
        )


def max_pool3d(
    input, kernel_size, stride=None,
    padding=0, dilation=1, ceil_mode=False
):
    return function(_max_pool3d)(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


class _avg_pool1d(Policy):
    def forward(
        self, input, kernel_size, stride=None,
        padding=0, ceil_mode=False, count_include_pad=True
    ):
        stride = stride or kernel_size
        self.ctx.save(
            input.shape, kernel_size, stride,
            padding, ceil_mode, count_include_pad
        )
        return F.avg_pool1d(
            input, kernel_size, stride,
            padding, ceil_mode, count_include_pad
        )

    def backward(self, grad):
        input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad = self.ctx.release
        batch_size, channels, input_length = input_shape

        weight = torch.ones((channels, 1, kernel_size),
                            dtype=grad.dtype, device=grad.device)

        grad_input = F.conv_transpose1d(
            grad, weight, stride=stride, padding=padding, groups=channels
        )

        if count_include_pad:
            norm = kernel_size
            grad_input = grad_input / norm
        else:
            input_ones = torch.ones(input_shape, dtype=grad.dtype, device=grad.device)
            norm = F.avg_pool1d(
                input_ones, kernel_size, stride,
                padding, ceil_mode, count_include_pad
            )
            norm = F.conv_transpose1d(
                norm, weight, stride=stride, padding=padding, groups=channels
            )
            grad_input = grad_input / (norm + 1e-8)

        return grad_input, None, None, None, None, None


def avg_pool1d(
    input, kernel_size, stride=None,
    padding=0, ceil_mode=False, count_include_pad=True
):
    return function(_avg_pool1d)(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad
    )



class _avg_pool2d(Policy):
    def forward(
        self, input, kernel_size, stride=None,
        padding=0, ceil_mode=False, count_include_pad=True
    ):
        stride = stride or kernel_size
        self.ctx.save(
            input.shape, kernel_size, stride,
            padding, ceil_mode, count_include_pad
        )
        return F.avg_pool2d(
            input, kernel_size, stride,
            padding, ceil_mode, count_include_pad
        )

    def backward(self, grad):
        input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad = self.ctx.release
        batch_size, channels, height, width = input_shape

        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        weight = torch.ones((channels, 1, kH, kW),
                            dtype=grad.dtype, device=grad.device)

        grad_input = F.conv_transpose2d(
            grad, weight, stride=stride, padding=padding, groups=channels
        )

        if count_include_pad:
            grad_input = grad_input / (kH * kW)
        else:
            input_ones = torch.ones(input_shape, dtype=grad.dtype, device=grad.device)
            norm = F.avg_pool2d(
                input_ones, kernel_size, stride,
                padding, ceil_mode, count_include_pad
            )
            norm = F.conv_transpose2d(
                norm, weight, stride=stride, padding=padding, groups=channels
            )
            grad_input = grad_input / (norm + 1e-8)

        return grad_input, None, None, None, None, None


def avg_pool2d(
    input, kernel_size, stride=None,
    padding=0, ceil_mode=False, count_include_pad=True
):
    return function(_avg_pool2d)(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad
    )



class _avg_pool3d(Policy):
    def forward(
        self, input, kernel_size, stride=None,
        padding=0, ceil_mode=False, count_include_pad=True
    ):
        stride = stride or kernel_size
        self.ctx.save(
            input.shape, kernel_size, stride,
            padding, ceil_mode, count_include_pad
        )
        return F.avg_pool3d(
            input, kernel_size, stride,
            padding, ceil_mode, count_include_pad
        )

    def backward(self, grad):
        input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad = self.ctx.release
        batch_size, channels, D, H, W = input_shape

        kD, kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        weight = torch.ones((channels, 1, kD, kH, kW),
                            dtype=grad.dtype, device=grad.device)

        grad_input = F.conv_transpose3d(
            grad, weight, stride=stride, padding=padding, groups=channels
        )

        if count_include_pad:
            grad_input = grad_input / (kD * kH * kW)
        else:
            input_ones = torch.ones(input_shape, dtype=grad.dtype, device=grad.device)
            norm = F.avg_pool3d(
                input_ones, kernel_size, stride,
                padding, ceil_mode, count_include_pad
            )
            norm = F.conv_transpose3d(
                norm, weight, stride=stride, padding=padding, groups=channels
            )
            grad_input = grad_input / (norm + 1e-8)

        return grad_input, None, None, None, None, None


def avg_pool3d(
    input, kernel_size, stride=None,
    padding=0, ceil_mode=False, count_include_pad=True
):
    return function(_avg_pool3d)(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
