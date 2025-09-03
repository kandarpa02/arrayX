from nexnet.functions import function
from nexnet._src.autograd.FUNCTION_REGISTER import Policy, custom_grad
from nexnet._torch.lite_tensor import LiteTensor

import torch
import torch.nn.functional as F

__all__ = [
    'max_pool1d', 'max_pool2d', 'max_pool3d',
    'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
]


# ---------- MAX POOLING ----------

@custom_grad
def _max_pool1d(input, *, kernel_size, stride=None, padding, dilation, ceil_mode=False):
    stride = stride or kernel_size
    output, indices = F.max_pool1d(
        input.data, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation,
        ceil_mode=ceil_mode, return_indices=True
    )
    out = LiteTensor(output)

    def backward(grad,
                 indices=indices,
                 out_shape=input.data.shape,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding):
        grad_in = F.max_unpool1d(grad, indices, kernel_size, stride, padding, output_size=out_shape)
        return (grad_in,)

    return out, (input,), backward


@custom_grad
def _max_pool2d(input, *, kernel_size, stride=None, padding, dilation, ceil_mode=False):
    stride = stride or kernel_size
    output, indices = F.max_pool2d(
        input.data, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation,
        ceil_mode=ceil_mode, return_indices=True
    )
    out = LiteTensor(output)

    def backward(grad,
                 indices=indices,
                 out_shape=input.data.shape,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding):
        grad_in = F.max_unpool2d(grad, indices, kernel_size, stride, padding, output_size=out_shape)
        return (grad_in,)

    return out, (input,), backward


@custom_grad
def _max_pool3d(input, *, kernel_size, stride=None, padding, dilation, ceil_mode=False):
    stride = stride or kernel_size
    output, indices = F.max_pool3d(
        input.data, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation,
        ceil_mode=ceil_mode, return_indices=True
    )
    out = LiteTensor(output)

    def backward(grad,
                 indices=indices,
                 out_shape=input.data.shape,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding):
        grad_in = F.max_unpool3d(grad, indices, kernel_size, stride, padding, output_size=out_shape)
        return (grad_in,)

    return out, (input,), backward


# ---------- AVG POOLING ----------

@custom_grad
def _avg_pool1d(input, *, kernel_size, stride=None, padding, ceil_mode=False, count_include_pad=True):
    stride = stride or kernel_size
    output = F.avg_pool1d(
        input.data, kernel_size=kernel_size, stride=stride,
        padding=padding, ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
    out = LiteTensor(output)

    def backward(grad,
                 in_shape=input.data.shape,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding,
                 ceil_mode=ceil_mode,
                 count_include_pad=count_include_pad):
        B, C, L = in_shape
        weight = torch.ones((C, 1, kernel_size), dtype=grad.dtype, device=grad.device)
        grad_in = F.conv_transpose1d(grad, weight, stride=stride, padding=padding, groups=C)
        if count_include_pad:
            grad_in = grad_in / kernel_size
        else:
            ones = torch.ones(in_shape, dtype=grad.dtype, device=grad.device)
            norm = F.avg_pool1d(ones, kernel_size=kernel_size, stride=stride,
                                padding=padding, ceil_mode=ceil_mode,
                                count_include_pad=count_include_pad)
            norm = F.conv_transpose1d(norm, weight, stride=stride, padding=padding, groups=C)
            grad_in = grad_in / (norm + 1e-8)
        return (grad_in,)

    return out, (input,), backward


@custom_grad
def _avg_pool2d(input, *, kernel_size, stride=None, padding, ceil_mode=False, count_include_pad=True):
    stride = stride or kernel_size
    output = F.avg_pool2d(
        input.data, kernel_size=kernel_size, stride=stride,
        padding=padding, ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
    out = LiteTensor(output)

    def backward(grad,
                 in_shape=input.data.shape,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding,
                 ceil_mode=ceil_mode,
                 count_include_pad=count_include_pad):
        B, C, H, W = in_shape
        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        weight = torch.ones((C, 1, kH, kW), dtype=grad.dtype, device=grad.device)
        grad_in = F.conv_transpose2d(grad, weight, stride=stride, padding=padding, groups=C)
        if count_include_pad:
            grad_in = grad_in / (kH * kW)
        else:
            ones = torch.ones(in_shape, dtype=grad.dtype, device=grad.device)
            norm = F.avg_pool2d(ones, kernel_size=kernel_size, stride=stride,
                                padding=padding, ceil_mode=ceil_mode,
                                count_include_pad=count_include_pad)
            norm = F.conv_transpose2d(norm, weight, stride=stride, padding=padding, groups=C)
            grad_in = grad_in / (norm + 1e-8)
        return (grad_in,)

    return out, (input,), backward


@custom_grad
def _avg_pool3d(input, *, kernel_size, stride=None, padding, ceil_mode=False, count_include_pad=True):
    stride = stride or kernel_size
    output = F.avg_pool3d(
        input.data, kernel_size=kernel_size, stride=stride,
        padding=padding, ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
    out = LiteTensor(output)

    def backward(grad,
                 in_shape=input.data.shape,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding,
                 ceil_mode=ceil_mode,
                 count_include_pad=count_include_pad):
        B, C, D, H, W = in_shape
        kD, kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        weight = torch.ones((C, 1, kD, kH, kW), dtype=grad.dtype, device=grad.device)
        grad_in = F.conv_transpose3d(grad, weight, stride=stride, padding=padding, groups=C)
        if count_include_pad:
            grad_in = grad_in / (kD * kH * kW)
        else:
            ones = torch.ones(in_shape, dtype=grad.dtype, device=grad.device)
            norm = F.avg_pool3d(ones, kernel_size=kernel_size, stride=stride,
                                padding=padding, ceil_mode=ceil_mode,
                                count_include_pad=count_include_pad)
            norm = F.conv_transpose3d(norm, weight, stride=stride, padding=padding, groups=C)
            grad_in = grad_in / (norm + 1e-8)
        return (grad_in,)

    return out, (input,), backward


def max_pool1d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False
):
    kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    stride = (stride,) if isinstance(stride, int) else stride
    padding = (padding,) if isinstance(padding, int) else padding
    dilation = (dilation,) if isinstance(dilation, int) else dilation

    return _max_pool1d(
        input,
        kernel_size=kernel_size,
        stride=stride or kernel_size,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def max_pool2d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False
):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    return _max_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride or kernel_size,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def max_pool3d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False
):
    kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
    stride = (stride,) * 3 if isinstance(stride, int) else stride
    padding = (padding,) * 3 if isinstance(padding, int) else padding
    dilation = (dilation,) * 3 if isinstance(dilation, int) else dilation

    return _max_pool3d(
        input,
        kernel_size=kernel_size,
        stride=stride or kernel_size,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def avg_pool1d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True
):
    kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    stride = (stride,) if isinstance(stride, int) else stride
    padding = (padding,) if isinstance(padding, int) else padding

    return _avg_pool1d(
        input,
        kernel_size=kernel_size,
        stride=stride or kernel_size,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )


def avg_pool2d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True
):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    return _avg_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride or kernel_size,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )


def avg_pool3d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True
):
    kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
    stride = (stride,) * 3 if isinstance(stride, int) else stride
    padding = (padding,) * 3 if isinstance(padding, int) else padding

    return _avg_pool3d(
        input,
        kernel_size=kernel_size,
        stride=stride or kernel_size,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
