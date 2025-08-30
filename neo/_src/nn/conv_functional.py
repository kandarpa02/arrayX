from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Tracelet

from neo._torch.lite_tensor import LiteTensor
from typing import Optional, Union, Tuple

import torch
from torch.nn.grad import (
    conv1d_input, conv1d_weight,
    conv2d_input, conv2d_weight,
    conv3d_input, conv3d_weight,
)

__all__ = ['conv1d', 'conv2d', 'conv3d']


def conv1d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
    ):
    
    def to_tuple(x):
        return (x,) if isinstance(x, int) else x

    stride = to_tuple(stride)
    padding = to_tuple(padding)
    dilation = to_tuple(dilation)

    out = torch.nn.functional.conv1d(input, weight, bias, stride, padding, dilation, groups)

    def conv1d_backward(grad):
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
    with Tracelet() as t:
        t.register(out, (input, weight, bias), backward=conv1d_backward)
    return out



def conv2d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
    ):
       
    def to_tuple(x):
        return (x, x) if isinstance(x, int) else x

    stride = to_tuple(stride)
    padding = to_tuple(padding)
    dilation = to_tuple(dilation)

    out = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)


    def conv2d_backward(grad):
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
    with Tracelet() as t:
        t.register(out, (input, weight, bias), backward=conv2d_backward)
    return out



def conv3d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
    ):
        
    def to_tuple(x):
        return (x, x, x) if isinstance(x, int) else x

    stride = to_tuple(stride)
    padding = to_tuple(padding)
    dilation = to_tuple(dilation)

    out = torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups)


    def conv3d_backward( grad):
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
    
    with Tracelet() as t:
        t.register(out, (input, weight, bias), backward=conv3d_backward)
    return out

    