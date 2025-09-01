from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Tracelet, custom_grad

from neo._torch.lite_tensor import LiteTensor
from typing import Optional, Union, Tuple

import torch
from torch.nn.grad import (
    conv1d_input, conv1d_weight,
    conv2d_input, conv2d_weight,
    conv3d_input, conv3d_weight,
)

@custom_grad
def _conv1d(
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

    _out = torch.nn.functional.conv1d(
        input=input.data,
        weight=weight.data,
        bias= None if bias is None else bias.data,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
        )
    
    out = LiteTensor(_out)

    def conv1d_backward(grad):
        grad_input = conv1d_input(
            input.data.shape, weight.data, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_weight = conv1d_weight(
            input.data, weight.data.shape, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_bias = grad.sum(dim=(0, 2)) if bias is not None else None

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )
    return out, (input, weight, bias), conv1d_backward



@custom_grad
def _conv2d(
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

    _out = torch.nn.functional.conv2d(
        input=input.data,
        weight=weight.data,
        bias=None if bias is None else bias.data,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
        )
    
    out = LiteTensor(_out)

    def conv2d_backward(grad):
        grad_input = conv2d_input(
            input.data.shape, weight.data, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_weight = conv2d_weight(
            input.data, weight.data.shape, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_bias = grad.sum(dim=(0, 2, 3)) if bias is not None else None

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )
    return out, (input, weight, bias), conv2d_backward



@custom_grad
def _conv3d(
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

    _out = torch.nn.functional.conv3d(
        input=input.data,
        weight=weight.data,
        bias=None if bias is None else bias.data,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
        )
    
    out = LiteTensor(_out)

    def conv3d_backward( grad):
        grad_input = conv3d_input(
            input.data.shape, weight.data, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_weight = conv3d_weight(
            input.data, weight.data.shape, grad,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )

        grad_bias = grad.sum(dim=(0, 2, 3, 4)) if bias is not None else None

        return (
            grad_input, grad_weight, grad_bias,
            None, None, None, None
        )
    
    return out, (input, weight, bias), conv3d_backward


def conv1d(
    input: LiteTensor,
    weight: LiteTensor,
    bias: Optional[LiteTensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
    ):
    return _conv1d(
        input = input,
        weight = weight,
        bias  = bias,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups
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
    return _conv2d(
        input = input,
        weight = weight,
        bias  = bias,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups
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
    return _conv3d(
        input = input,
        weight = weight,
        bias  = bias,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups
    )



__all__ = ['conv1d', 'conv2d', 'conv3d']
