import torch
from ._helper import _dtype, _device, _auto_device
from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch
from typing import Any
import warnings


def tensor(data, dtype:torch.device|str|Any=None, device:torch.device|str|Any=None):
    dtype = _dtype(dtype)
    device = _auto_device() if device is None else _device(device)
    return torch.as_tensor(data, dtype=dtype, device=device)

# === tensor Constructors ===

def full(shape, fill_value, dtype:Any=''):
    return tensor(torch.full(
        shape, fill_value,
        dtype=_dtype(dtype) if dtype else None
    ).numpy())

def ones(shape, dtype:Any=''):
    return tensor(torch.ones(
        shape,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def zeros(shape, dtype:Any=''):
    return tensor(torch.zeros(
        shape,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def arange(start, end=None, step=1, dtype:Any='', device:Any=''):
    dtype = _dtype(dtype) if dtype else None
    device = _device(device) if device else None

    if end is None:
        return tensor(torch.arange(0, start, step, dtype=dtype, device=device))
    else:
        return tensor(torch.arange(start, end, step, dtype=dtype, device=device))


def empty(shape, dtype:Any=''):
    return tensor(torch.empty(
        shape,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())


# === _like Constructors ===

def ones_like(x, dtype:Any=''):
    return tensor(torch.ones_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def zeros_like(x, dtype:Any=''):
    return tensor(torch.zeros_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def empty_like(x, dtype:Any=''):
    return tensor(torch.empty_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())


def one_hot(x, num_classes):
    return tensor(torch.nn.functional.one_hot(
        x.data,
        num_classes=num_classes
    ).cpu().numpy())

# === Shape/View/Manipulation Ops ===
    
from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch


def permute(x, *dims):
    out = x.permute(*dims)

    def permute_backward(grad):
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        return grad.permute(*inv_dims)

    with Tracelet() as t:
        t.register(out, (x,), permute_backward)

    return out


def transpose(x, dim0, dim1):
    out = x.transpose(dim0, dim1)

    def transpose_backward(grad):
        return grad.transpose(dim0, dim1)

    with Tracelet() as t:
        t.register(out, (x,), transpose_backward)

    return out


def squeeze(x, dim=None):
    out = x.squeeze(dim)

    def squeeze_backward(grad):
        if dim is None:
            return grad.reshape(x.shape)
        else:
            shape = list(grad.shape)
            shape.insert(dim, 1)
            return grad.reshape(shape)

    with Tracelet() as t:
        t.register(out, (x,), squeeze_backward)

    return out


def unsqueeze(x, dim):
    out = x.unsqueeze(dim)

    def unsqueeze_backward(grad):
        return grad.squeeze(dim)

    with Tracelet() as t:
        t.register(out, (x,), unsqueeze_backward)

    return out


def flatten(x, start_dim=0, end_dim=-1):
    out = x.flatten(start_dim, end_dim)

    def flatten_backward(grad):
        return grad.reshape(x.shape)

    with Tracelet() as t:
        t.register(out, (x,), flatten_backward)

    return out


def view(x, shape):
    out = x.view(shape)

    def view_backward(grad):
        return grad.view(x.shape)

    with Tracelet() as t:
        t.register(out, (x,), view_backward)

    return out


def expand(x, shape):
    out = x.expand(shape)

    def expand_backward(grad):
        while len(grad.shape) > len(x.shape):
            grad = grad.sum(dim=0)
        for i, (s_orig, s_new) in enumerate(zip(x.shape, grad.shape)):
            if s_orig == 1 and s_new > 1:
                grad = grad.sum(dim=i, keepdim=True)
        return grad

    with Tracelet() as t:
        t.register(out, (x,), expand_backward)

    return out


def repeat(x, repeats):
    out = x.repeat(repeats)

    def repeat_backward(grad):
        orig_shape = x.shape
        new_shape = []
        for s_orig, r in zip(orig_shape, repeats):
            new_shape.extend([r, s_orig])
        grad = grad.reshape(new_shape)
        for i in range(0, len(new_shape), 2):
            grad = grad.sum(dim=i)
        return grad

    with Tracelet() as t:
        t.register(out, (x,), repeat_backward)

    return out


def cat(tensors, dim=0):
    out = torch.cat(tensors, dim=dim)

    def cat_backward(grad):
        splits = [t.shape[dim] for t in tensors]
        return tuple(torch.split(grad, splits, dim=dim))

    with Tracelet() as t:
        t.register(out, tensors, cat_backward)

    return out


def stack(tensors, dim=0):
    out = torch.stack(tensors, dim=dim)

    def stack_backward(grad):
        return tuple(grad.unbind(dim=dim))

    with Tracelet() as t:
        t.register(out, tensors, stack_backward)

    return out


def argmax(x, dim=None, keepdim=False):
    out = x.argmax(dim=dim, keepdim=keepdim)

    def argmax_backward(grad):
        return torch.zeros_like(x)

    with Tracelet() as t:
        t.register(out, (x,), argmax_backward)

    return out


def argmin(x, dim=None, keepdim=False):
    out = x.argmin(dim=dim, keepdim=keepdim)

    def argmin_backward(grad):
        return torch.zeros_like(x)

    with Tracelet() as t:
        t.register(out, (x,), argmin_backward)

    return out


def reshape(x, shape):
    out = x.reshape(shape)

    def reshape_backward(grad):
        return grad.reshape(x.shape)

    with Tracelet() as t:
        t.register(out, (x,), reshape_backward)

    return out


def maximum(x, y):
    out = torch.maximum(x, y)

    def maximum_backward(grad):
        mask_x = (x >= y)
        mask_y = (x < y)
        grad_x = grad * mask_x.to(x.dtype)
        grad_y = grad * mask_y.to(y.dtype)
        return grad_x, grad_y

    with Tracelet() as t:
        t.register(out, (x, y), maximum_backward)

    return out


def clamp(x, min=None, max=None):
    out = x.clamp(min=min, max=max)

    def clamp_backward(grad):
        mask = torch.ones_like(x, dtype=x.dtype)
        if min is not None:
            mask = mask * (x >= min)
        if max is not None:
            mask = mask * (x <= max)
        return grad * mask

    with Tracelet() as t:
        t.register(out, (x,), clamp_backward)

    return out
