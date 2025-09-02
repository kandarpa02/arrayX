from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
from ._helper import _dtype, _device
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Tracelet
import torch
from typing import Any
import warnings

def lite(data, dtype='', device=''):
    warnings.warn(
        "'neo.lite' is deprecated. Please use neo.Tensor instead.",
        DeprecationWarning,
        stacklevel=2 
    )
    return Tensor(data, dtype, device) 


def Tensor(data, dtype='', device=''):
    data = data.data if isinstance(data, LiteTensor) else data
    return LiteTensor(data=data, d_type=dtype, device=device)


# === Tensor Constructors ===

def full(shape, fill_value, dtype:Any=''):
    return Tensor(neolib.full(
        shape, fill_value,
        dtype=_dtype(dtype) if dtype else None
    ).numpy())

def ones(shape, dtype:Any=''):
    return Tensor(neolib.ones(
        shape,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def zeros(shape, dtype:Any=''):
    return Tensor(neolib.zeros(
        shape,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def arange(start, end=None, step=1, dtype:Any='', device:Any=''):
    dtype = _dtype(dtype) if dtype else None
    device = _device(device) if device else None

    if end is None:
        return Tensor(neolib.arange(0, start, step, dtype=dtype, device=device))
    else:
        return Tensor(neolib.arange(start, end, step, dtype=dtype, device=device))


def empty(shape, dtype:Any=''):
    return Tensor(neolib.empty(
        shape,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())


# === _like Constructors ===

def ones_like(x, dtype:Any=''):
    return Tensor(neolib.ones_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def zeros_like(x, dtype:Any=''):
    return Tensor(neolib.zeros_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())

def empty_like(x, dtype:Any=''):
    return Tensor(neolib.empty_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
    ).numpy())


def one_hot(x, num_classes):
    return Tensor(neolib.nn.functional.one_hot(
        x.data,
        num_classes=num_classes
    ).numpy())

# === Shape/View/Manipulation Ops ===
    
def permute(x, *dims):
    out = x.unary_op(lambda x: x.permute(*dims))

    def permute_backward(grad, dims=dims):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        # inverse permutation
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        return grad.permute(*inv_dims)

    with Tracelet() as t:
        t.register(out, (x,), permute_backward)

    return out

def transpose(x, dim0, dim1):
    out = x.unary_op(lambda x: x.transpose(dim0, dim1))

    def transpose_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        return grad.transpose(dim0, dim1)

    with Tracelet() as t:
        t.register(out, (x,), transpose_backward)

    return out


def squeeze(x, dim=None):
    out = x.unary_op(lambda x: x.squeeze(dim))

    def squeeze_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        if dim is None:
            # expand all singleton dims
            shape = [s if s != 1 else 1 for s in x.data.shape]
            return grad.reshape(x.data.shape)
        else:
            # expand only the specified dim
            shape = list(grad.shape)
            shape.insert(dim, 1)
            return grad.reshape(shape)

    with Tracelet() as t:
        t.register(out, (x,), squeeze_backward)

    return out


def unsqueeze(x, dim):
    out = x.unary_op(lambda x: x.unsqueeze(dim))

    def unsqueeze_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        return grad.squeeze(dim)

    with Tracelet() as t:
        t.register(out, (x,), unsqueeze_backward)

    return out


def flatten(x, start_dim=0, end_dim=-1):
    out = x.unary_op(lambda x: x.flatten(start_dim, end_dim))

    def flatten_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        return grad.reshape(x.data.shape)

    with Tracelet() as t:
        t.register(out, (x,), flatten_backward)

    return out


def view(x, shape):
    out = x.unary_op(lambda x: x.view(shape))

    def view_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        return grad.view(x.data.shape)

    with Tracelet() as t:
        t.register(out, (x,), view_backward)

    return out


def expand(x, shape):
    out = x.unary_op(lambda x: x.expand(shape))

    def expand_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        # sum over broadcasted dims
        while len(grad.shape) > len(x.data.shape):
            grad = grad.sum(dim=0)
        for i, (s_orig, s_new) in enumerate(zip(x.data.shape, grad.shape)):
            if s_orig == 1 and s_new > 1:
                grad = grad.sum(dim=i, keepdim=True)
        return grad

    with Tracelet() as t:
        t.register(out, (x,), expand_backward)

    return out


def repeat(x, repeats):
    out = x.unary_op(lambda x: x.repeat(repeats))

    def repeat_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        # sum over repeated blocks
        orig_shape = x.data.shape
        repeated_shape = out.data.shape
        # reshape grad to split repeated blocks
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
    out = tensors[0].unary_op(lambda _: torch.cat([t.data for t in tensors], dim=dim))

    def cat_backward(grad):
        grad = torch.as_tensor(grad, dtype=tensors[0].data.dtype, device=tensors[0].data.device)
        splits = [t.data.shape[dim] for t in tensors]
        return tuple(torch.split(grad, splits, dim=dim))

    with Tracelet() as t:
        t.register(out, tensors, cat_backward)

    return out


def stack(tensors, dim=0):
    out = tensors[0].unary_op(lambda _: torch.stack([t.data for t in tensors], dim=dim))

    def stack_backward(grad):
        grad = torch.as_tensor(grad, dtype=tensors[0].data.dtype, device=tensors[0].data.device)
        return tuple(grad.unbind(dim=dim))

    with Tracelet() as t:
        t.register(out, tensors, stack_backward)

    return out


def argmax(x, dim=None, keepdim=False):
    out = x.unary_op(lambda x: x.argmax(dim=dim, keepdim=keepdim))

    def argmax_backward(grad):
        # argmax is not differentiable; return zeros like PyTorch
        return torch.zeros_like(x.data)

    with Tracelet() as t:
        t.register(out, (x,), argmax_backward)

    return out


def argmin(x, dim=None, keepdim=False):
    out = x.unary_op(lambda x: x.argmin(dim=dim, keepdim=keepdim))

    def argmin_backward(grad):
        # argmin is not differentiable; return zeros like PyTorch
        return torch.zeros_like(x.data)

    with Tracelet() as t:
        t.register(out, (x,), argmin_backward)

    return out



# USER FACING FUNCTIONS, CONTAINING FORWARD AND BACKWARD SUPPORT
# THE POLICIES ARE WRITTEN ABOVE................................

def reshape(x, shape):
    out = x.unary_op(lambda x: x.reshape(shape))

    def reshape_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        return grad.reshape(x.data.shape)

    with Tracelet() as t:
        t.register(out, (x,), reshape_backward)

    return out


def maximum(x, y):
    out = x.binary_op(y, lambda a, b: torch.maximum(a, b))

    def maximum_backward(grad, x=x.data, y=y.data):
        grad = torch.as_tensor(grad, dtype=x.dtype, device=x.device)
        mask_x = (x >= y)
        mask_y = (x < y)

        grad_x = grad * mask_x.to(x.dtype)
        grad_y = grad * mask_y.to(y.dtype)

        return grad_x, grad_y

    with Tracelet() as t:
        t.register(out, (x, y), maximum_backward)

    return out



def clamp(x, min=None, max=None):
    out = x.unary_op(lambda x: x.clamp(min=min, max=max))

    def clamp_backward(grad):
        grad = torch.as_tensor(grad, dtype=x.data.dtype, device=x.data.device)
        mask = torch.ones_like(x.data, dtype=x.data.dtype)
        if min is not None:
            mask = mask * (x.data >= min)
        if max is not None:
            mask = mask * (x.data <= max)
        return grad * mask

    with Tracelet() as t:
        t.register(out, (x,), clamp_backward)

    return out
