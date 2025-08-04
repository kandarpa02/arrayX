from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
from ._helper import _dtype, _device

def data(data, dtype='', device=''):
    data = data.data if isinstance(data, LiteTensor) else data
    return LiteTensor(data=data, d_type=dtype, device=device)


# === Tensor Constructors ===

def full(shape, fill_value, dtype='', device=''):
    return data(neolib.full(
        shape, fill_value,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def ones(shape, dtype='', device=''):
    return data(neolib.ones(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def zeros(shape, dtype='', device=''):
    return data(neolib.zeros(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def arange(start, end=None, step=1, dtype='', device=''):
    dtype = _dtype(dtype) if dtype else None
    device = _device(device) if device else None

    if end is None:
        return data(neolib.arange(0, start, step, dtype=dtype, device=device))
    else:
        return data(neolib.arange(start, end, step, dtype=dtype, device=device))


def empty(shape, dtype='', device=''):
    return data(neolib.empty(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def rand(shape, dtype='', device=''):
    return data(neolib.rand(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def randn(shape, dtype='', device=''):
    return data(neolib.randn(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))


# === _like Constructors ===

def ones_like(x, dtype='', device=''):
    return data(neolib.ones_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def zeros_like(x, dtype='', device=''):
    return data(neolib.zeros_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def empty_like(x, dtype='', device=''):
    return data(neolib.empty_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))


# === Shape/View/Manipulation Ops ===

def reshape(x, *shape):
    return data(x.data.reshape(*shape))

def permute(x, *dims):
    return data(x.data.permute(*dims))

def transpose(x, dim0, dim1):
    return data(x.data.transpose(dim0, dim1))

def squeeze(x, dim=None):
    return data(x.data.squeeze(dim)) if dim is not None else data(x.data.squeeze())

def unsqueeze(x, dim):
    return data(x.data.unsqueeze(dim))

def flatten(x, start_dim=0, end_dim=-1):
    return data(x.data.flatten(start_dim, end_dim))

def view(x, shape):
    return data(x.data.view(*shape))

def expand(x, shape):
    return data(x.data.expand(*shape))

def repeat(x, repeats):
    return data(x.data.repeat(*repeats))

def cat(tensors, dim=0):
    return data(neolib.cat([t.data for t in tensors], dim=dim))

def stack(tensors, dim=0):
    return data(neolib.stack([t.data for t in tensors], dim=dim))

def argmax(x, dim=None, keepdim=False):
    return data(neolib.argmax(x.data, dim=dim, keepdim=keepdim))

def argmin(x, dim=None, keepdim=False):
    return data(neolib.argmin(x.data, dim=dim, keepdim=keepdim))
