from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
from ._helper import _dtype, _device

def Lite(data, dtype='', device=''):
    return LiteTensor(data=data, d_type=dtype, device=device)


# === Tensor Constructors ===

def full(shape, fill_value, dtype='', device=''):
    return Lite(neolib.full(
        shape, fill_value,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def ones(shape, dtype='', device=''):
    return Lite(neolib.ones(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def zeros(shape, dtype='', device=''):
    return Lite(neolib.zeros(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def arange(start, end=None, step=1, dtype='', device=''):
    dtype = _dtype(dtype) if dtype else None
    device = _device(device) if device else None

    if end is None:
        return Lite(neolib.arange(0, start, step, dtype=dtype, device=device))
    else:
        return Lite(neolib.arange(start, end, step, dtype=dtype, device=device))


def empty(shape, dtype='', device=''):
    return Lite(neolib.empty(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def rand(shape, dtype='', device=''):
    return Lite(neolib.rand(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def randn(shape, dtype='', device=''):
    return Lite(neolib.randn(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))


# === _like Constructors ===

def ones_like(x, dtype='', device=''):
    return Lite(neolib.ones_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def zeros_like(x, dtype='', device=''):
    return Lite(neolib.zeros_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def empty_like(x, dtype='', device=''):
    return Lite(neolib.empty_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))


# === Shape/View/Manipulation Ops ===

def reshape(x, *shape):
    return Lite(x.data.reshape(*shape))

def permute(x, *dims):
    return Lite(x.data.permute(*dims))

def transpose(x, dim0, dim1):
    return Lite(x.data.transpose(dim0, dim1))

def squeeze(x, dim=None):
    return Lite(x.data.squeeze(dim)) if dim is not None else Lite(x.data.squeeze())

def unsqueeze(x, dim):
    return Lite(x.data.unsqueeze(dim))

def flatten(x, start_dim=0, end_dim=-1):
    return Lite(x.data.flatten(start_dim, end_dim))

def view(x, shape):
    return Lite(x.data.view(*shape))

def expand(x, shape):
    return Lite(x.data.expand(*shape))

def repeat(x, repeats):
    return Lite(x.data.repeat(*repeats))

def cat(tensors, dim=0):
    return Lite(neolib.cat([t.data for t in tensors], dim=dim))

def stack(tensors, dim=0):
    return Lite(neolib.stack([t.data for t in tensors], dim=dim))

def argmax(x, dim=None, keepdim=False):
    return Lite(neolib.argmax(x.data, dim=dim, keepdim=keepdim))

def argmin(x, dim=None, keepdim=False):
    return Lite(neolib.argmin(x.data, dim=dim, keepdim=keepdim))
