from neo._torch.lite_tensor import LiteTensor
from neo._torch import neolib
from ._helper import _dtype, _device
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo.functions import function

def lite(data, dtype='', device=''):
    data = data.data if isinstance(data, LiteTensor) else data
    return LiteTensor(data=data, d_type=dtype, device=device)


# === Tensor Constructors ===

def full(shape, fill_value, dtype='', device=''):
    return lite(neolib.full(
        shape, fill_value,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def ones(shape, dtype='', device=''):
    return lite(neolib.ones(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def zeros(shape, dtype='', device=''):
    return lite(neolib.zeros(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def arange(start, end=None, step=1, dtype='', device=''):
    dtype = _dtype(dtype) if dtype else None
    device = _device(device) if device else None

    if end is None:
        return lite(neolib.arange(0, start, step, dtype=dtype, device=device))
    else:
        return lite(neolib.arange(start, end, step, dtype=dtype, device=device))


def empty(shape, dtype='', device=''):
    return lite(neolib.empty(
        shape,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))


# === _like Constructors ===

def ones_like(x, dtype='', device=''):
    return lite(neolib.ones_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def zeros_like(x, dtype='', device=''):
    return lite(neolib.zeros_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))

def empty_like(x, dtype='', device=''):
    return lite(neolib.empty_like(
        x.data,
        dtype=_dtype(dtype) if dtype else None,
        device=_device(device) if device else None
    ))


# === Shape/View/Manipulation Ops ===

class _reshape(Policy):
    def forward(self, x, shape):
        self.ctx.save(x.shape)
        out = x.reshape(shape=shape)
        return out
    
    def backward(self, grad):
        shape, = self.ctx.release
        return grad.reshape(shape)
    
class _maximum(Policy):
    def forward(self, x, y):
        out = neolib.maximum(x, y)
        self.ctx.save(x, y, out)
        return out
    def backward(self, grad):
        x, y, out = self.ctx.release
        x_grad = (out==x) * grad
        y_grad = (out==y) * grad
        return x_grad, y_grad

@function
class _clamp(Policy):
    def forward(self, x, min=None, max=None):
        self.ctx.save(x, min, max)
        return neolib.clamp(x, min, max)

    def backward(self, grad):
        x, min, max = self.ctx.release
        mask = neolib.ones_like(x)
        if min is not None:
            mask = mask * (x > min)
        if max is not None:
            mask = mask * (x < max)
        return grad * mask
    

def permute(x, *dims):
    return lite(x.data.permute(*dims))

def transpose(x, dim0, dim1):
    return lite(x.data.transpose(dim0, dim1))

def squeeze(x, dim=None):
    return lite(x.data.squeeze(dim)) if dim is not None else lite(x.data.squeeze())

def unsqueeze(x, dim):
    return lite(x.data.unsqueeze(dim))

def flatten(x, start_dim=0, end_dim=-1):
    return lite(x.data.flatten(start_dim, end_dim))

def view(x, shape):
    return lite(x.data.view(*shape))

def expand(x, shape):
    return lite(x.data.expand(*shape))

def repeat(x, repeats):
    return lite(x.data.repeat(*repeats))

def cat(tensors, dim=0):
    return lite(neolib.cat([t.data for t in tensors], dim=dim))

def stack(tensors, dim=0):
    return lite(neolib.stack([t.data for t in tensors], dim=dim))

def argmax(x, dim=None, keepdim=False):
    return lite(neolib.argmax(x.data, dim=dim, keepdim=keepdim))

def argmin(x, dim=None, keepdim=False):
    return lite(neolib.argmin(x.data, dim=dim, keepdim=keepdim))


# USER FACING FUNCTIONS, CONTAINING FORWARD AND BACKWARD SUPPORT
# THE POLICIES ARE WRITTEN ABOVE................................

def reshape(x:LiteTensor, shape):
    return function(_reshape)(x, shape)

def maximum(x:LiteTensor, y:LiteTensor):
    return function(_maximum)(x, y)

def clamp(x:LiteTensor, min=None, max=None):
    return _clamp(x, min, max)

