# ============================================================================================================
# LiteTensor: "Lite" wrapper for Torch tensors, NeoNet edition, FAST MODE
# 
# WHY THIS EXISTS:
# ----------------
# - Raw torch.Tensor is awesome, but we want:
#     * overloaded math operators for NeoNet
#     * prettier prints than <torch.Tensor>
#     * memory-light wrapper (d_type, device)
#     * future room for crazy optimizations
# 
# FAST DESIGN NOTES:
# -----------------
# - No per-op device checks! You MUST ensure all LiteTensors are already on the correct device.
# - Device promotion only happens at creation / explicit .to()/.cuda()/.cpu()
# - This mimics JAX/TF style: minimal Python overhead in inner loops.
# - Python overhead is now basically 0 for math ops.
# - Safe_input and _device_hierarchy are still available but optional.
# 
# WARNING:
# --------
# - LiteTensor is NOT torch.Tensor.
# - numpy() always copies to CPU.
# - You must manage device manually if doing custom loops.
# - Misaligned devices will cause CUDA runtime errors.
# 
# Author: Kandarpa Sarkar (c) 2025
# License: MIT
# ============================================================================================================

from typing import Any, Callable
from dataclasses import dataclass
from torch import Tensor
import torch
import numpy as np
from nexnet._torch import neolib

# Helpers
def _auto_device():
    """Pick CUDA if available, else CPU. Only called on tensor creation."""
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

def _device(arg):
    """Ensure argument is a torch.device"""
    if arg is None: return None
    if isinstance(arg, torch.device): return arg
    if isinstance(arg, str): return torch.device(arg)
    raise TypeError(f"Invalid device: {arg}")

def _dtype(arg):
    """Ensure argument is torch.dtype"""
    if arg is None: return None
    if isinstance(arg, torch.dtype): return arg
    if isinstance(arg, str):
        try: return getattr(torch, arg)
        except AttributeError: raise TypeError(f"Invalid dtype string: '{arg}'")
    raise TypeError(f"Invalid dtype: {arg}")

def _neo_dtype(arg):
    """Convert torch dtype to string representation without torch. prefix"""
    arg = str(arg)
    if 'torch.' in arg:
        return arg.removeprefix('torch.')

# LiteTensor Class
class LiteTensor:
    """LiteTensor: thin, fast wrapper around torch.Tensor for NeoNet
    
    Stores:
        data : torch.Tensor
        d_type : dtype
        device : cpu/cuda
    
    Provides:
        - operator overloads: +,-,*,/,**, @, neg, comparisons
        - NeoNet-friendly __repr__ printing
        - convenience methods: cpu(), cuda(), numpy(), ones_like(), zeros_like()
    
    WARNING:
        - No per-op device promotion, you must manage devices manually.
        - .numpy() always copies to CPU.
        - Designed for fast inner loops; treat like an immutable device-bound tensor.
    """

    __slots__ = ('data', 'd_type', 'device')

    def __init__(self, data, d_type='', device=''):
        """Initialize LiteTensor; device/dtype is applied at creation only."""
        if not isinstance(data, Tensor):
            dtype = _dtype(d_type) if d_type else None
            dev = _device(device) if device else _auto_device()
            self.data = torch.as_tensor(data, dtype=dtype, device=dev).detach()
        else:
            self.data = data.detach()
            dtype = _dtype(d_type) if d_type else None
            dev = _device(device) if device else None
            if dtype or dev:
                self.data = self.data.to(dtype=dtype, device=dev)
        self.d_type = self.data.dtype
        self.device = self.data.device

    # Op creation
    def binary_op(self, other, fn:Callable, **kwargs):
        out = LiteTensor(
            data = fn(self.data, other.data, **kwargs),
        )
        return out
    
    def nary_op(self, args, fn: Callable, **kwargs):
        out = LiteTensor(
            data = fn(self.data, *(arg.data for arg in args), **kwargs)
        )
        return out

    def unary_op(self, fn:Callable, **kwargs):
        out = LiteTensor(
            data = fn(self.data, **kwargs)
        )
        return out

    # Properties
    @property
    def _t(self): return self.data
    @property
    def dtype(self): return self.d_type
    @property
    def ndtype(self): return _neo_dtype(self.dtype)
    @property
    def shape(self):
        try: return self.data.cpu().numpy().shape
        except TypeError: return self.to(torch.float32).cpu().numpy().shape
    @property
    def size(self): return self.data.cpu().numpy().size
    @property
    def T(self): return LiteTensor(self.data.T)

    # Conversion
    def to(self, *args, **kwargs):
        """Move to device/dtype or copy from another tensor"""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, LiteTensor): return LiteTensor(self.data.to(arg.data.device, arg.data.dtype))
            elif isinstance(arg, Tensor): return LiteTensor(self.data.to(arg.device, arg.dtype))
            elif isinstance(arg, (str, torch.device)): return LiteTensor(self.data.to(device=_device(arg)))
            elif isinstance(arg, (torch.dtype, str)): return LiteTensor(self.data.to(dtype=_dtype(arg)))
            else: raise TypeError(f"Unsupported type for .to(): {type(arg)}")
        elif len(args) == 2:
            device = _device(args[0])
            dtype = _dtype(args[1])
            return LiteTensor(self.data.to(device=device, dtype=dtype))
        elif not args and kwargs:
            device = _device(kwargs.get("device")) if "device" in kwargs else None
            dtype = _dtype(kwargs.get("dtype")) if "dtype" in kwargs else None
            return LiteTensor(self.data.to(device=device, dtype=dtype))
        else:
            raise TypeError("Invalid arguments passed to .to()")

    def to_(self, *args, **kwargs):
        """In-place device/dtype conversion (returns self)."""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, LiteTensor):
                self.data = self.data.to(arg.data.device, arg.data.dtype)
                return self
            elif isinstance(arg, torch.Tensor):
                self.data = self.data.to(arg.device, arg.dtype)
                return self
            elif isinstance(arg, (str, torch.device)):
                self.data = self.data.to(device=_device(arg))
                return self
            elif isinstance(arg, (torch.dtype, str)):
                self.data = self.data.to(dtype=_dtype(arg))
                return self
            else:
                raise TypeError(f"Unsupported type for .to_(): {type(arg)}")

        elif len(args) == 2:
            device = _device(args[0])
            dtype = _dtype(args[1])
            self.data = self.data.to(device=device, dtype=dtype)
            return self

        elif not args and kwargs:
            device = _device(kwargs.get("device")) if "device" in kwargs else None
            dtype = _dtype(kwargs.get("dtype")) if "dtype" in kwargs else None
            self.data = self.data.to(device=device, dtype=dtype)
            return self

        else:
            raise TypeError("Invalid arguments passed to .to_()")

    def cuda(self, device=None): return LiteTensor(self.data.cuda(device=device))
    def cpu(self): return LiteTensor(self.data.cpu())
    def numpy(self): return self.data.detach().cpu().numpy()

    # Magic Methods / Operators
    def __repr__(self):
        shape = self.data.to('cpu').detach().numpy().shape
        dtype = _neo_dtype(self.data.dtype)
        device = self.data.device
        arr_str = np.array2string(self.numpy(), precision=4, suppress_small=False,
                                  threshold=6, edgeitems=3, max_line_width=80, separator=', ',
                                  prefix=(' ' * 7))
        rpr =  f"Tensor(<shape={shape}, dtype={dtype}, device={device}>\n"
        rpr += f"       {arr_str})\n"
        return rpr
    
    __str__ = __repr__

    def __len__(self): return 0 if self.data.dim() == 0 else len(self.data)
    def __getitem__(self, index): return self.unary_op(lambda x:x[index])
    def __eq__(self, other): return isinstance(other, LiteTensor) and self.data == other.data
    def __ne__(self, other): return not self.__eq__(other)
    def __lt__(self, other): return self.data < other.data if isinstance(other, LiteTensor) else NotImplemented
    def __le__(self, other): return self.data <= other.data if isinstance(other, LiteTensor) else NotImplemented
    def __gt__(self, other): return self.data > other.data if isinstance(other, LiteTensor) else NotImplemented
    def __ge__(self, other): return self.data >= other.data if isinstance(other, LiteTensor) else NotImplemented
    def __hash__(self): return id(self)
    def __neg__(self): 
        from nexnet._torch.functions import neg
        return neg(self)

    # Fast math ops: assume correct devices, no checks
    def __add__(self, other): 
        from nexnet._torch.functions import add
        return add(self, other if isinstance(other, LiteTensor) else LiteTensor(other))
    def __sub__(self, other): 
        from nexnet._torch.functions import sub
        return sub(self, other if isinstance(other, LiteTensor) else LiteTensor(other))
    def __mul__(self, other): 
        from nexnet._torch.functions import mul
        return mul(self, other if isinstance(other, LiteTensor) else LiteTensor(other))
    def __truediv__(self, other): 
        from nexnet._torch.functions import div
        return div(self, other if isinstance(other, LiteTensor) else LiteTensor(other))
    def __pow__(self, other): 
        from nexnet._torch.functions import power
        return power(self, other if isinstance(other, LiteTensor) else LiteTensor(other))
    def __matmul__(self, other): 
        from nexnet._torch.functions import matmul
        return matmul(self, other if isinstance(other, LiteTensor) else LiteTensor(other))
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rdiv__(self, other): return self.__truediv__(other)
    def __rpow__(self, other): return self.__pow__(other)

    # Convenience
    def ones_like(self): return LiteTensor(neolib.ones_like(self.data))
    def zeros_like(self): return LiteTensor(neolib.zeros_like(self.data))
    def sum(self, dim=None, keepdim=False): 
        from nexnet._torch.functions import sum
        return sum(self, dim, keepdim)
    def relu(self): from nexnet._src.nn._activations import relu; return relu(self)
    def tanh(self): from nexnet._src.nn._activations import tanh; return tanh(self)
    def reshape(self, shape):
        from nexnet._torch.user_functions import reshape as _reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
        return _reshape(self, shape)
