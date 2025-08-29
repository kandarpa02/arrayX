# ============================================================================================================
# LiteTensor: "Lite" wrapper for Torch tensors, NeoNet edition
# 
# WHY THIS EXISTS:
# ----------------
# Because raw torch.Tensor is awesome but sometimes you want:
#   - auto device/dtype promotion (so CUDA magically happens)
#   - prettier prints than <torch.Tensor>
#   - overloaded math operators that play nicely with NeoNet functions
#   - space to later add crazy optimizations without touching user code
#
# WARNING:
# --------
# - LiteTensor is NOT a drop-in replacement for torch.Tensor
# - numpy() always copies to CPU (deal with it)
# - .to() is convenient but has edge-case bugs
# - If you try to subclass it or hack __dict__ -> segfaults might happen
# 
# DESIGN NOTES:
# -------------
# - We only store `data`, `d_type`, `device` (we hate memory overhead)
# - __slots__ = ('data','d_type','device') to save RAM and marginal speed
# - Every math op delegates to NeoNet backend (neo.functions)
# - Device promotion logic is Python-y, minor overhead unavoidable
#
# PERFORMANCE:
# ------------
# - Primary bottleneck: device promotion and Python function call overhead
# - If you’re benchmarking: blame safe_input() and _device_hierarchy()
#
# Author: Kandarpa Sarkar (c) 2025
# License: MIT
# ============================================================================================================

from typing import Any
from dataclasses import dataclass
from torch import Tensor
import torch
import numpy as np
from neo._torch import neolib
from .functions import *

# Helpers
def _auto_device():
    """Pick CUDA if available, else CPU. This is our "automatic magic"."""
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def _device_hierarchy(self, other):
    """
    Promote both operands to the same device.
    If either is on CUDA, everything goes CUDA.
    Python overhead is minimal, deal with it.
    """
    target_device = "cuda" if (
        self.data.device.type == "cuda" or 
        (hasattr(other, "data") and other.data.device.type == "cuda") or
        (isinstance(other, Tensor) and other.device.type == "cuda")
    ) else "cpu"

    # Move self
    self.data = self.data.to(target_device)
    self.device = self.data.device

    # Move other
    if isinstance(other, Tensor):
        other = other.to(target_device)
    elif hasattr(other, "data"):
        other.data = other.data.to(target_device)
        other.device = other.data.device

    return self, other


def safe_input(self, x):
    """
    Wrap inputs into LiteTensor if they are not already.
    Scalars, numpy arrays, torch tensors handled.
    Promotes devices automatically.
    """
    if not isinstance(x, LiteTensor):
        if isinstance(x, (int, float, Tensor, np.ndarray)):
            x = LiteTensor(x, d_type=self.d_type, device=self.device)
    self, other = _device_hierarchy(self, x)
    return self, other


def _device(arg):
    if arg is None:
        return None
    if isinstance(arg, torch.device):
        return arg
    if isinstance(arg, str):
        return torch.device(arg)
    raise TypeError(f"Invalid device: {arg}")


def _dtype(arg):
    if arg is None:
        return None
    if isinstance(arg, torch.dtype):
        return arg
    if isinstance(arg, str):
        try:
            return getattr(torch, arg)
        except AttributeError:
            raise TypeError(f"Invalid dtype string: '{arg}'")
    raise TypeError(f"Invalid dtype: {arg}")


def _neo_dtype(arg):
    """
    Convert torch dtype to string representation without torch. prefix.
    """
    arg = str(arg)
    if 'torch.' in arg:
        return arg.removeprefix('torch.')


# LiteTensor Class
class LiteTensor:
    """
    LiteTensor: thin wrapper around torch.Tensor for NeoNet.
    
    Stores:
      - data: torch.Tensor
      - d_type: dtype
      - device: cpu/cuda
    
    Provides:
      - operator overloads: +,-,*,/,**, @, neg, comparisons
      - device/dtype promotion
      - NeoNet-friendly __repr__ printing
      - convenience methods: cpu(), cuda(), numpy(), ones_like(), zeros_like()
    
    WARNING:
      - .numpy() always copies to CPU
      - Use with NeoNet functions for autograd
      - Do NOT assume it behaves 100% like torch.Tensor
      - .to() tries its best but edge cases exist
    """

    __slots__ = ('data', 'd_type', 'device')

    def __init__(self, data, d_type='', device=''):
        """
        Initialize LiteTensor.
        
        Parameters:
        -----------
        data : array-like, torch.Tensor, int, float, np.ndarray
        d_type : str or torch.dtype
        device : str or torch.device
        """
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

    # Properties
    @property
    def dtype(self):
        """Torch dtype"""
        return self.d_type
    
    @property
    def ndtype(self):
        """Neo-style dtype string without torch. prefix"""
        return _neo_dtype(self.dtype)

    @property
    def shape(self):
        """Returns shape as tuple"""
        try:
            shp = self.data.cpu().numpy().shape
        except TypeError:
            shp = self.to(torch.float32).cpu().numpy().shape
        return shp
    
    @property
    def size(self):
        """Number of elements"""
        return self.data.cpu().numpy().size

    @property
    def T(self):
        """Transpose (shortcut)"""
        return LiteTensor(self.data.T)

    # Conversion
    def to(self, *args, **kwargs):
        """Move to device/dtype or copy from another tensor"""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, LiteTensor):
                return LiteTensor(self.data.to(arg.data.device, arg.data.dtype))
            elif isinstance(arg, Tensor):
                return LiteTensor(self.data.to(arg.device, arg.dtype))
            elif isinstance(arg, (str, torch.device)):
                return LiteTensor(self.data.to(device=_device(arg)))
            elif isinstance(arg, (torch.dtype, str)):
                return LiteTensor(self.data.to(dtype=_dtype(arg)))
            else:
                raise TypeError(f"Unsupported type for .to(): {type(arg)}")
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

    def cuda(self, device=None):
        """Move to CUDA"""
        return LiteTensor(self.data.cuda(device=device))

    def cpu(self):
        """Move to CPU"""
        return LiteTensor(self.data.cpu())

    def numpy(self):
        """Return as NumPy array (always CPU copy)"""
        return self.data.detach().cpu().numpy()
        
    # Magic Methods / Operators
    def __repr__(self):
        shape = self.data.to('cpu').detach().numpy().shape
        dtype = _neo_dtype(self.data.dtype)
        device = self.data.device
        arr_str = np.array2string(self.numpy(), precision=4, suppress_small=True,
                                  threshold=6, edgeitems=3, max_line_width=80, separator='  ',
                                  prefix=' ' * 8)
        return f"Tensor(<shape={shape}, dtype={dtype}, device={device}>\n       {arr_str})\n"
    __str__ = __repr__

    def __len__(self):
        """Length along first dimension"""
        if self.data.dim() == 0:
            return 0
        return len(self.data)

    def __getitem__(self, index):
        """Indexing access"""
        return self.data[index]

    def __eq__(self, other):
        """Equality check"""
        return isinstance(other, LiteTensor) and self.data == other.data

    def __ne__(self, other): return not self.__eq__(other)
    def __lt__(self, other): return self.data < other.data if isinstance(other, LiteTensor) else NotImplemented
    def __le__(self, other): return self.data <= other.data if isinstance(other, LiteTensor) else NotImplemented
    def __gt__(self, other): return self.data > other.data if isinstance(other, LiteTensor) else NotImplemented
    def __ge__(self, other): return self.data >= other.data if isinstance(other, LiteTensor) else NotImplemented
    def __hash__(self): return id(self)
    def __neg__(self): return neg(self)
    def __add__(self, other): self, b = safe_input(self, other); return add(self, b)
    def __sub__(self, other): self, b = safe_input(self, other); return sub(self, b)
    def __mul__(self, other): self, b = safe_input(self, other); return mul(self, b)
    def __pow__(self, other): self, b = safe_input(self, other); return power(self, other)
    def __truediv__(self, other): self, b = safe_input(self, other); return div(self, b)
    def __matmul__(self, other): self, other = _device_hierarchy(self, other); return matmul(self, other)
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rdiv__(self, other): return self.__truediv__(other)
    def __rpow__(self, other): return self.__pow__(other)

    # Convenience
    def ones_like(self): return LiteTensor(neolib.ones_like(self.data))
    def zeros_like(self): return LiteTensor(neolib.zeros_like(self.data))
    def sum(self, dim=None, keepdim=False): return sum(self, dim, keepdim)
    def relu(self): from neo._src.nn._activations import relu; return relu(self)
    def tanh(self): from neo._src.nn._activations import tanh; return tanh(self)
    def reshape(self, shape): 
        from neo._torch.user_functions import reshape as _reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
        return _reshape(self, shape)
