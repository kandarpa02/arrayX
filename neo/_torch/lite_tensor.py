from typing import NamedTuple
from torch import tensor, Tensor, dtype as Dtype
from neo._torch import neolib
from .functions import *
# from .d_config import _dtype
import numpy as np

def safe_input(self, x):
    if not isinstance(x, LiteTensor):
        if isinstance(x, int|float|Tensor):
            x = LiteTensor(x, d_type=self.dtype, device=self.device)
    return x

def _dtype(d_type):
    return getattr(torch, d_type) if isinstance(d_type, str) else d_type

def _device(device):
    return torch.device(device) if isinstance(device, str) else device

class LiteTensor:
    def __init__(
            self, 
            data,
            d_type = '',
            device = '',

              ):
        
        if not isinstance(data, Tensor):
            dtype = _dtype(d_type) if d_type else None
            dev = _device(device) if device else None
            self.data = torch.as_tensor(data, dtype=dtype, device=dev).detach()
        else:
            self.data = data.detach()

        self.d_type = self.data.dtype
        self.device = self.data.device

    @property
    def dtype(self):
        return self.d_type

    def __repr__(self):
        prefix = " " * len("LiteTensor(")
        arr_str = np.array2string(
            self.data.numpy(),
            precision=4,
            suppress_small=True,
            threshold=6,
            edgeitems=3,
            max_line_width=80,
            separator=', ',
            prefix=prefix
        )
        return f"LiteTensor({arr_str})"
    
    def __str__(self):
        prefix = " " * len("LiteTensor(")
        arr_str = np.array2string(
            self.data.numpy(),
            precision=4,
            suppress_small=True,
            threshold=6,
            edgeitems=3,
            max_line_width=80,
            separator=', ',
            prefix=prefix
        )
        return f"LiteTensor({arr_str})"

    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return LiteTensor(neolib.reshape(self.data, shape), device=self.device)

    def shape(self):
        return self.data.shape
    
    def size(self):
        return self.data.size 

    def __eq__(self, other): return isinstance(other, LiteTensor) and self.data == other.data

    def __len__(self):
        """Returns the number of elements along the first axis."""
        return len(self.data)

    def __getitem__(self, index):
        """Indexing access to data."""
        return self.data[index]

    def __ne__(self, other):
        """Non-equality check."""
        if isinstance(other, self.__class__):
            return self.data != other.data
        return NotImplemented

    def __lt__(self, other):
        """Less-than comparison."""
        if isinstance(other, self.__class__):
            return self.data < other.data
        return NotImplemented

    def __le__(self, other):
        """Less-than or equal comparison."""
        if isinstance(other, self.__class__):
            return self.data <= other.data
        return NotImplemented

    def __gt__(self, other):
        """Greater-than comparison."""
        if isinstance(other, self.__class__):
            return self.data > other.data
        return NotImplemented

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        if isinstance(other, self.__class__):
            return self.data >= other.data
        return NotImplemented

    def __hash__(self):
        """Object hash based on id."""
        return id(self)
    
    def __neg__(self):
        return neg(self.data)

    def __add__(self, other):
        b = safe_input(self, other)
        return add(self, b)
    
    def __sub__(self, other):
        b = safe_input(self, other)
        return sub(self, b)
    
    def __mul__(self, other):
        b = safe_input(self, other)
        return mul(self, b)
    
    def __pow__(self, other):
        b = safe_input(self, other)
        return power(self, other)

    
    def __truediv__(self, other):
        b = safe_input(self, other)
        return div(self, b)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rdiv__(self, other):
        return self.__truediv__(other)
    
    def __rpow__(self, other):
        return self.__pow__(other)
    
    def ones_like(self):
        out = neolib.ones_like(self.data)
        return LiteTensor(out, dtype=out.dtype, device=out.device)