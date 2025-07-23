from typing import NamedTuple
from torch import tensor, Tensor, dtype as Dtype
from neo._torch import neolib
from .functions import *
import numpy as np

def safe_input(self, x):
    if not isinstance(x, LiteTensor):
        if isinstance(x, int|float|Tensor):
            x = LiteTensor(x, d_type=self.dtype, device=self.device)
    return x

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
            self.data.cpu().numpy(),
            precision=4,
            suppress_small=True,
            threshold=6,
            edgeitems=3,
            max_line_width=80,
            separator=', ',
            prefix=prefix
        )
        return f"LiteTensor({arr_str})"
    
    def to(self, *args, **kwargs):
        """
        - .to(dtype)
        - .to(device)
        - .to(device, dtype)
        - .to(other_tensor)
        - .to(dtype=dtype, device=device)
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, LiteTensor):
                return LiteTensor(self.data.to(arg.data.device, arg.data.dtype))
            elif isinstance(arg, torch.Tensor):
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
        """Moves the tensor to CUDA. Optionally specify device like 0 or 'cuda:1'."""
        return LiteTensor(self.data.cuda(device=device))

    def cpu(self):
        """Moves the tensor to CPU."""
        return LiteTensor(self.data.cpu())

    def numpy(self):
        """Returns the underlying tensor as a NumPy array. Must be on CPU."""
        return self.data.detach().cpu().numpy()

    
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