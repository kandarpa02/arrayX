from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from neo.backend import get_xp, HAS_CUPY
from neo.numpy.math.arithmetic_policy import *
from neo.numpy.math.log_policy import *
from neo.numpy.math.unary_policy import *
from neo.numpy.math.reductions_policy import *
from neo.functions import neo_function
import numpy as np

__all__ = []

def safe_input(x):
    if not isinstance(x, Array):
        try:
            x = Array(x)
        except Exception:
            raise ValueError(f"{x} is not a valid dtype to convert to Array")
    return x


import warnings
from ..config import get_dtype, DTYPE_MAP

DEFAULT_DTYPE = "float32"

@dataclass
class Array:
    value: Any
    device: str = 'cpu'
    dtype: str = DEFAULT_DTYPE  # this is the string identifier like 'float32', 'bfloat16', etc.
    _id: int = field(default_factory=lambda: uuid4().int, init=False, repr=False)

    def __post_init__(self):
        if self.device == 'cuda' and not HAS_CUPY:
            warnings.warn("[Neo] CUDA not available. Falling back to CPU.")
            self.device = 'cpu'

        xp = self.xp
        dtype_obj = get_dtype(self.dtype, self.device)

        try:
            # Avoid redundant casting if already correct type and backend
            if not isinstance(self.value, xp.ndarray):
                self.value = xp.asarray(self.value, dtype=dtype_obj)
            elif self.value.dtype != dtype_obj:
                self.value = self.value.astype(dtype_obj)
        except Exception as e:
            warnings.warn(f"[Neo] Failed to cast array to {self.dtype}: {e}. Falling back to float32.")
            self.dtype = 'float32'
            self.value = xp.asarray(self.value, dtype=xp.float32)




    def astype(self, new_dtype: str) -> "Array":
        if new_dtype not in DTYPE_MAP:
            warnings.warn(f"Unsupported dtype '{new_dtype}', falling back to {DEFAULT_DTYPE}.")
            new_dtype = DEFAULT_DTYPE

        xp = self.xp
        backend = 'cupy' if self.device == 'cuda' else 'numpy'
        target_dtype = xp.__dict__[DTYPE_MAP[new_dtype][backend]]

        try:
            new_value = self.value.astype(target_dtype)
        except Exception as e:
            warnings.warn(f"[Neo] astype failed: {e}. Falling back to float32.")
            new_value = self.value.astype(xp.float32)
            new_dtype = 'float32'

        return Array(new_value, device=self.device, dtype=new_dtype)


    def __repr__(self):
        prefix = " " * len("Array(")
        arr_str = self.xp.array2string(
            self.value,
            precision=4,
            suppress_small=True,
            threshold=6,
            edgeitems=3,
            max_line_width=80,
            separator=', ',
            prefix=prefix
        )
        return f"Array({arr_str}, dtype={self.value.dtype}, device='{self.device}')"

    def __str__(self):
        return str(self.value)

    @property
    def xp(self):
        return get_xp(self.device)

    def to(self, device: str):
        if device == self.device:
            return self

        if device == 'cuda' and not HAS_CUPY:
            warnings.warn("[Neo] CUDA not available. Staying on CPU.")
            return self

        target_xp = get_xp(device)
        dtype_obj = get_dtype(self.dtype, device)

        # Move data
        if self.device == 'cuda' and device == 'cpu':
            value = self.value.get()
        elif self.device == 'cpu' and device == 'cuda':
            value = target_xp.asarray(self.value, dtype=dtype_obj)
        else:
            value = target_xp.asarray(self.value, dtype=dtype_obj)

        return Array(value=value, device=device, dtype=self.dtype)



    def numpy(self):
        return np.asarray(self.value) if self.device == 'cpu' else self.value
    
    @property
    def shape(self): return self.value.shape

    # @property
    # def dtype(self): return self.d_type


    def __hash__(self): return hash(self._id)

    def __eq__(self, other): return isinstance(other, Array) and self._id == other._id

    def __len__(self):
        """Returns the number of elements along the first axis."""
        return len(self.value)

    def __getitem__(self, index):
        """Indexing access to value."""
        return self.value[index]

    def __ne__(self, other):
        """Non-equality check."""
        if isinstance(other, self.__class__):
            return self.value != other.value
        return NotImplemented

    def __lt__(self, other):
        """Less-than comparison."""
        if isinstance(other, self.__class__):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        """Less-than or equal comparison."""
        if isinstance(other, self.__class__):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        """Greater-than comparison."""
        if isinstance(other, self.__class__):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        if isinstance(other, self.__class__):
            return self.value >= other.value
        return NotImplemented

    def __hash__(self):
        """Object hash based on id."""
        return id(self)
    
    def __neg__(self):
        return neo_function(negative_op)(self)

    def __add__(self, other):
        b = safe_input(other)
        return neo_function(addition)(self, b)
    
    def __sub__(self, other):
        b = safe_input(other)
        return neo_function(subtraction)(self, b)
    
    def __mul__(self, other):
        b = safe_input(other)
        return neo_function(multiplication)(self, b)
    
    def __truediv__(self, other):
        b = safe_input(other)
        return neo_function(division)(self, b)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rdiv__(self, other):
        return self.__truediv__(other)
    


    def ones_like(self, dtype='float32'): return Array(self.xp.ones_like(self.value))

    def zeros_like(self, dtype='float32'): return Array(self.xp.zeros_like(self.value))

    