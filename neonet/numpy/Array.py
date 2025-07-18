from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from neonet.backend import get_xp
from neonet.numpy.math.arithmetic import *
import numpy as np

__all__ = []

def safe_input(x):
    if not isinstance(x, Array):
        try:
            x = Array(x)
        except Exception:
            raise ValueError(f"{x} is not a valid dtype to convert to Array")
    return x


def add(x, y):
    op = addition()
    out_val = op.forward(x.value, y.value) 
    out = Array(out_val, device=x.device) 
    
    node = Node(out, (x, y), op.backward)
    TapeContext.add_node(node)
    return out
  

def mul(x, y):
    op = multiplication()
    out_val = op.forward(x.value, y.value)
    out = Array(out_val, device=x.device)
    
    node = Node(out, (x, y), op.backward)
    TapeContext.add_node(node)
    return out


@dataclass
class Array:
    value: Any
    device: str | None = 'cpu'
    _id: int = field(default_factory=lambda: uuid4().int, init=False, repr=False)


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
        return f"Array({arr_str})"

    def __str__(self):
        return str(self.value)
    

    @property
    def xp(self):
        return get_xp(self.device)
    
    def to(self, device: str, copy=True):
        return Array(self.xp.asarray(self.value, copy=copy), device=device)


    def numpy(self):
        return np.asarray(self.value) if self.device == 'cpu' else self.value
    
    @property
    def shape(self): return self.value.shape

    @property
    def dtype(self): return self.value.dtype


    def __hash__(self): return hash(self._id)

    def __eq__(self, other): return isinstance(other, Array) and self._id == other._id


    def __add__(self, other):
        b = safe_input(other)
        return add(self, b)
    
    def __mul__(self, other):
        b = safe_input(other)
        return mul(self, b) 

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    
    def ones_like(self):
        return Array(self.xp.ones_like(self.value))