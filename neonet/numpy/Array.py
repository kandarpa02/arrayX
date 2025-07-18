from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from neonet.backend import get_xp
import numpy as np

def safe_input(x):
    if not isinstance(x, Array):
        try:
            x = Array(x)
        except Exception:
            raise ValueError(f"{x} is not a valid dtype to convert to Array")
    return x

@dataclass
class Array:
    value: Any
    device: str = 'cpu'
    _id: int = field(default_factory=lambda: uuid4().int, init=False, repr=False)

    @property
    def xp(self):
        return get_xp(self.device)

    def numpy(self):
        return np.asarray(self.value) if self.device == 'cpu' else self.value.get()
    
    def to(self, device: str, copy=True):
        return Array(self.xp.asarray(self.value, copy=copy), device=device)

    @property
    def shape(self): return self.value.shape
    @property
    def dtype(self): return self.value.dtype

    def __hash__(self): return hash(self._id)
    def __eq__(self, other): return isinstance(other, Array) and self._id == other._id

    def __add__(self, other):
        a = safe_input(self)
        b = safe_input(other)
        return Array(a.value + b.value, device=a.device)

    def __mul__(self, other):
        a = safe_input(self)
        b = safe_input(other)
        return Array(a.value * b.value, device=a.device)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

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
