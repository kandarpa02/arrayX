from typing import List, Union
from numpy.typing import NDArray
import numpy as np

NumericObject = NDArray

class GradBuffer:
    pass

def shift(data):
    if isinstance(data, GradBuffer):
        return data
    if isinstance(data, DeviceBuffer):
        return GradBuffer(data)
    # If scalar or array, wrap as DeviceBuffer then as GradBuffer
    return GradBuffer(DeviceBuffer(np.array(data)), parents=(), bwd_fn=None)


class DeviceBuffer:
    __slots__ = ('_rawbuffer',)

    def __init__(self, data: NumericObject):
        self._rawbuffer = data._rawbuffer if isinstance(data, DeviceBuffer) else data

    def __repr__(self):
        return f"tensor({self._rawbuffer})"

    def ones_like(self):
        return DeviceBuffer(np.ones_like(self._rawbuffer))
    
    def zero_like(self):
        return DeviceBuffer(np.zeros_like(self._rawbuffer))


class GradBuffer(DeviceBuffer):
    def __init__(self, data, parents=(), bwd_fn=None):
        super().__init__(data._rawbuffer if isinstance(data, DeviceBuffer) else data)
        # Always ensure parents is a tuple of length 2 by padding with None
        if len(parents) == 1:
            parents = (parents[0], None)
        self.parents = parents
        self.bwd_fn = bwd_fn

    def __add__(self, other):
        other = shift(other)
        out = GradBuffer(self._rawbuffer + other._rawbuffer,
                         parents=(self, other),
                         bwd_fn=self._grad_add)
        return out

    def __mul__(self, other):
        other = shift(other)
        out = GradBuffer(self._rawbuffer * other._rawbuffer,
                        parents=(self, other))
        out._left = self
        out._right = other
        out.bwd_fn = out._grad_mul
        return out

    def _grad_add(self, grad):
        return grad, grad

    def _grad_mul(self, grad):
        return grad * self._right, grad * self._left

    def __radd__(self, other):
        return shift(other) + self
    
    def __rmul__(self, other):
        return shift(other) * self
