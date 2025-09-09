from typing import List, Union
from numpy.typing import NDArray
from ..autograd.graph import TapeContext, Node
NumericObject = NDArray

def shift(data):
    return DeviceBuffer(data) if not isinstance(data, DeviceBuffer) else data

import numpy as np

def make_scalar(array):
    if isinstance(array, np.ndarray) and array.shape == ():
        return array.item()
    return array


class DeviceBuffer:
    __slots__ = ('_rawbuffer',)

    def __init__(self, data: NumericObject):
        self._rawbuffer = data

    def __repr__(self):
        return f"tensor({self._rawbuffer})"

    def __add__(self, other):
        other_buf = shift(other)
        out = DeviceBuffer(self._rawbuffer + other_buf._rawbuffer)
        
        TapeContext.add(
            Node(out, (self, other_buf), lambda grad: (
                make_scalar(grad), 
                make_scalar(grad)
            ))
        )
        
        return out
    
    def __mul__(self, other):
        other_buf = shift(other)
        x, y = self._rawbuffer, other_buf._rawbuffer
        out = DeviceBuffer(x * y)
        
        TapeContext.add(
            Node(out, (self, other_buf), lambda grad: (
                make_scalar(grad * y), 
                make_scalar(grad * x)
            ))
        )
        
        return out
