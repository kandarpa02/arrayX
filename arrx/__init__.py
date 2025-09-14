from .typing import Array, NumericObject

def array(_x:NumericObject):
    from .core.Array import DeviceArray
    import numpy
    return DeviceArray(numpy.array(_x))

from .autograd.utils import custom_grad
from .ops import *