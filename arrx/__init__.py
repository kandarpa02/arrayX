# Device init
from .backend import Backend
lib = Backend.initiate()

from typing import Union
from .autograd.backward import grad as _grad, value_and_grad as _vag

grad = _grad
value_and_grad = _vag


import numpy as np

NumericObject = Union[np.ndarray, int, float, list] 
def array(_x:NumericObject):
    from .core.Array import ArrayImpl
    import numpy
    return ArrayImpl(lib.array(_x))

from .autograd.utils import custom_grad
from .ops import *