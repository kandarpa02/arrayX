# Device init
from .backend import Backend, device
lib = Backend.initiate()

# Dtypes
from .core.Dtype import (
Dtype, floating, integer, unsignedinteger,
float16, float32, float64, float128,
int8, int16, int32, int64, int128,
uint8, uint16, uint32, uint64, uint128,
boolean
)

float16 = float16()
float32 = float32()
float64 = float64()
float128 = float128()
int8 = int8()
int16 = int16()
int32 = int32()
int64 = int64()
int128 = int128()
uint8 = uint8()
uint16 = uint16()
uint32 = uint32()
uint64 = uint64()
uint128 = uint128()
bool = boolean()


from typing import Union
from .autograd.backward import grad as _grad, value_and_grad as _vag

grad = _grad
value_and_grad = _vag


import numpy as np

NumericObject = Union[np.ndarray, int, float, list] 
def array(_x:NumericObject, dtype:Dtype = None): # type:ignore
    return ArrayImpl(_x, dtype=dtype)

from .autograd.utils import custom_grad
from .ops import *