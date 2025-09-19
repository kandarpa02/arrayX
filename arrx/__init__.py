# Device init
from .backend import Backend, device
lib = Backend.initiate()

# Dtypes
from .Core.Dtype import (
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

# Auton module and typings
from typing import Union
from arrx import autometa as _atm
autometa = _atm

# User API
import numpy as np
NumericObject = Union[np.ndarray, int, float, list] 
def array(_x:NumericObject, dtype:Dtype = None): # type:ignore
    return ArrayImpl(_x, dtype=dtype)
# Core ops
from .Ops import *
from .Ops.util_fns import where, arange