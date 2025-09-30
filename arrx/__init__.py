from .backend import Backend
Backend.initiate()

# # Dtypes
from .Dtype import (
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

from .autograd.graph import Function
from .autograd.custom import Trace
from .autograd.graph_utils import FlashGraph
from .Tensor.base import scalar, placeholder, vector, matrix
from .Tensor.arithmetic import *
from .Tensor.logarithmic import log
from .Tensor.extraops import where

from .Tensor.utils import variable