from .autograd.graph import Function
from .autograd.custom import Trace
from .autograd.graph_utils import FlashGraph
from .Tensor.base import scalar, placeholder, vector, matrix
from .Tensor.arithmetic import matmul, dot
from .Tensor.logarithmic import log
from .Tensor.extraops import cond, where

from .utils import variable