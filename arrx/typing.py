from .src.Tensor.base import placeholder, vector, scalar, matrix
from .src.Tensor.utils import lib

Tensorlike = placeholder|scalar|vector|matrix
NDarray = lib.ndarray