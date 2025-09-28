from .base import internal
from ..Tensor.base import placeholder, vector, scalar, matrix
from ..autograd.custom import Trace
from typing import Any, Union

__all__ = ['where_', 'where_sig']

TensorLike = Union[placeholder, scalar, vector, matrix]

class whereObj(internal):
    def __init__(self) -> None:
        super().__init__(name='where_', signature=('condition', 'x', 'y'))

    def call(self, condition, x, y):
        if condition:
            return x
        else:
            return y
    
def where_(condition, x, y):
    return whereObj.apply(condition, x, y)

def where_sig():
    return whereObj()()