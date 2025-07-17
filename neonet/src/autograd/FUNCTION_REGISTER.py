from typing import Any
from .._object import context
from dataclasses import dataclass


@dataclass
class _ctx:
    val = context()
    def save_tensors(self, *args):
        return self.val.save(*args)
    
    @property
    def saved_tensors(self):
        return self.val.release

class Policy:
    ctx = _ctx()
    def __init__(self):
        self._forward = self.__class__.forward
        self._backward = self.__class__.backward
    
    @classmethod
    def forward(cls, *args):
        pass

    @classmethod
    def backward(cls, kwargs=None):
        pass

    def fwd(self, *tensors) -> Any:
        return self._forward(*tensors)
    
    def bwd(self, grad) -> Any:
        return self._backward(grad)
    
