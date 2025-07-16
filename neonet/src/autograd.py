from typing import Any

class function:
    def __init__(self):
        self._forward = self.forward
        self._backward = self.backward

    @staticmethod
    def forward(*args):
        pass

    @staticmethod
    def backward(tensors, kwargs):
        pass

    def fwd(self, *args):
        return self._forward(*args)
    
    def bwd(self, tensors, grad):
        return self._backward(tensors, grad)