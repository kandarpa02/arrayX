from typing import Any, Sequence
from .utils import _unbroadcast

class placeholder:
    def __init__(self, name=None, shape:Sequence[Any]=[]): #type:ignore
        self.name = name
        self.shape = tuple(shape)
        self.parents = ()
        self.grad_fn = None
        self.grad = None

    def __repr__(self):
        if hasattr(self, 'repr'):
            return self.repr() #type:ignore
        return f"placeholder({self.name}, shape={self.shape})"


    def __add__(self, other):
        out = placeholder(f"({self.name} + {other.name})", shape=[])
        out.parents = (self, other)
        def _grad_add(grad):
            g1 = _unbroadcast(grad, self.shape)
            g2 = _unbroadcast(grad, other.shape)
            return g1, g2
        
        out.grad_fn = _grad_add
        return out
    

    def __mul__(self, other):
        out = placeholder(f"({self.name} * {other.name})", shape=[])
        out.parents = (self, other)
        # note: grad_fn returns placeholders (symbolic expressions using names)
        def _grad_mul(grad):
            g1 = _unbroadcast(grad * other, self.shape)
            g2 = _unbroadcast(grad * self, other.shape)
            return g1, g2
        
        out.grad_fn = _grad_mul

        return out

    def repr(self):
        return f"scalar({self.name}, shape={self.shape})"
    
    def reshape(self, *shape):
        out = placeholder(f"{self.name.reshape(shape)}") #type:ignore
        out.parents = (self, )

        def _grad_reshape(grad):
            return grad.reshape(self.shape),

        out.grad_fn = _grad_reshape
        return out


class scalar(placeholder):
    def __init__(self, name=None, shape: Sequence = []): #type:ignore
        super().__init__(name, shape)

    def repr(self):
        return f"scalar({self.name}, shape={self.shape})"
    
class vector(placeholder):
    def __init__(self, name=None, shape: Sequence = []):
        super().__init__(name, shape)

    def repr(self):
        return f"vector({self.name}, shape={self.shape})"
    
class matrix(placeholder):
    def __init__(self, name=None, shape: Sequence = []): 
        super().__init__(name, shape)
    
    def repr(self):
        return f"matrix({self.name}, shape={self.shape})"