from typing import Any, Sequence
from .utils import _unbroadcast
from arrx import lib
from .errors import ShapeError

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
    
    @property
    def ndim(self):
        return len(self.shape) if self.shape != () else 0


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
        out = placeholder(f"{self.name}.reshape(shape)") #type:ignore
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


    def sum(self, axis=None, keepdims=False):
        out = vector(f"{self.name}.sum()")
        out.parents = (self,)

        def _grad_sum(grad):
            if keepdims:
                grad_expanded = grad._rawbuffer
            else:
                shape = list(self.shape)
                if axis is None:
                    shape = [1] * self.ndim
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                grad_expanded = grad.reshape(shape)
            
            # grad_broadcasted = lib.broadcast_to(grad_expanded, self.shape)
            return (grad_expanded,)

        out.grad_fn = _grad_sum
        return out

    def repr(self):
        return f"vector({self.name}, shape={self.shape})"
    
class matrix(vector):
    def __init__(self, name=None, shape: Sequence = []): 
        super().__init__(name, shape)
        self._check_dim()
    
    def _check_dim(self):
        if self.ndim < 2:
            raise ShapeError(self.ndim, 'matrix')

    
    def repr(self):
        return f"matrix({self.name}, shape={self.shape})"