from typing import Any, Sequence
from ..utils import (
    _unbroadcast,
    broadcast_shape,
    filler_name,
    data_by_dim,
    reduced_shape,
    broadcast_to,
)

from arrx import lib
from ..errors import ShapeError


class placeholder:
    def __init__(self, shape:Sequence[Any]=[], name=None): #type:ignore
        self.name_given = False if name is None else True
        self.name = name if name is not None else filler_name()
        self.shape = tuple(shape)
        self.parents = ()
        self.grad_fn = None
        self.grad = None

    @staticmethod
    def _make_place(a): 
        if isinstance(a, int|float|lib.ndarray):
            shape = ()
            if isinstance(a, lib.ndarray):
                shape = a.shape
            return placeholder.place(*shape, name=f"{a}")
        elif isinstance(a, placeholder):
            return a

    @staticmethod
    def _TYPE_MAP(name):
        tmap = {
            'matrix':matrix,
            'vector':vector,
            'scalar':scalar
        }
        return tmap.get(name, placeholder)
    
    @staticmethod
    def as_place(data, name=None):
        shape = data.shape
        _name = data_by_dim(*shape)
        tmap = {
            'matrix':matrix,
            'vector':vector,
            'scalar':scalar
        }
        return tmap.get(_name, placeholder)(list(shape), name) #type:ignore
    
    @staticmethod
    def place(*shape, name=None):
        _name = data_by_dim(*shape)
        tmap = {
            'matrix':matrix,
            'vector':vector,
            'scalar':scalar
        }
        return tmap.get(_name, placeholder)(list(shape), name) #type:ignore
    
    @staticmethod
    def object(*shape):
        name = data_by_dim(*shape)
        tmap = {
            'matrix':matrix,
            'vector':vector,
            'scalar':scalar
        }
        return tmap.get(name, placeholder) #type:ignore
    
    def __repr__(self):
        if hasattr(self, 'repr'):
            return self.repr() #type:ignore
        ret = f"Placeholder(shape={self.shape})"
        return ret
    
    @property
    def ndim(self):
        return len(self.shape) if self.shape != () else 0


    def __add__(self, other):
        other = placeholder._make_place(other) 
        _shape = broadcast_shape(self.shape, other.shape)#type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"({self.name} + {other.name})", shape=_shape) #type:ignore
        out.parents = (self, other)
        def _grad_add(grad):
            g1 = _unbroadcast(grad, self.shape)
            g2 = _unbroadcast(grad, other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_add
        return out
    

    def __mul__(self, other):
        other = placeholder._make_place(other)
        _shape = broadcast_shape(self.shape, other.shape) #type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"({self.name} * {other.name})", shape=_shape) #type:ignore
        out.parents = (self, other)
        # note: grad_fn returns placeholders (symbolic expressions using names)
        def _grad_mul(grad):
            g1 = _unbroadcast(grad * other, self.shape)
            g2 = _unbroadcast(grad * self, other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_mul

        return out

    
    def reshape(self, *shape):
        # create a placeholder of the correct type and give it the explicit name
        obj = placeholder.object(*shape)           # get class (vector/matrix/...)
        out = obj(list(shape), f"{self.name}.reshape{shape}")
        out.parents = (self, )

        def _grad_reshape(grad):
            # return a placeholder representing grad reshaped back to original
            return grad.reshape(self.shape),

        out.grad_fn = _grad_reshape
        return out

    
class scalar(placeholder):
    def __init__(self, shape:Sequence[Any]=[], name=None): #type:ignore
        super().__init__(shape, name)

    def repr(self):
        res = f"Scalar(shape={self.shape})" 
        return res



class vector(placeholder):
    def __init__(self, shape:Sequence[Any]=[], name=None): #type:ignore
        super().__init__(shape, name)


    def sum(self, axis=None, keepdims=False):
        shape = reduced_shape(self.shape, axis=axis, keepdims=keepdims)
        obj = placeholder.object(*shape)
        out = obj(shape, f"{self.name}.sum(axis={axis}, keepdims={keepdims})")
        out.parents = (self,)

        def _grad_sum(grad):
            if keepdims:
                grad_expanded = grad
            else:
                shape = list(self.shape)
                if axis is None:
                    shape = [1] * self.ndim
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                grad_expanded = grad.reshape(tuple(shape))
            
            grad_broadcasted = broadcast_to(grad_expanded.name, self.shape)
            return (grad_broadcasted,)

        out.grad_fn = _grad_sum
        return out

    def repr(self):
        res = f"Vector(shape={self.shape})" 
        return res
    
class matrix(vector):
    def __init__(self, shape:Sequence[Any]=[], name=None): #type:ignore
        super().__init__(shape, name)
        self._check_dim()
    
    def _check_dim(self):
        if self.ndim < 2:
            raise ShapeError(self.ndim, 'matrix')

    
    def repr(self):
        res = f"Matrix(shape={self.shape})" 
        return res 