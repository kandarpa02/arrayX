from typing import Any, Sequence
from .utils import (
    _unbroadcast,
    broadcast_shape,
    filler_name,
    data_by_dim,
    reduced_shape,
    broadcast_to,
    matmul_shape,
    reshape_shape,
    transpose_shape,
    max_min_shape,
)

from ..errors import ShapeError


class placeholder:
    def __init__(self, shape:Sequence[Any]=[], name=None): #type:ignore
        self.expr_given = False if name is None else True
        self.expr = name if name is not None else filler_name()
        self.shape = tuple(shape)
        self.parents = ()
        self.grad_fn = None
        self.grad = None

    @staticmethod
    def _make_place(a): 
        from .utils import lib
        NDarray = lib.Array
        if isinstance(a, int|float|NDarray):
            shape = ()
            if isinstance(a, NDarray):
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
    def zeors(*shape):
        return placeholder.place(*shape, name=f"lib.numpy.zeros({shape})")
    @staticmethod
    def ones(*shape):
        return placeholder.place(*shape, name=f"lib.numpy.ones({shape})")
    
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
    
    @property
    def size(self):
        res = 1
        for s in self.shape:
            res *= s
        return res
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        other = placeholder._make_place(other)
        shape = broadcast_shape(self.shape, other.shape) #type:ignore
        out = placeholder.place(*shape, name=f'({self.expr}=={other.expr})') #type:ignore

        out.parents = (self, other)

        def grad_fn(grad):
            def _zero_like_place(x):
                return placeholder.place(*x.shape, name=f"lib.zeros({x.shape})")
            g_self = _zero_like_place(self)
            g_other = _zero_like_place(other)
            return g_self, g_other

        out.grad_fn = grad_fn
        return out

    def __ne__(self, other):
        other = placeholder._make_place(other)
        shape = broadcast_shape(self.shape, other.shape) #type:ignore
        out = placeholder.place(*shape, name=f'({self.expr}!={other.expr})') #type:ignore

        out.parents = (self, other)

        def grad_fn(grad):
            def _zero_like_place(x):
                return placeholder.place(*x.shape, name=f"lib.zeros({x.shape})")
            g_self = _zero_like_place(self)
            g_other = _zero_like_place(other)
            return g_self, g_other

        out.grad_fn = grad_fn
        return out


    def __gt__(self, other):
        other = placeholder._make_place(other)
        shape = broadcast_shape(self.shape, other.shape) #type:ignore
        out = placeholder.place(*shape, name=f'({self.expr}>{other.expr})') #type:ignore

        out.parents = (self, other)

        def grad_fn(grad):
            def _zero_like_place(x):
                return placeholder.place(*x.shape, name=f"lib.zeros({x.shape})")
            g_self = _zero_like_place(self)
            g_other = _zero_like_place(other)
            return g_self, g_other

        out.grad_fn = grad_fn
        return out


    def __lt__(self, other):
        other = placeholder._make_place(other)
        shape = broadcast_shape(self.shape, other.shape) #type:ignore
        out = placeholder.place(*shape, name=f'({self.expr}<{other.expr})') #type:ignore

        out.parents = (self, other)

        def grad_fn(grad):
            def _zero_like_place(x):
                return placeholder.place(*x.shape, name=f"lib.zeros({x.shape})")
            g_self = _zero_like_place(self)
            g_other = _zero_like_place(other)
            return g_self, g_other

        out.grad_fn = grad_fn
        return out

    def __ge__(self, other):
        other = placeholder._make_place(other)
        shape = broadcast_shape(self.shape, other.shape) #type:ignore
        out = placeholder.place(*shape, name=f'({self.expr}>={other.expr})') #type:ignore

        out.parents = (self, other)

        def grad_fn(grad):
            def _zero_like_place(x):
                return placeholder.place(*x.shape, name=f"lib.zeros({x.shape})")
            g_self = _zero_like_place(self)
            g_other = _zero_like_place(other)
            return g_self, g_other

        out.grad_fn = grad_fn
        return out

    def __le__(self, other):
        other = placeholder._make_place(other)
        shape = broadcast_shape(self.shape, other.shape) #type:ignore
        out = placeholder.place(*shape, name=f'({self.expr}<={other.expr})') #type:ignore

        out.parents = (self, other)

        def grad_fn(grad):
            def _zero_like_place(x):
                return placeholder.place(*x.shape, name=f"lib.zeros({x.shape})")
            g_self = _zero_like_place(self)
            g_other = _zero_like_place(other)
            return g_self, g_other

        out.grad_fn = grad_fn
        return out


    def __add__(self, other):
        other = placeholder._make_place(other) 
        _shape = broadcast_shape(self.shape, other.shape) #type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"lib.add({self.expr}, {other.expr})", shape=_shape) #type:ignore
        out.parents = (self, other)
        def _grad_add(grad):
            g1 = _unbroadcast(grad, self.shape)
            g2 = _unbroadcast(grad, other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_add
        return out
    
    def __sub__(self, other):
        other = placeholder._make_place(other) 
        _shape = broadcast_shape(self.shape, other.shape)#type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"(lib.subtract{self.expr}, {other.expr})", shape=_shape) #type:ignore
        out.parents = (self, other)
        def _grad_sub(grad):
            g1 = _unbroadcast(grad, self.shape)
            g2 = _unbroadcast(-grad, other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_sub
        return out
    

    def __mul__(self, other):
        other = placeholder._make_place(other)
        _shape = broadcast_shape(self.shape, other.shape) #type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"lib.multiply({self.expr}, {other.expr})", shape=_shape) #type:ignore
        out.parents = (self, other)
        # note: grad_fn returns placeholders (symbolic expressions using names)
        def _grad_mul(grad):
            g1 = _unbroadcast(grad * other, self.shape)
            g2 = _unbroadcast(grad * self, other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_mul
        return out
    
    def __truediv__(self, other):
        other = placeholder._make_place(other)
        _shape = broadcast_shape(self.shape, other.shape) #type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"lib.divide({self.expr}, {other.expr})", shape=_shape) #type:ignore
        out.parents = (self, other)
        # note: grad_fn returns placeholders (symbolic expressions using names)
        def _grad_div(grad):
            g1 = _unbroadcast(grad / other, self.shape)
            g2 = _unbroadcast(-grad * self / (other * other), other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_div
        return out
    
    def __pow__(self, other):
        from .logarithmic import log
        other = placeholder._make_place(other)
        _shape = broadcast_shape(self.shape, other.shape) #type:ignore
        obj = placeholder.object(*_shape)
        out = obj(name=f"lib.power({self.expr}, {other.expr})", shape=_shape) #type:ignore
        out.parents = (self, other)
        # note: grad_fn returns placeholders (symbolic expressions using names)
        def _grad_mul(grad):
            g1 = _unbroadcast(grad * other * self ** (other - 1), self.shape) #type:ignore
            g2 = _unbroadcast(grad * self ** other * log(self + 1e-12), other.shape) #type:ignore
            return g1, g2
        
        out.grad_fn = _grad_mul
        return out
    
    def __neg__(self):
        out = placeholder.place(*self.shape, name=f"-{self.expr}")
        out.parents = (self,)
        def _grad_neg(grad):
            g = _unbroadcast(-grad, self.shape)
            return (g,)
        return out
    
    def reshape(self, *shape):
        def tup_norm(s):
            res = []
            for i in s:
                if isinstance(i, tuple):
                    res.extend(tup_norm(i))
                else: 
                    res.extend([i])

            return tuple(res)
        
        shape = tup_norm(shape)
        new_shape = reshape_shape(self.shape, shape)
        obj = placeholder.object(*new_shape)           # get class (vector/matrix/...)
        out = obj(new_shape, f"{self.expr}.reshape{shape}")
        out.parents = (self, )

        def _grad_reshape(grad):
            # return a placeholder representing grad reshaped back to original
            return grad.reshape(self.shape),

        out.grad_fn = _grad_reshape
        return out
    
    def exp(self):
        out = placeholder.place(*self.shape, name=f"lib.exp({self.expr})")
        out.parents = (self,)

        def _exp_grad(grad):
            return grad * out,
        
        out.grad_fn = _exp_grad
        return out

class scalar(placeholder):
    def __init__(self, shape: Sequence[Any] = [], name=None):  # type: ignore
        super().__init__(shape, name)
        self._check_dim()
    
    def _check_dim(self):
        if self.ndim != 0:
            raise ShapeError(
                f"Invalid shape for 'scalar': {self.shape} (ndim={self.ndim}). "
                f"A scalar must have no dimensions (shape=())."
            )

    def repr(self):
        return f"Scalar(shape={self.shape})"

class vector(placeholder):
    def __init__(self, shape:Sequence[Any]=[], name=None): #type:ignore
        super().__init__(shape, name)
        self._check_dim()
    
    def _check_dim(self):
        if self.ndim != 1:
            raise ShapeError(
                f"Invalid shape for 'vector': {self.shape} (ndim={self.ndim}). "
                f"A vector must be one-dimensional."
            )

    def repr(self):
        res = f"Vector(shape={self.shape})" 
        return res


    def sum(self, axis=None, keepdims=False):
        shape = reduced_shape(self.shape, axis=axis, keepdims=keepdims)
        obj = placeholder.object(*shape)
        out = obj(shape, f"lib.sum({self.expr}, axis={axis}, keepdims={keepdims})")
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
            
            grad_broadcasted = broadcast_to(grad_expanded.expr, self.shape)
            return (grad_broadcasted,)

        out.grad_fn = _grad_sum
        return out
    
    def mean(self, axis=None, keepdims=False):
        shape = reduced_shape(self.shape, axis=axis, keepdims=keepdims)
        obj = placeholder.object(*shape)
        out = obj(shape, f"lib.mean({self.expr}, axis={axis}, keepdims={keepdims})")
        out.parents = (self,)
        
        def _grad_mean(grad):
            if axis is None:
                n = self.size
                shape = [1] * self.ndim
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)

                import numpy as np
                n = np.prod([self.shape[ax] for ax in axes]).item()
                shape = list(self.shape)
                for ax in axes:
                    shape[ax] = 1

            if not keepdims:
                grad_shape = list(grad.shape)
                for ax in sorted(axes):
                    grad_shape.insert(ax, 1)
                grad_expanded = grad.reshape(tuple(grad_shape))
            else:
                grad_expanded = grad

            grad_broadcasted = broadcast_to((grad_expanded / n).expr, self.shape)
            return (grad_broadcasted, )

        out.grad_fn = _grad_mean
        return out
    
    def max(self, axis=None, keepdims=False):
        shape = max_min_shape(self.shape, axis, keepdims)
        out = placeholder.place(*shape, name=f'lib.max({self.expr}, axis={axis}, keepdims={keepdims})') #type:ignore
        out.parents = self,

        def _grad_max(grad):
            # Expand grad to ilibut shape if keepdims is False
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

            grad_broadcasted = broadcast_to(grad_expanded.expr, self.shape)
            if axis is None:
                mask = (self == out)
            else:
                max_expanded = out
                if not keepdims:
                    shape = list(self.shape)
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                    max_expanded = out.reshape(tuple(shape))
                mask = (self == max_expanded)

            grad_final = grad_broadcasted * mask #.astype(self.dtype)
            return grad_final, 
        out.grad_fn = _grad_max
        return out
    
    def min(self, axis=None, keepdims=False):
        shape = max_min_shape(self.shape, axis, keepdims)
        out = placeholder.place(*shape, name=f'lib.min({self.expr}, axis={axis}, keepdims={keepdims})') #type:ignore
        out.parents = self,

        def _grad_max(grad):
            # Expand grad to ilibut shape if keepdims is False
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

            grad_broadcasted = broadcast_to(grad_expanded.expr, self.shape)
            if axis is None:
                mask = (self == out)
            else:
                max_expanded = out
                if not keepdims:
                    shape = list(self.shape)
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                    max_expanded = out.reshape(tuple(shape))
                mask = (self == max_expanded)

            grad_final = grad_broadcasted * mask #.astype(self.dtype)
            return grad_final, 
        out.grad_fn = _grad_max
        return out
    
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        shape = transpose_shape(self.shape, axes)
        out = placeholder.place(*shape, name=f"{self.expr}.transpose({axes})")
        out.parents = (self,)

        def _grad_transpose(grad):
            def inverse_permutation(axes):
                inv = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv[a] = i
                return tuple(inv)

            inv_axes = inverse_permutation(axes)
            return grad.transpose(inv_axes),

        out.grad_fn = _grad_transpose
        return out
    
    @property
    def T(self):
        return self.transpose()


    def flatten(self):
        return self.reshape(-1)
    
    def __matmul__(self, other):
        shape = matmul_shape(self.shape, other.shape)
        out = placeholder.place(*shape, name=f"lib.matmul({self.expr}, {other.expr})")
        out.parents = (self, other)

        def _grad_matmul(grad):

            a_exp = self if self.ndim > 1 else self.reshape(1, -1)
            b_exp = other if other.ndim > 1 else other.reshape(-1, 1)
            grad_exp = grad if grad.ndim > 1 else grad.reshape(1, -1)

            grad_a = grad_exp @ b_exp.T
            grad_b = a_exp.T @ grad_exp

            # reshape outputs if inputs were originally 1D
            grad_a = grad_a.flatten() if self.ndim == 1 else grad_a
            grad_b = grad_b.flatten() if other.ndim == 1 else grad_b

            return grad_a, grad_b
        
        out.grad_fn = _grad_matmul
        
        return out
    
class matrix(vector):
    def __init__(self, shape: Sequence[Any] = [], name=None):  # type: ignore
        super().__init__(shape, name)
        self._check_dim()
    
    def _check_dim(self):
        if self.ndim < 2:
            raise ShapeError(
                f"Invalid shape for 'matrix': {self.shape} (ndim={self.ndim}). "
                f"A matrix must have at least 2 dimensions."
            )

    def repr(self):
        return f"Matrix(shape={self.shape})"
