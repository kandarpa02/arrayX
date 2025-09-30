from typing import List, Tuple, Optional, Callable, Union, Any
from numpy.typing import NDArray
from dataclasses import dataclass
from arrx import lib
import numpy as np
from .Dtype import Dtype, dmap, int32, float32, boolean

NumericObject = NDArray

def shift(data):
    if isinstance(data, ArrayImpl):
        return data
    if isinstance(data, ArrayStorage):
        return ArrayImpl(data)
    # If scalar or array, wrap as ArrayStorage then as ArrayImpl
    return ArrayImpl(data, parents=(), bwd_fn=None)


def _unbroadcast(grad, shape: Tuple[int, ...]):
    """Reduce grad to match shape by summing over broadcasted dimensions."""
    grad_shape = grad._rawbuffer.shape
    if grad_shape == shape:
        return grad
    # Add leading ones to shape if needed
    while len(shape) < len(grad_shape):
        shape = (1,) + shape
    axes = tuple(i for i, (g, s) in enumerate(zip(grad_shape, shape)) if s == 1)
    if axes:
        grad_reduced = grad._rawbuffer.sum(axis=axes, keepdims=True)
    else:
        grad_reduced = grad._rawbuffer
    # Now remove the extra dimensions
    grad_reduced = grad_reduced.reshape(shape)
    return ArrayImpl(grad_reduced)


def dtype_init(data) -> Dtype: 
    if isinstance(data, bool):       # check this first
        return boolean()
    elif isinstance(data, int):
        return int32()
    elif isinstance(data, float):
        return float32()
    
    elif isinstance(data, list):
        def check_data(data):
            flag = False
            for i in data:
                if isinstance(i, float):
                    flag = True
                elif isinstance(i, list):
                    flag = check_data(i)
            return flag
        if not check_data(data):
            return int32()
        else:
            return float32()
        
    elif isinstance(data, ArrayImpl):
        return dmap(data._rawbuffer.dtype.type)  # type:ignore
    
    else:
        return dmap(data.dtype.type)  # type:ignore

    
class ArrayStorage:
    __slots__ = ('_rawbuffer', '_dtype')

    def __init__(self, data: NumericObject, _dtype:Dtype|type = None): #type:ignore
        self._dtype = _dtype() if _dtype is not None else dtype_init(data)()
        self._rawbuffer = data._rawbuffer if isinstance(data, ArrayStorage) else lib.array(data, dtype=self._dtype)

    def __repr__(self):
        out = lib.array2string(self._rawbuffer, prefix='array(')
        return f'array({out}, dtype={self.dtype})'

    def __str__(self):
        return str(self._rawbuffer)
    
    def astype(self, dtype:Dtype):
        return ArrayImpl(self._rawbuffer.astype(dtype()), dtype=dtype)
    
    def all(self, axis=None, keepdims=False):
        _all = ArrayImpl(self._rawbuffer.all(axis=axis, keepdims=keepdims))
        _all._dtype = self._dtype
        return _all

    @property
    def shape(self):
        return self._rawbuffer.shape

    @property
    def dtype(self):
        return dmap(self._rawbuffer.dtype.type)

    def numpy(self):
        return np.asarray(self._rawbuffer)


class ArrayImpl(ArrayStorage):
    def __init__(self, data, parents=(), bwd_fn=None, dtype:Dtype=None): #type:ignore
        super().__init__(data._rawbuffer if isinstance(data, ArrayStorage) else data, _dtype=dtype)
        self.parents: Tuple['ArrayImpl', ...] = parents
        self.bwd_fn: Optional[Callable] = bwd_fn

    # Get items
    def __setitem__(self, k, v):
        self._rawbuffer[k]=v._rawbuffer

    def __getitem__(self, i):
        # Handle multiple indices (tuple indexing)
        if isinstance(i, tuple):
            idx = tuple(j._rawbuffer if isinstance(j, ArrayImpl) else j for j in i)
        else:
            idx = i._rawbuffer if isinstance(i, ArrayImpl) else i

        out = ArrayImpl(self._rawbuffer[idx], parents=(self,))

        def _get_backward(grad):
            grad_buf = grad._rawbuffer
            # create zero buffer with same shape/dtype as the original array
            zero_buf = lib.zeros_like(self._rawbuffer)

            try:
                zero_buf[idx] = zero_buf[idx] + grad_buf
            except Exception:
                if hasattr(lib, "add") and hasattr(lib.add, "at"):
                    lib.add.at(zero_buf, idx, grad_buf)
                else:
                    zero_buf[idx] += grad_buf

            return (ArrayImpl(zero_buf),)

        out.bwd_fn = _get_backward
        return out
    
    def __len__(self):
        try:
            return len(self._rawbuffer)
        except TypeError:
            raise TypeError("len() of unsized object")

    
    # Comparison operations
    def __eq__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer == other._rawbuffer))

    def __ne__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer != other._rawbuffer))

    def __gt__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer > other._rawbuffer))

    def __lt__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer < other._rawbuffer))

    def __ge__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer >= other._rawbuffer))

    def __le__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer <= other._rawbuffer))

    # Arithmetic operations
    def __add__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer + other._rawbuffer, parents=(self, other))

        def _grad_add(grad):
            g1 = _unbroadcast(grad, self._rawbuffer.shape)
            g2 = _unbroadcast(grad, other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_add
        return out

    def __radd__(self, other):
        return shift(other) + self

    def __sub__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer - other._rawbuffer, parents=(self, other))

        def _grad_sub(grad):
            g1 = _unbroadcast(grad, self._rawbuffer.shape)
            g2 = _unbroadcast(-grad, other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_sub
        return out


    def __rsub__(self, other):
        other = shift(other)
        out = ArrayImpl(other._rawbuffer - self._rawbuffer, parents=(other, self))

        def _grad_rsub(grad):
            g1 = _unbroadcast(grad, other._rawbuffer.shape)   # wrt other (left)
            g2 = _unbroadcast(-grad, self._rawbuffer.shape)   # wrt self (right)
            return g1, g2

        out.bwd_fn = _grad_rsub
        return out

    def __mul__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer * other._rawbuffer, parents=(self, other))

        def _grad_mul(grad):
            g1 = _unbroadcast(grad * other, self._rawbuffer.shape)
            g2 = _unbroadcast(grad * self, other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_mul
        return out

    def __rmul__(self, other):
        return shift(other) * self

    def __truediv__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer / other._rawbuffer, parents=(self, other))

        def _grad_div(grad):
            # ensure grad is an ArrayImpl so ops create new nodes
            grad = shift(grad)
            # now these are ArrayImpl operations (traceable)
            g1 = _unbroadcast(grad / other, self._rawbuffer.shape)
            g2 = _unbroadcast(-grad * self / (other * other), other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_div
        return out

    def __rtruediv__(self, other):
        other = shift(other)
        out = ArrayImpl(other._rawbuffer / self._rawbuffer, parents=(other, self))

        def _grad_rdiv(grad):
            grad = shift(grad)
            g1 = _unbroadcast(grad / self, other._rawbuffer.shape)
            g2 = _unbroadcast(-grad * other / (self * self), self._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_rdiv
        return out


    def __pow__(self, other):
        from ..Ops import log
        other = shift(other)
        out = ArrayImpl(self._rawbuffer ** other._rawbuffer, parents=(self, other))

        def _grad_pow(grad):
            g1 = _unbroadcast(grad * other * self ** (other - 1), self._rawbuffer.shape)
            g2 = _unbroadcast(grad * self ** other * log(self + 1e-12), other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_pow
        return out

    def __rpow__(self, other):
        from ..Ops import log
        other = shift(other)
        out = ArrayImpl(other._rawbuffer ** self._rawbuffer, parents=(other, self))

        def _grad_rpow(grad):
            g1 = _unbroadcast(grad * self * other ** (self - 1), other._rawbuffer.shape)
            g2 = _unbroadcast(grad * log(other + 1e-12) * other ** self, self._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_rpow
        return out
    
    def __matmul__(self, other):
        out = ArrayImpl(self._rawbuffer @ other._rawbuffer, parents=(self, other))

        def _grad_matmul(grad):
            a = self
            b = other

            a_val = a._rawbuffer
            b_val = b._rawbuffer
            grad_val = grad._rawbuffer

            # ensure all inputs are at least 2D
            a_exp = a_val if a_val.ndim > 1 else a_val.reshape(1, -1)
            b_exp = b_val if b_val.ndim > 1 else b_val.reshape(-1, 1)
            grad_exp = grad_val if grad_val.ndim > 1 else grad_val.reshape(1, -1)

            grad_a = grad_exp @ b_exp.T
            grad_b = a_exp.T @ grad_exp

            # reshape outputs if inputs were originally 1D
            grad_a = grad_a.flatten() if a_val.ndim == 1 else grad_a
            grad_b = grad_b.flatten() if b_val.ndim == 1 else grad_b

            return ArrayImpl(grad_a), ArrayImpl(grad_b)

    
        out.bwd_fn = _grad_matmul
        return out

    
    def __neg__(self):
        out = ArrayImpl(-self._rawbuffer, parents=(self,))

        def _grad_neg(grad):
            g = _unbroadcast(-grad, self._rawbuffer.shape)
            return (g,)

        out.bwd_fn = _grad_neg
        return out
    
    # Reduction operations
    def sum(self, axis=None, keepdims=False):
        out = ArrayImpl(self._rawbuffer.sum(axis=axis, keepdims=keepdims), parents=(self,))

        def _grad_sum(grad):
            if keepdims:
                grad_expanded = grad._rawbuffer
            else:
                shape = list(self._rawbuffer.shape)
                if axis is None:
                    shape = [1] * self._rawbuffer.ndim
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                grad_expanded = grad._rawbuffer.reshape(shape)
            
            grad_broadcasted = lib.broadcast_to(grad_expanded, self._rawbuffer.shape)
            return (ArrayImpl(grad_broadcasted),)

        out.bwd_fn = _grad_sum
        return out

    def mean(self, axis=None, keepdims=False):
        out = ArrayImpl(self._rawbuffer.mean(axis=axis, keepdims=keepdims), parents=(self,))

        def _grad_mean(grad):
            if axis is None:
                n = self._rawbuffer.size
                shape = [1] * self._rawbuffer.ndim
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                n = lib.prod([self._rawbuffer.shape[ax] for ax in axes])
                shape = list(self._rawbuffer.shape)
                for ax in axes:
                    shape[ax] = 1
            grad_expanded = grad._rawbuffer.reshape(shape if not keepdims else grad._rawbuffer.shape)
            grad_broadcasted = lib.broadcast_to(grad_expanded / n, self._rawbuffer.shape)
            return (ArrayImpl(grad_broadcasted),)

        out.bwd_fn = _grad_mean
        return out

    def var(self, axis=None, keepdims=False):
        # Compute variance along axis
        out = ArrayImpl(self._rawbuffer.var(axis=axis, keepdims=keepdims), parents=(self,))

        def _grad_var(grad):
            if axis is None:
                n = self._rawbuffer.size
                axes = tuple(range(self._rawbuffer.ndim))
                shape = [1] * self._rawbuffer.ndim
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                n = lib.prod([self._rawbuffer.shape[ax] for ax in axes])
                shape = list(self._rawbuffer.shape)
                for ax in axes:
                    shape[ax] = 1
            
            # Compute the mean for broadcasting
            mean = self._rawbuffer.mean(axis=axis, keepdims=True)
            
            # Reshape grad if keepdims is False
            grad_expanded = grad._rawbuffer.reshape(shape if not keepdims else grad._rawbuffer.shape)
            grad_broadcasted = lib.broadcast_to(grad_expanded, self._rawbuffer.shape)
            
            # Gradient formula
            grad_final = (2 / n) * (self._rawbuffer - mean) * grad_broadcasted
            return (ArrayImpl(grad_final),)

        out.bwd_fn = _grad_var
        return out


    # Reshape operation
    def reshape(self, *shape):
        out = ArrayImpl(self._rawbuffer.reshape(shape), parents=(self,))

        def _grad_reshape(grad):
            return (ArrayImpl(grad._rawbuffer.reshape(self._rawbuffer.shape)),)

        out.bwd_fn = _grad_reshape
        return out

    # Transpose operation
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(reversed(range(len(self._rawbuffer.shape))))
        out = ArrayImpl(self._rawbuffer.transpose(axes), parents=(self,))

        def _grad_transpose(grad):
            axes_rev = tuple(lib.argsort(axes))
            return (ArrayImpl(grad._rawbuffer.transpose(axes_rev)),)

        out.bwd_fn = _grad_transpose
        return out

    # Other methods
    def max(self, axis=None, keepdims=False):
        out = ArrayImpl(self._rawbuffer.max(axis=axis, keepdims=keepdims), parents=(self,))

        def _grad_max(grad):
            # Expand grad to ilibut shape if keepdims is False
            if keepdims:
                grad_expanded = grad._rawbuffer
            else:
                shape = list(self._rawbuffer.shape)
                if axis is None:
                    shape = [1] * self._rawbuffer.ndim
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                grad_expanded = grad._rawbuffer.reshape(shape)

            grad_broadcasted = lib.broadcast_to(grad_expanded, self._rawbuffer.shape)
            if axis is None:
                mask = (self._rawbuffer == out._rawbuffer)
            else:
                max_expanded = out._rawbuffer
                if not keepdims:
                    shape = list(self._rawbuffer.shape)
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                    max_expanded = out._rawbuffer.reshape(shape)
                mask = (self._rawbuffer == max_expanded)

            grad_final = grad_broadcasted * mask.astype(self._rawbuffer.dtype)
            return (ArrayImpl(grad_final),)

        out.bwd_fn = _grad_max
        return out


    def min(self, axis=None, keepdims=False):
        out = ArrayImpl(self._rawbuffer.min(axis=axis, keepdims=keepdims), parents=(self,))

        def _grad_min(grad):
            if keepdims:
                grad_expanded = grad._rawbuffer
            else:
                shape = list(self._rawbuffer.shape)
                if axis is None:
                    shape = [1] * self._rawbuffer.ndim
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                grad_expanded = grad._rawbuffer.reshape(shape)

            grad_broadcasted = lib.broadcast_to(grad_expanded, self._rawbuffer.shape)
            if axis is None:
                mask = (self._rawbuffer == out._rawbuffer)
            else:
                max_expanded = out._rawbuffer
                if not keepdims:
                    shape = list(self._rawbuffer.shape)
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in axes:
                        shape[ax] = 1
                    max_expanded = out._rawbuffer.reshape(shape)
                mask = (self._rawbuffer == max_expanded)

            grad_final = grad_broadcasted * mask.astype(self._rawbuffer.dtype)
            return (ArrayImpl(grad_final),)

        out.bwd_fn = _grad_min
        return out
    
    def argmax(self, axis=None, keepdims=False):
        indices = self._rawbuffer.argmax(axis=axis)
        if keepdims:
            if axis is None:
                shape = [1] * self._rawbuffer.ndim
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                shape = list(self._rawbuffer.shape)
                for ax in axes:
                    shape[ax] = 1
            indices = lib.reshape(indices, shape)
        
        out = ArrayImpl(indices, parents=(self,))
        
        def _grad_argmax(grad):
            return (self.zero_like(),)
        
        out.bwd_fn = _grad_argmax
        return out

    def argmin(self, axis=None, keepdims=False):
        indices = self._rawbuffer.argmin(axis=axis)
        
        if keepdims:
            if axis is None:
                shape = [1] * self._rawbuffer.ndim
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                shape = list(self._rawbuffer.shape)
                for ax in axes:
                    shape[ax] = 1
            indices = lib.reshape(indices, shape)
        
        out = ArrayImpl(indices, parents=(self,))
        
        def _grad_argmin(grad):
            return (self.zero_like(),)
        
        out.bwd_fn = _grad_argmin
        return out


    def ones_like(self):
        return ArrayImpl(lib.ones_like(self._rawbuffer))

    def zero_like(self):
        return ArrayImpl(lib.zeros_like(self._rawbuffer))


# Helper function for expanding shapes during mean/var etc.
def shape_expand(shape, axis):
    shape = list(shape)
    if isinstance(axis, int):
        shape[axis] = 1
    elif isinstance(axis, tuple):
        for ax in axis:
            shape[ax] = 1
    return tuple(shape)

class DeviceArray(ArrayImpl):
    __slots__ = ['data']
    def __init__(self, data):
        super().__init__(data._rawbuffer if isinstance(data, ArrayImpl) else data)
