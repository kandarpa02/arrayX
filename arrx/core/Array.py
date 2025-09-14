from typing import List, Tuple, Optional, Callable
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np

NumericObject = NDArray

def shift(data):
    if isinstance(data, ArrayImpl):
        return data
    if isinstance(data, ArrayStorage):
        return ArrayImpl(data)
    # If scalar or array, wrap as ArrayStorage then as ArrayImpl
    return ArrayImpl(ArrayStorage(np.array(data)), parents=(), bwd_fn=None)


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


class ArrayStorage:
    __slots__ = ('_rawbuffer',)

    def __init__(self, data: NumericObject):
        self._rawbuffer = data._rawbuffer if isinstance(data, ArrayStorage) else data

    def __repr__(self):
        out = np.array2string(self._rawbuffer, prefix='array(')
        return f'array({out})'

    def __str__(self):
        return str(self._rawbuffer)
    
    def __getitem__(self, i):
        return ArrayStorage(self._rawbuffer[i])

    def numpy(self):
        return self._rawbuffer


class ArrayImpl(ArrayStorage):
    def __init__(self, data, parents=(), bwd_fn=None):
        super().__init__(data._rawbuffer if isinstance(data, ArrayStorage) else data)
        self.parents: Tuple['ArrayImpl', ...] = parents
        self.bwd_fn: Optional[Callable[[ 'ArrayImpl'], Tuple['ArrayImpl', ...]]] = bwd_fn

    # Comparison operations
    def __eq__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer == other._rawbuffer).astype(float))

    def __ne__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer != other._rawbuffer).astype(float))

    def __gt__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer > other._rawbuffer).astype(float))

    def __lt__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer < other._rawbuffer).astype(float))

    def __ge__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer >= other._rawbuffer).astype(float))

    def __le__(self, other):
        other = shift(other)
        return ArrayImpl((self._rawbuffer <= other._rawbuffer).astype(float))

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
        return shift(other) - self

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
            g1 = _unbroadcast(grad / other, self._rawbuffer.shape)
            g2 = _unbroadcast(-grad * self / (other * other), other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_div
        return out

    def __rtruediv__(self, other):
        other = shift(other)
        out = ArrayImpl(other._rawbuffer / self._rawbuffer, parents=(other, self))

        def _grad_rdiv(grad):
            g1 = _unbroadcast(grad / self, other._rawbuffer.shape)
            g2 = _unbroadcast(-grad * other / (self * self), self._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_rdiv
        return out

    def __pow__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer ** other._rawbuffer, parents=(self, other))

        def _grad_pow(grad):
            g1 = _unbroadcast(grad * other * self._rawbuffer ** (other._rawbuffer - 1), self._rawbuffer.shape)
            g2 = _unbroadcast(grad * self._rawbuffer ** other._rawbuffer * np.log(self._rawbuffer + 1e-12), other._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_pow
        return out

    def __rpow__(self, other):
        other = shift(other)
        out = ArrayImpl(other._rawbuffer ** self._rawbuffer, parents=(other, self))

        def _grad_rpow(grad):
            g1 = _unbroadcast(grad * self._rawbuffer * other._rawbuffer ** (self._rawbuffer - 1), other._rawbuffer.shape)
            g2 = _unbroadcast(grad * np.log(other._rawbuffer + 1e-12) * other._rawbuffer ** self._rawbuffer, self._rawbuffer.shape)
            return g1, g2

        out.bwd_fn = _grad_rpow
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
            
            grad_broadcasted = np.broadcast_to(grad_expanded, self._rawbuffer.shape)
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
                n = np.prod([self._rawbuffer.shape[ax] for ax in axes])
                shape = list(self._rawbuffer.shape)
                for ax in axes:
                    shape[ax] = 1
            grad_expanded = grad._rawbuffer.reshape(shape if not keepdims else grad._rawbuffer.shape)
            grad_broadcasted = np.broadcast_to(grad_expanded / n, self._rawbuffer.shape)
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
                n = np.prod([self._rawbuffer.shape[ax] for ax in axes])
                shape = list(self._rawbuffer.shape)
                for ax in axes:
                    shape[ax] = 1
            
            # Compute the mean for broadcasting
            mean = self._rawbuffer.mean(axis=axis, keepdims=True)
            
            # Reshape grad if keepdims is False
            grad_expanded = grad._rawbuffer.reshape(shape if not keepdims else grad._rawbuffer.shape)
            grad_broadcasted = np.broadcast_to(grad_expanded, self._rawbuffer.shape)
            
            # Gradient formula
            grad_final = (2 / n) * (self._rawbuffer - mean) * grad_broadcasted
            return (ArrayImpl(grad_final),)

        out.bwd_fn = _grad_var
        return out


    # Reshape operation
    def reshape(self, newshape):
        out = ArrayImpl(self._rawbuffer.reshape(newshape), parents=(self,))

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
            axes_rev = tuple(np.argsort(axes))
            return (ArrayImpl(grad._rawbuffer.transpose(axes_rev)),)

        out.bwd_fn = _grad_transpose
        return out

    # Other methods
    def max(self, axis=None, keepdims=False):
        out = ArrayImpl(self._rawbuffer.max(axis=axis, keepdims=keepdims), parents=(self,))

        def _grad_max(grad):
            # Expand grad to input shape if keepdims is False
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

            grad_broadcasted = np.broadcast_to(grad_expanded, self._rawbuffer.shape)
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

            grad_broadcasted = np.broadcast_to(grad_expanded, self._rawbuffer.shape)
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

    def ones_like(self):
        return ArrayImpl(np.ones_like(self._rawbuffer))

    def zero_like(self):
        return ArrayImpl(np.zeros_like(self._rawbuffer))

    def numpy(self):
        return self._rawbuffer

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
