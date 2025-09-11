from typing import List, Tuple, Optional
from numpy.typing import NDArray
import numpy as np

NumericObject = NDArray


def shift(data):
    if isinstance(data, ArrayImpl):
        return data
    if isinstance(data, ArrayStorage):
        return ArrayImpl(data)
    # If scalar or array, wrap as ArrayStorage then as ArrayImpl
    return ArrayImpl(ArrayStorage(np.array(data)), parents=(), bwd_fn=None)


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
    def __init__(self, data, parents: Tuple['ArrayImpl', ...] = (), bwd_fn=None):
        super().__init__(data._rawbuffer if isinstance(data, ArrayStorage) else data)
        self.parents = parents
        self.bwd_fn = bwd_fn

    def __add__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer + other._rawbuffer,
                         parents=(self, other),
                         bwd_fn=self._grad_add)
        return out

    def __mul__(self, other):
        other = shift(other)
        out = ArrayImpl(self._rawbuffer * other._rawbuffer,
                        parents=(self, other))

        def _grad_mul(grad):
            return grad * other, grad * self

        out.bwd_fn = _grad_mul
        return out

    def _grad_add(self, grad):
        return grad, grad

    def __radd__(self, other):
        return shift(other) + self

    def __rmul__(self, other):
        return shift(other) * self

    def ones_like(self):
        return ArrayImpl(np.ones_like(self._rawbuffer))

    def zero_like(self):
        return ArrayImpl(np.zeros_like(self._rawbuffer))
