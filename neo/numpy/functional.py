from neo.numpy.Array import Array
from ..config import get_dtype

from neo.backend import get_xp
import numpy as np


def define_device(x):
    import numpy as np
    device = 'cpu'
    if not isinstance(x, np.ndarray):
        device = 'cuda'
    return device


def array(data, device='cpu', dtype='float32') -> Array:
    xp = get_xp(device)
    return Array(xp.asarray(data), dtype=dtype, device=device)

def full(shape, data, device='cpu', dtype='float32') -> Array:
    xp = get_xp(device)
    return Array(xp.full(shape, data), dtype=dtype, device=device)


def reshape(data: Array, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    
    xp = get_xp(define_device(data))
    return Array(xp.reshape(data.value, shape), device=data.device, dtype=data.dtype)

def argmax(data: Array, axis=None):
    xp = get_xp(define_device(data.value))
    out = xp.argmax(data.value, axis=axis) 
    return Array(out, device=data.device, dtype="int32")
