from neo.numpy.Array import Array
from ..config import get_dtype

from neo.backend import get_xp
import numpy as np

def array(data, device='cpu', dtype='float32') -> Array:
    xp = get_xp(device)
    return Array(xp.asarray(data), dtype=dtype, device=device)

def full(shape, data, device='cpu', dtype='float32') -> Array:
    xp = get_xp(device)
    return Array(xp.full(shape, data), dtype=dtype, device=device)


