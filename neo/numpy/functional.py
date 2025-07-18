from neo.numpy.Array import Array
from neo.backend import get_xp
import numpy as np

def array(data, device=None) -> Array:
    xp = get_xp(device)
    return Array(xp.asarray(data), device=device)

def full(shape, data, device) -> Array:
    xp = get_xp(device)
    return Array(xp.full(shape, data), device=device)


