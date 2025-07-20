from neo.backend import get_xp
from neo.numpy import array

def randn(shape, device=None, dtype = 'float32'):
    xp = get_xp(device)
    return array(xp.asarray(xp.random.randn(*shape)), dtype=dtype device=device)

def randint(low, high=None, size=None, device=None, dtype = 'float32'):
    xp = get_xp(device)
    return array(xp.asarray(xp.random.randint(low, high, size)), dtype=dtype, device=device)

__all__ = []