from neonet.backend import get_xp
from neonet.numpy import array

def randn(shape, device=None):
    xp = get_xp(device)
    return array(xp.asarray(xp.random.randn(*shape)), device=device)

def randint(low, high=None, size=None, device=None):
    xp = get_xp(device)
    return array(xp.asarray(xp.random.randint(low, high, size)), device=device)

__all__ = []