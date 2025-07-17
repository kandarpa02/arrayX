# neonet/random.py
from neonet._src.struc import tensor
from neonet.backend import get_xp

def randn(shape, device=None):
    xp = get_xp(device)
    return tensor(xp.asarray(xp.random.randn(*shape)))

def randint(low, high=None, size=None, device=None):
    xp = get_xp(device)
    return tensor(xp.asarray(xp.random.randint(low, high, size)))
