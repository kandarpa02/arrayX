from neonet.numpy.Array import Array
from neonet.backend import get_xp

def array(data, device='cpu') -> Array:
    xp = get_xp(device)
    return Array(xp.asarray(data), device=device)
