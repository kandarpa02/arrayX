from neonet.numpy.Array import Array
from neonet.backend import get_xp

def array(data, device=None) -> Array:
    xp = get_xp(device)
    return Array(xp.asarray(data), device=device)
