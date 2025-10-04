from ..random_modules import uniform, normal, RNGKey

def xavier_init(shape=[], dtype=None, key=None):
    r = 6/(shape[-1] +shape[-2])
    uni = uniform(*shape, key=key).value

    return uni * r

def zero_init(shape=[], dtype=None):
    import jax
    return jax.numpy.zeros(shape, dtype=dtype)