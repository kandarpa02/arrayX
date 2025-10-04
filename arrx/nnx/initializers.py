import jax.numpy as jnp
from ..random_modules import uniform, normal, RNGKey

def xavier_init(shape, dtype=jnp.float32, key=None):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = (6.0 / (fan_in + fan_out)) ** 0.5
    return uniform(*shape, a=-limit, b=limit, key=key).value.astype(dtype) #type:ignore

def xavier_normal(shape, dtype=jnp.float32, key=None):
    fan_in, fan_out = shape[-2], shape[-1]
    std = (2.0 / (fan_in + fan_out)) ** 0.5
    return normal(*shape, mu=0.0, sigma=std, key=key).value.astype(dtype) #type:ignore

def zero_init(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)

