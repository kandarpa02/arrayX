import neo
import math

def xavier_uniform(shape, dtype, device, gain=1.0, key=None):
    """
    Xavier uniform initialization for Neo LiteTensors.

    Args:
        shape (tuple): (fan_in, fan_out)
        dtype: Neo dtype
        device: Neo device
        gain (float): optional scaling factor
        key: RNG key for Neo

    Returns:
        LiteTensor with Xavier-uniform initialization
    """
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * math.sqrt(6 / (fan_in + fan_out))
    return neo.uniform(low=-limit, high=limit, size=shape, dtype=dtype, device=device, key=key)

def xavier_normal(shape, dtype, device, gain=1.0, key=None):
    """
    Xavier (Glorot) normal initialization for Neo LiteTensors.

    Args:
        shape (tuple): (fan_in, fan_out)
        dtype: Neo dtype
        device: Neo device
        gain (float): optional scaling factor
        key: RNG key for Neo

    Returns:
        LiteTensor with Xavier-normal initialization
    """
    fan_in, fan_out = shape[0], shape[1]
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return neo.randn(shape, dtype=dtype, device=device, key=key) * std


def zero_init(shape, dtype, device):
    return neo.zeros(shape, dtype=dtype, device=device)
