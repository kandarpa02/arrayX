from ..Core.Array import ArrayImpl, _unbroadcast
from .basic_math import *
from .array_builders import *
from arrx import lib

def sum(array: 'ArrayImpl', axis=None, keepdims=False):
    """
    Computes the sum of array elements over the specified axis or axes.

    Parameters:
        array (ArrayImpl): The input array whose elements are to be summed.
        axis (int or tuple of ints, optional): Axis or axes along which to sum.
            If None, sum over all elements. Defaults to None.
        keepdims (bool, optional): Whether to retain reduced dimensions with size 1.
            Defaults to False.

    Returns:
        ArrayImpl: A new ArrayImpl instance containing the summed result,
            with broadcasting-aware gradients for backward propagation.

    Notes:
        - Supports broadcasting rules during gradient computation.
        - Compatible with higher-order derivatives.
        - Works similarly to numpy.sum and jax.numpy.sum.
    """
    return array.sum(axis=axis, keepdims=keepdims)


def mean(array: 'ArrayImpl', axis=None, keepdims=False):
    """
    Computes the arithmetic mean of array elements over the specified axis or axes.

    Parameters:
        array (ArrayImpl): The input array to compute the mean from.
        axis (int or tuple of ints, optional): Axis or axes along which to compute the mean.
            If None, computes the mean over all elements. Defaults to None.
        keepdims (bool, optional): Whether to retain reduced dimensions with size 1.
            Defaults to False.

    Returns:
        ArrayImpl: A new ArrayImpl instance containing the mean values,
            with broadcasting-aware gradients for backward propagation.

    Notes:
        - Supports broadcasting and shape expansion during gradient computation.
        - Compatible with higher-order derivatives.
        - Mirrors behavior of numpy.mean and jax.numpy.mean.
    """
    return array.mean(axis=axis, keepdims=keepdims)


def var(array: 'ArrayImpl', axis=None, keepdims=False):
    """
    Computes the variance of array elements over the specified axis or axes.

    Parameters:
        array (ArrayImpl): The input array whose variance is to be computed.
        axis (int or tuple of ints, optional): Axis or axes along which to compute the variance.
            If None, computes variance over all elements. Defaults to None.
        keepdims (bool, optional): Whether to retain reduced dimensions with size 1.
            Defaults to False.

    Returns:
        ArrayImpl: A new ArrayImpl instance containing the variance values,
            with broadcasting-aware gradients for backward propagation.

    Notes:
        - Variance is computed as the average squared deviation from the mean.
        - Supports broadcasting during gradient computation.
        - Compatible with higher-order derivatives.
        - Similar in functionality to numpy.var and jax.numpy.var.
    """
    return array.var(axis=axis, keepdims=keepdims)


def max(array: 'ArrayImpl', axis=None, keepdims=False):
    """
    Computes the maximum of array elements over the specified axis or axes.

    Parameters:
        array (ArrayImpl): The input array from which to find maximum values.
        axis (int or tuple of ints, optional): Axis or axes along which to compute the maximum.
            If None, computes the global maximum over all elements. Defaults to None.
        keepdims (bool, optional): Whether to retain reduced dimensions with size 1.
            Defaults to False.

    Returns:
        ArrayImpl: A new ArrayImpl instance containing the maximum values,
            with broadcasting-aware gradients for backward propagation.

    Notes:
        - Uses masking during backward computation to propagate gradients only to max locations.
        - Supports broadcasting during gradient computation.
        - Compatible with higher-order derivatives.
        - Similar to numpy.max and jax.numpy.max.
    """
    return array.max(axis=axis, keepdims=keepdims)


def min(array: 'ArrayImpl', axis=None, keepdims=False):
    """
    Computes the minimum of array elements over the specified axis or axes.

    Parameters:
        array (ArrayImpl): The input array from which to find minimum values.
        axis (int or tuple of ints, optional): Axis or axes along which to compute the minimum.
            If None, computes the global minimum over all elements. Defaults to None.
        keepdims (bool, optional): Whether to retain reduced dimensions with size 1.
            Defaults to False.

    Returns:
        ArrayImpl: A new ArrayImpl instance containing the minimum values,
            with broadcasting-aware gradients for backward propagation.

    Notes:
        - Uses masking during backward computation to propagate gradients only to min locations.
        - Supports broadcasting during gradient computation.
        - Compatible with higher-order derivatives.
        - Similar to numpy.min and jax.numpy.min.
    """
    return array.min(axis=axis, keepdims=keepdims)


def reshape(array: 'ArrayImpl', *newshape):
    """
    Returns a new array with the same data but reshaped to the specified dimensions.

    Parameters:
        array (ArrayImpl): The input array to reshape.
        newshape (tuple of ints): The desired shape of the output array.

    Returns:
        ArrayImpl: A new ArrayImpl instance with the specified shape,
            with gradients that correctly reshape during backward propagation.

    Notes:
        - Reshaping preserves data layout and broadcasting behavior.
        - Supports higher-order gradients.
        - Similar in functionality to numpy.reshape and jax.numpy.reshape.
    """
    return array.reshape(*newshape)


def transpose(array: 'ArrayImpl', axes=None):
    """
    Returns a transposed view of the input array, permuting its dimensions.

    Parameters:
        array (ArrayImpl): The input array to transpose.
        axes (tuple of ints, optional): The permutation of axes. If None, reverses the axes.
            Defaults to None.

    Returns:
        ArrayImpl: A new ArrayImpl instance with axes transposed,
            with gradients that correctly transpose during backward propagation.

    Notes:
        - Supports arbitrary axis permutation.
        - Compatible with broadcasting during gradient computation.
        - Works with higher-order derivatives.
        - Similar to numpy.transpose and jax.numpy.transpose.
    """
    return array.transpose(axes)


def ones_like(array: 'ArrayImpl'):
    """
    Creates a new array filled with ones having the same shape as the input array.

    Parameters:
        array (ArrayImpl): The input array whose shape is used for the output.

    Returns:
        ArrayImpl: A new ArrayImpl instance filled with ones,
            with no gradients associated (constant output).

    Notes:
        - Useful for constructing gradient-compatible constants.
        - Similar to numpy.ones_like and jax.numpy.ones_like.
    """
    return array.ones_like()


def zero_like(array: 'ArrayImpl'):
    """
    Creates a new array filled with zeros having the same shape as the input array.

    Parameters:
        array (ArrayImpl): The input array whose shape is used for the output.

    Returns:
        ArrayImpl: A new ArrayImpl instance filled with zeros,
            with no gradients associated (constant output).

    Notes:
        - Useful for constructing gradient-compatible constants.
        - Similar to numpy.zeros_like and jax.numpy.zeros_like.
    """
    return array.zero_like()

def argmax(array: 'ArrayImpl', axis=None, keepdims=False):
    return array.argmax(axis, keepdims)

def argmin(array: 'ArrayImpl', axis=None, keepdims=False):
    return array.argmin(axis, keepdims) 

def where(condition, x, y):
    condition = shift(condition)
    x = shift(x)
    y = shift(y)
    out = ArrayImpl(lib.where(condition._rawbuffer, x._rawbuffer, y._rawbuffer), parents=(condition, x, y))

    def _grad_where(grad):
        grad = shift(grad)
        grad_condition = None  # where gradient typically ignored for condition
        grad_x = _unbroadcast(grad * condition, x._rawbuffer.shape)
        grad_y = _unbroadcast(grad * (1 - condition), y._rawbuffer.shape)
        return grad_condition, grad_x, grad_y

    out.bwd_fn = _grad_where
    return out
