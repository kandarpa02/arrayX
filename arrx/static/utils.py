from typing import Sequence

def _unbroadcast(grad, shape: Sequence):
    """Reduce grad to match shape by summing over broadcasted dimensions."""
    grad_shape = grad.shape
    if grad_shape == shape:
        return grad
    # Add leading ones to shape if needed
    while len(shape) < len(grad_shape):
        shape = (1,) + shape
    axes = tuple(i for i, (g, s) in enumerate(zip(grad_shape, shape)) if s == 1)
    if axes:
        grad_reduced = grad.sum(axis=axes, keepdims=True)
    else:
        grad_reduced = grad
    # Now remove the extra dimensions
    grad_reduced = grad_reduced.reshape(shape)
    return grad_reduced
