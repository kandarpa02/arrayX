from typing import Sequence
from .errors import ShapeError
from typing import Tuple, Union, Sequence
import uuid

def filler_name():
    return f"anon_{uuid.uuid4().hex[:8]}"

def broadcast_shape(shape1, shape2):
    """
    Given two shapes (tuples/lists of ints), check if they are broadcastable.
    If yes, return the resulting broadcasted shape.
    Else, raise ValueError.
    """
    # Make shapes equal length by prepending 1s
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = (1,) * (len2 - len1) + tuple(shape1)
    elif len2 < len1:
        shape2 = (1,) * (len1 - len2) + tuple(shape2)

    result = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            result.append(max(dim1, dim2))
        else:
            raise ShapeError(f"Shapes {shape1} and {shape2} are not broadcastable.")
    return tuple(result)

def broadcast_to(data, shape):
    from .Tensor.struc import placeholder
    expr = f"lib.broadcast_to({data}, {shape})"
    return placeholder.place(*shape, name=expr)

def _unbroadcast(grad, shape: Sequence):
    """Reduce grad to match shape by summing over broadcasted dimensions."""
    grad_shape = grad.shape
    if grad_shape == shape:
        return grad
    # Add leading ones to shape if needed
    while len(shape) < len(grad_shape):
        shape = (1,) + shape #type:ignore
    axes = tuple(i for i, (g, s) in enumerate(zip(grad_shape, shape)) if s == 1)
    if axes:
        grad_reduced = grad.sum(axis=axes, keepdims=True)
    else:
        grad_reduced = grad
    # Now remove the extra dimensions
    grad_reduced = grad_reduced.reshape(shape)
    return grad_reduced


def reduced_shape(
    shape: Tuple[int, ...],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False
) -> Tuple[int, ...]:
    """
    Compute the resulting shape of a reduction operation.

    Args:
        shape: Original tensor shape as a tuple.
        axis: Axis or axes to reduce. If None, reduce over all axes.
        keepdims: Whether to keep reduced dimensions with size 1.

    Returns:
        Tuple[int, ...]: The resulting shape.
    """
    ndim = len(shape)

    # Normalize axis argument
    if axis is None:
        axes = list(range(ndim))
    elif isinstance(axis, int):
        axes = [axis % ndim]  # handle negative axis
    else:
        axes = [a % ndim for a in axis]

    if keepdims:
        # Replace reduced axes with 1
        return tuple(1 if i in axes else shape[i] for i in range(ndim))
    else:
        # Remove reduced axes
        return tuple(shape[i] for i in range(ndim) if i not in axes)


def data_by_dim(*shape):
    dim = len(shape) if shape != () else 0
    if dim == 0:
        return 'scalar'
    elif dim == 1:
        return 'vector'
    elif dim >=2:
        return 'matrix'
    
