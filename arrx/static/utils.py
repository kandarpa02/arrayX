from typing import Sequence
from .errors import ShapeError
from typing import Tuple, Union, Sequence
import uuid
from arrx import lib

def variable(shape=[], name=None):
    from .Tensor.base import placeholder
    return placeholder.place(*shape, name=name)

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

def matmul_shape(shape1, shape2):
    """
    Infer the result shape of a matmul operation given two input shapes.
    """
    # Handle 1D special cases
    if len(shape1) == 1 and len(shape2) == 1:
        # (n,) @ (n,) -> ()
        if shape1[0] != shape2[0]:
            raise ShapeError(f"Incompatible dimensions for dot: {shape1}, {shape2}")
        return ()

    if len(shape1) == 1 and len(shape2) == 2:
        # (n,) @ (n, m) -> (m,)
        if shape1[0] != shape2[0]:
            raise ShapeError(f"Incompatible dimensions: {shape1}, {shape2}")
        return (shape2[1],)

    if len(shape1) == 2 and len(shape2) == 1:
        # (m, n) @ (n,) -> (m,)
        if shape1[1] != shape2[0]:
            raise ShapeError(f"Incompatible dimensions: {shape1}, {shape2}")
        return (shape1[0],)

    # General case (>=2D each)
    if shape1[-1] != shape2[-2]:
        raise ShapeError(f"Incompatible core dims: {shape1[-1]} vs {shape2[-2]}")

    batch1 = shape1[:-2]
    batch2 = shape2[:-2]
    batch = broadcast_shape(batch1, batch2)

    return batch + (shape1[-2], shape2[-1])

def reshape_shape(input_shape, new_shape):
    def prod(a):
        x = 1 
        for i in a:
            x*= i
        return x
    input_size = prod(input_shape)
    new_shape = list(new_shape)

    # Count -1s
    neg_count = new_shape.count(-1)
    if neg_count > 1:
        raise ValueError("Only one dimension can be -1")

    # Compute product of specified dims (ignoring -1)
    known_product = 1
    for dim in new_shape:
        if dim != -1:
            known_product *= dim

    if neg_count == 1:
        if input_size % known_product != 0:
            raise ValueError("Cannot infer dimension: sizes don't match")
        # Replace -1 with inferred dimension
        for i, dim in enumerate(new_shape):
            if dim == -1:
                new_shape[i] = input_size // known_product
                break

    # Final check
    if prod(new_shape) != input_size:
        raise ValueError(f"Shape mismatch: cannot reshape {input_shape} to {tuple(new_shape)}")

    return tuple(new_shape)

def transpose_shape(shape, axes):
    import numpy as np
    dummy = np.empty(shape, dtype=np.ubyte)
    return dummy.transpose(axes).shape

def broadcast_to(data, shape):
    from .Tensor.base import placeholder
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
    
