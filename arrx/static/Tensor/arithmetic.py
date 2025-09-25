from .base import placeholder, vector, matrix

def add(a: vector | matrix, b: vector | matrix):
    return a + b
def sub(a: vector | matrix, b: vector | matrix):
    return a - b
def mul(a: vector | matrix, b: vector | matrix):
    return a * b
def div(a: vector | matrix, b: vector | matrix):
    return a / b
def pow(a: vector | matrix, b: vector | matrix):
    return a ** b


def matmul(a: vector | matrix, b: vector | matrix):
    """
    Perform matrix multiplication between two placeholder objects.

    This function supports multiplication for vectors and matrices:
    - Vector · Vector → scalar (inner product)
    - Matrix · Vector → vector
    - Vector · Matrix → vector
    - Matrix · Matrix → matrix

    Parameters
    ----------
    a : vector or matrix
        The left-hand operand.
    b : vector or matrix
        The right-hand operand.

    Returns
    -------
    placeholder
        The result of the multiplication.

    Raises
    ------
    TypeError
        If either `a` or `b` is not a `vector` or `matrix`.
    """
    if not isinstance(a, vector | matrix) or not isinstance(b, vector | matrix):
        raise TypeError(
            f"function:matmul expects both its arguments to be vector or matrix objects, but found "
            f"{type(a)} and {type(b)}."
        )
    return a @ b


def dot(a: vector | matrix, b: vector | matrix):
    """
    Compute the dot product of two placeholder objects.

    This function is equivalent to `matmul` and follows the same rules:
    - Vector · Vector → scalar
    - Matrix · Vector → vector
    - Vector · Matrix → vector
    - Matrix · Matrix → matrix

    Parameters
    ----------
    a : vector or matrix
        The left-hand operand.
    b : vector or matrix
        The right-hand operand.

    Returns
    -------
    placeholder
        The result of the dot product.

    Notes
    -----
    Provided for API familiarity. Internally, it simply calls `matmul`.
    """
    return matmul(a, b)
