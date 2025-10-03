from ..src.Tensor.base import placeholder, vector, scalar, matrix

def variable(shape=[], name=None) -> placeholder|vector|scalar|matrix:
    out = placeholder.place(*shape, name=name)
    out.grad_required = True
    return out

def constant(shape=[], name=None) -> placeholder|vector|scalar|matrix:
    out = placeholder.place(*shape, name=name)
    return out