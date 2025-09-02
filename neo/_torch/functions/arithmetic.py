from neo._src.autograd.FUNCTION_REGISTER import Tracelet
from neo.functions import function

def add(x, y):
    out = x.binary_op(y, (lambda a, b: a + b))
    def add_backward(grad):
        grad_x = grad
        grad_y = grad

        if grad_x.shape != x.shape:
            reduce_dims = tuple(i for i, (gx, xx) in enumerate(zip(grad_x.shape, x.shape)) if gx != xx)
            grad_x = grad_x.sum(dim=reduce_dims, keepdim=True)

        if grad_y.shape != y.shape:
            reduce_dims = tuple(i for i, (gy, yy) in enumerate(zip(grad_y.shape, y.shape)) if gy != yy)
            grad_y = grad_y.sum(dim=reduce_dims, keepdim=True)

        return grad_x, grad_y
    
    with Tracelet() as t:
        t.register(out, (x, y), add_backward)

    return out


def mul(x, y):
    out = x.binary_op(y, (lambda a, b: a * b))
    def mul_backward(grad):
        grad_x = y._t*grad
        grad_y = x._t*grad

        if grad_x.shape != x.shape:
            reduce_dims = tuple(i for i, (gx, xx) in enumerate(zip(grad_x.shape, x.shape)) if gx != xx)
            grad_x = grad_x.sum(dim=reduce_dims, keepdim=True)

        if grad_y.shape != y.shape:
            reduce_dims = tuple(i for i, (gy, yy) in enumerate(zip(grad_y.shape, y.shape)) if gy != yy)
            grad_y = grad_y.sum(dim=reduce_dims, keepdim=True)

        return grad_x, grad_y
    
    with Tracelet() as t:
        t.register(out, (x, y), mul_backward)

    return out


def sub(x, y):
    out = x.binary_op(y, (lambda a, b: a - b))
    sub_backward = lambda grad: (grad, -grad)
    with Tracelet() as t:
        t.register(out, (x, y), sub_backward)
    return out


def div(x, y):
    out = x.binary_op(y, lambda a, b: a/b)
    def div_backward(grad):
        def sum_to_shape(grad, shape):
            while grad.ndim > len(shape):
                grad = grad.sum(axis=0)
            for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
                if g_dim > s_dim:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        
        grad_x = grad / y._t
        grad_y = -(x._t * grad) / (y._t ** 2)
        
        grad_x = sum_to_shape(grad_x, x.shape)
        grad_y = sum_to_shape(grad_y, y.shape)
        
        return grad_x, grad_y
    
    with Tracelet() as t:
        t.register(out, (x, y), div_backward)

    return out


def power(x, y):
    out = x.binary_op(y, (lambda a, b: a ** b))
    def pow_backward(grad):
        return (y * x ** (y-1)) * grad, (out._t * x._t.log()) * grad

    with Tracelet() as t:
        t.register(out, (x, y), pow_backward)
    
    return out

def neg(x):
    out = x.unary_op(lambda x: -x)
    neg_backward = lambda grad: -grad
    with Tracelet() as t:
        t.register(out, (x, ), neg_backward)
    return out


def matmul(X, Y):
    out = X.binary_op(Y, (lambda a, b: a @ b))
    _x = X._t
    _y = Y._t
    def matmul_backward(grad):
        X = _x
        Y = _y
        X_shape = X.shape
        Y_shape = Y.shape
        
        # Match PyTorch matmul rules for grad shapes
        if X.ndim == 1 and Y.ndim == 1:
            # dot product case -> scalar grad
            grad_x = grad * Y
            grad_y = grad * X
        elif X.ndim == 2 and Y.ndim == 1:
            # matrix @ vector -> vector grad
            grad_x = grad.unsqueeze(1) @ Y.unsqueeze(0)  # outer product
            grad_y = X.T @ grad
        elif X.ndim == 1 and Y.ndim == 2:
            # vector @ matrix -> vector grad
            grad_x = grad @ Y.T
            grad_y = X.unsqueeze(1) @ grad.unsqueeze(0)
        else:
            # matrix @ matrix
            grad_x = grad @ Y.T
            grad_y = X.T @ grad

        # Ensure same shapes as original inputs
        return grad_x.reshape(X_shape), grad_y.reshape(Y_shape)
    
    with Tracelet() as t:
        t.register(out, (X, Y), matmul_backward)
    
    return out
