from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet

def add(x, y):
    out = x + y

    def add_backward(grad):
        grad_x, grad_y = grad, grad

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
    out = x * y

    def mul_backward(grad):
        grad_x = y * grad
        grad_y = x * grad

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
    out = x - y
    sub_backward = lambda grad: (grad, -grad)

    with Tracelet() as t:
        t.register(out, (x, y), sub_backward)

    return out


def div(x, y):
    out = x / y

    def div_backward(grad):
        def sum_to_shape(grad, shape):
            while grad.ndim > len(shape):
                grad = grad.sum(axis=0)
            for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
                if g_dim > s_dim:
                    grad = grad.sum(dim=i, keepdims=True)
            return grad

        grad_x = grad / y
        grad_y = -(x * grad) / (y ** 2)

        grad_x = sum_to_shape(grad_x, x.shape)
        grad_y = sum_to_shape(grad_y, y.shape)

        return grad_x, grad_y

    with Tracelet() as t:
        t.register(out, (x, y), div_backward)

    return out


def power(x, y):
    out = x ** y

    def pow_backward(grad):
        grad_x = (y * x ** (y - 1)) * grad
        grad_y = (out * x.log()) * grad
        return grad_x, grad_y

    with Tracelet() as t:
        t.register(out, (x, y), pow_backward)

    return out


def neg(x):
    out = -x
    neg_backward = lambda grad: -grad

    with Tracelet() as t:
        t.register(out, (x,), neg_backward)

    return out


def matmul(X, Y):
    out = X @ Y
    X_shape, Y_shape = X.shape, Y.shape

    def matmul_backward(grad):
        if X.ndim == 1 and Y.ndim == 1:
            grad_x = grad * Y
            grad_y = grad * X
        elif X.ndim == 2 and Y.ndim == 1:
            grad_x = grad.unsqueeze(1) @ Y.unsqueeze(0)
            grad_y = X.T @ grad
        elif X.ndim == 1 and Y.ndim == 2:
            grad_x = grad @ Y.T
            grad_y = X.unsqueeze(1) @ grad.unsqueeze(0)
        else:
            grad_x = grad @ Y.T
            grad_y = X.T @ grad

        return grad_x.reshape(X_shape), grad_y.reshape(Y_shape)

    with Tracelet() as t:
        t.register(out, (X, Y), matmul_backward)

    return out
