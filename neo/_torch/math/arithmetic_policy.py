from neo._src.autograd import Node, TapeContext, Policy
from ..math import neolib

class addition(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x + y

    def backward(self, grad):
        x, y = self.ctx.release
        grad_x = grad
        grad_y = grad
        return grad_x, grad_y


class subtraction(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x - y

    def backward(self, grad):
        x, y = self.ctx.release
        grad_x = grad
        grad_y = -grad
        return grad_x, grad_y


class multiplication(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x * y

    def backward(self, grad):
        x, y = self.ctx.release
        grad_x = y * grad
        grad_y = x * grad
        return grad_x, grad_y


class division(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x / y

    def backward(self, grad):
        x, y = self.ctx.release
        
        def sum_to_shape(grad, shape):
            while grad.ndim > len(shape):
                grad = grad.sum(axis=0)
            for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
                if g_dim > s_dim:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        
        grad_x = grad / y
        grad_y = -(x * grad) / (y ** 2)
        
        grad_x = sum_to_shape(grad_x, x.shape)
        grad_y = sum_to_shape(grad_y, y.shape)
        
        return grad_x, grad_y


class power_op(Policy):
    def forward(self, x, y):
        z = x ** y
        self.ctx.save(x, y, z)
        return z
    
    def backward(self, grad):
        x, y, z = self.ctx.release
        return (y * x ** (y-1)) * grad, (z * neolib.log(x)) * grad
    

class negative(Policy):
    def forward(self, x):
        return -x
    def backward(self, grad):
        return -grad
    

class matmul_op(Policy):
    def forward(self, X, Y):
        self.ctx.save(X, Y)
        return X @ Y

    def backward(self, grad):
        X, Y = self.ctx.release

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
