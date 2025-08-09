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
        return grad_x, grad_x


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
        return grad @ Y.T, X.T @ grad