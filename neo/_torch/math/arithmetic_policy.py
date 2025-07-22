from neo._src.autograd import Node, TapeContext, Policy
from .helpers import define_device
from ..math import neolib

# neo/
# └── math/
#     ├── __init__.py
#     ├── arithmetic.py        # User-facing: add, sub, mul, div, pow
#     ├── unary_policy.py      # Core: negative_op, abs_op, signum_op, exp_op, sqrt_op
#     ├── log_policy.py        # Core: log_e, log_10
#     ├── activation_policy.py # Core: relu_op, sigmoid_op, etc. (later)
#     ├── reductions.py        # sum, mean, max 
#     ├── wrappers.py          # User-facing wrappers for Policy ops
#     └── utils.py             # EPSILON, clamp, safe_log, etc.


# add
class addition(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x + y

    def backward(self, grad):
        x, y = self.ctx.release
        return grad, grad
    

# sub
class subtraction(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x - y
    
    def backward(self, grad):
        x, y = self.ctx.release
        return grad, -grad


# mul 
class multiplication(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x * y

    def backward(self, grad):
        x, y = self.ctx.release
        return y*grad, x*grad
    

# div
class division(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x / y
    
    def backward(self, grad):
        x, y = self.ctx.release
        return 1/y * grad, -x/(y**2) * grad


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
        self.ctx.save()
        device = define_device(x)
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