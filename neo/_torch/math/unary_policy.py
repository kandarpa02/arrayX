from neo._src.autograd import Policy
from ..math import neolib

class absolute_op(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return neolib.abs(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad * neolib.sign(x)

class signum_op(Policy):
    def forward(self, x):
        self.ctx.save()
        return neolib.sign(x)
    
    def backward(self, grad):
        return neolib.zeros_like(grad)

class exponential_op(Policy):
    def forward(self, x):
        out = neolib.exp(x)
        self.ctx.save(out)
        return out
    
    def backward(self, grad):
        out, = self.ctx.release
        return out * grad


class sqrt_op(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return neolib.sqrt(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return (0.5 / neolib.sqrt(x)) * grad
