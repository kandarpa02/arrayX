from neo._src.autograd import Policy
from .helpers import define_device
from ..math import neolib

class absolute_op(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x)
        return neolib.abs(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x)
        return grad * neolib.sign(x)

class signum_op(Policy):
    def forward(self, x):
        self.ctx.save()
        device = define_device(x)
        return neolib.sign(x)
    
    def backward(self, grad):
        device = define_device(grad)
        return neolib.zeros_like(grad)

class exponential_op(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x)
        return neolib.exp(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x)
        return neolib.exp(x) * grad

class sqrt_op(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x)
        return neolib.sqrt(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x)
        return (0.5 / neolib.sqrt(x)) * grad
