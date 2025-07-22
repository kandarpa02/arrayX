from neo._src.autograd import Policy
from neo.backend import get_xp
from .helpers import define_device
from neo.functions import function

@function
class log(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return self.lib.log(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad / x

@function
class log10(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return self.lib.log10(x)

    def backward(self, grad):
        x, = self.ctx.release
        return grad / (x * self.lib.log(10))
