from neo._src.autograd import Policy
from neo.backend import get_xp
from .helpers import define_device

class log_e(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return self.lib.log(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad / x


class log_10(Policy):
    def forward(self, x):
        self.ctx.save(x)
        return self.lib.log10(x)

    def backward(self, grad):
        x, = self.ctx.release
        return grad / (x * self.lib.log(10))
