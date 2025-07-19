from neo._src.autograd import Policy
from neo.backend import get_xp
from .helpers import define_device

class log_e(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x) 
        xp = get_xp(device=device)
        return xp.log(x)    
    
    def backward(self, grad):
        x, = self.ctx.release
        return grad / x


class log_10(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x) 
        xp = get_xp(device=device)
        return xp.log10(x)

    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x) 
        xp = get_xp(device=device)
        return grad / (x * xp.log(10))
