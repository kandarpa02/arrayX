from neo._src.autograd import Policy
from neo.backend import get_xp
from .helpers import define_device
from neo.functions import function


class negative_op(Policy):
    def forward(self, x):
        self.ctx.save()
        device = define_device(x)
        xp = get_xp(device=device)
        return -x
    
    def backward(self, grad):
        return -grad

@function
class absolute(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x)
        xp = get_xp(device=device)
        return xp.abs(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x)
        xp = get_xp(device=device)
        return grad * xp.sign(x)

@function
class sign(Policy):
    def forward(self, x):
        self.ctx.save()
        device = define_device(x)
        xp = get_xp(device=device)
        return xp.sign(x)
    
    def backward(self, grad):
        device = define_device(grad)
        xp = get_xp(device=device)
        return xp.zeros_like(grad)

@function
class exp(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x)
        xp = get_xp(device=device)
        return xp.exp(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x)
        xp = get_xp(device=device)
        return xp.exp(x) * grad

@function
class sqrt(Policy):
    def forward(self, x):
        self.ctx.save(x)
        device = define_device(x)
        xp = get_xp(device=device)
        return xp.sqrt(x)
    
    def backward(self, grad):
        x, = self.ctx.release
        device = define_device(x)
        xp = get_xp(device=device)
        return (0.5 / xp.sqrt(x)) * grad
