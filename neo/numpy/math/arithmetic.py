from neo._src.autograd import Node, TapeContext, Policy
from neo.backend import get_xp


# add
class addition(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x + y

    def backward(self, grad):
        x, y = self.ctx.release
        return grad, grad


# mul 
class multiplication(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x * y

    def backward(self, grad):
        x, y = self.ctx.release
        return y*grad, x*grad
    