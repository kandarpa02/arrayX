from ..custom_typing import TensorObj
from ..autograd import Node, TapeContext, Policy

from neonet._src.TEMPORARY import xp

# add
class addition(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x + y

    def backward(self, grad):
        x, y = self.ctx.release
        return grad, grad

def add(x:TensorObj, y:TensorObj):
    op = addition()
    out = op.forward(x, y)

    node = Node(out, (x, y), op.backward)
    TapeContext.add_node(node)
    return out


# mul 
class multiplication(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x * y

    def backward(self, grad):
        x, y = self.ctx.release
        return y*grad, x*grad

def mul(x:TensorObj, y:TensorObj):
    op = multiplication()
    out = op.forward(x, y)

    node = Node(out, (x, y), op.backward)
    TapeContext.add_node(node)

    return out