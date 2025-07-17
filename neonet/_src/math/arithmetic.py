from ..custom_typing import TensorObj
from ..autograd import Node, Tape, TapeContext, Policy

from neonet._src.TEMPORARY import xp

# add
class addition(Policy):
    @classmethod
    def forward(cls, x, y):
        cls.ctx.save_tensors(x, y)
        return x + y

    @classmethod
    def backward(cls, grad):
        x, y = cls.ctx.saved_tensors
        return 1*grad, 1*grad

def add(x:TensorObj, y:TensorObj):
    op = addition()
    out = op.fwd(x, y)

    node = Node(out, (x, y), op.bwd)
    TapeContext.add_node(node)
    # print(node)
    return out


# mul 

class multiplication(Policy):
    @classmethod
    def forward(cls, x, y):
        cls.ctx.save_tensors(x, y)
        return x * y

    @classmethod
    def backward(cls, grad):
        x, y = cls.ctx.saved_tensors
        return y*grad, x*grad

def mul(x:TensorObj, y:TensorObj):
    op = multiplication()
    out = op.fwd(x, y)

    node = Node(out, (x, y), op.bwd)
    TapeContext.add_node(node)

    return out