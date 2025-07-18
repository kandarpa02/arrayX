from ..custom_typing import TensorObj
from ..autograd import Node, TapeContext, Policy
from ..struc import tensor
from neonet.backend import get_xp

# add
class addition(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x + y

    def backward(self, grad):
        x, y = self.ctx.release
        return tensor(grad), tensor(grad)


# mul 
class multiplication(Policy):
    def forward(self, x, y):
        self.ctx.save(x, y)
        return x * y

    def backward(self, grad):
        x, y = self.ctx.release
        return y*grad, x*grad







def add(x, y):
    op = addition()
    assert x.device == y.device, "Both tensors must be in same device"
    out = op.forward(x, y)

    node = Node(out, (x, y), op.backward)
    TapeContext.add_node(node)
    return out


def mul(x, y):
    op = multiplication()
    assert x.device == y.device, "Both tensors must be in same device"
    out = op.forward(x, y)

    node = Node(out, (x, y), op.backward)
    TapeContext.add_node(node)

    return out