from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib
from ._act._act_fn import *

@function
class relu(Policy):
    def forward(self, X):
        self.ctx.save(X)
        return X.relu_() 

    def backward(self, grad):
        X, = self.ctx.release
        grad[X <= 0] = 0 
        del self.ctx
        return grad



@function
class tanh(Policy):
    def forward(self, x):
        out = tanh_fwd(x)
        self.ctx.save(out)
        return out

    def backward(self, grad):
        out, = self.ctx.release
        return tanh_bwd(out, grad)


@function
class softmax(Policy):
    def forward(self, x, dim):
        x = x.to(neolib.float32)
        out = softmax_fwd(x, dim=dim)
        self.ctx.save(out, dim)
        return out

    
    def backward(self, grad):
        out, dim = self.ctx.release
        return softmax_bwd(out, grad, dim=dim), None
    