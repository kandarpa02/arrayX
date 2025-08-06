from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib

def linear_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b), X, w

def linear_bwd(grad: neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    return grad @ w, grad.T @ X, grad.sum(0)

@function
class linear(Policy):
    def forward(self, X, w, b):
        out, X, w = linear_fwd(X, w, b)
        self.ctx.save(X, w)
        return out

    def backward(self, grad):
        X, w = self.ctx.release
        del self.ctx
        return linear_bwd(grad, X, w)
