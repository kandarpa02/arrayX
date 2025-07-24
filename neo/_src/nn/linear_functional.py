from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib

# def linear_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
#     if X.ndim == 1:
#         X = X.unsqueeze(0)
#     out = X @ w.T + b
#     return out, X, w

# def linear_bwd(grad: neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
#     dx = grad @ w
#     dw = grad.T @ X
#     db = grad.sum(0)
#     return dx, dw, db

@function
class linear(Policy):
    def forward(self, X, w, b):
        self.ctx.save(X, w)
        return neolib.addmm(b, X, w.T)  # out alloc unavoidable here unless passed

    def backward(self, grad):
        X, w = self.ctx.release

        dx = neolib.empty((X.shape[0], w.shape[1]), device=X.device, dtype=X.dtype)
        neolib.matmul(grad, w, out=dx)

        dw = neolib.empty((w.shape[0], X.shape[1]), device=X.device, dtype=X.dtype)
        neolib.matmul(grad.T, X, out=dw)

        db = grad.sum(0)  # sum result is usually small, no need for out=

        del self.ctx
        return dx, dw, db
