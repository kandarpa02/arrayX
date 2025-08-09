from neo._src.autograd import Policy
from ..math import neolib

class sum_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        self.ctx.save(x, dim, keepdim)
        return neolib.sum(x, dim=dim, keepdim=keepdim)

    def backward(self, grad):
        x, dim, keepdim = self.ctx.release
        if dim is None:
            # grad is a scalar (0-d tensor). Create a tensor like x filled with that scalar.
            dx = neolib.ones_like(x) * grad
        else:
            if not keepdim:
                grad = neolib.unsqueeze(grad, dim=dim)
            dx = neolib.ones_like(x) * grad
        return dx


class mean_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        self.ctx.save(x, dim, keepdim)
        return neolib.mean(x, dim=dim, keepdim=keepdim)

    def backward(self, grad):
        x, dim, keepdim = self.ctx.release
        # compute number of elements averaged per output element (as float)
        if dim is None:
            size = float(int(neolib.prod(neolib.tensor(list(x.shape))).item()))
        else:
            # assume single integer dim for now
            n = int(x.shape[dim])
            size = float(n)
            if not keepdim:
                grad = neolib.unsqueeze(grad, dim=dim)
        dx = neolib.ones_like(x) * (grad / size)
        return dx



class max_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        result = neolib.max(x, dim=dim, keepdim=keepdim)
        values = result.values
        indices = result.indices
        self.ctx.save(x, values, indices, dim, keepdim)
        return values

    # def backward(self, grad):
    #     x, values, indices, dim, keepdim = self.ctx.release
    #     if not keepdim and dim is not None:
    #         values = neolib.unsqueeze(values, dim=dim)
    #         grad = neolib.unsqueeze(grad, dim=dim)
    #     mask = x == values
    #     return grad * mask

    def backward(self, grad):
        x, values, indices, dim, keepdim = self.ctx.release
        if not keepdim and dim is not None:
            values = neolib.unsqueeze(values, dim=dim)
            grad = neolib.unsqueeze(grad, dim=dim)

        mask = x == values
        count = neolib.sum(mask, dim=dim, keepdim=True)  
        grad_per_max = grad / count
        return grad_per_max * mask
