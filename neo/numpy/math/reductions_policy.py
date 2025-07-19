from neo._src.autograd import Policy
from neo.backend import get_xp
from .helpers import define_device


class sum_op(Policy):
    def forward(self, x, axis=None, keepdims=False):
        self.ctx.save(x, axis, keepdims)
        xp = get_xp(device=define_device(x))
        return xp.sum(x, axis=axis, keepdims=keepdims)
    def backward(self, grad):
        from neo.backend import get_xp
        x, axis, keepdims = self.ctx.release
        xp = get_xp(device=define_device(x))
        if axis is None:
            return xp.ones_like(x) * grad
        if not keepdims:
            grad = xp.expand_dims(grad, axis=axis)
        return xp.ones_like(x) * grad


class mean_op(Policy):
    def forward(self, x, axis=None, keepdims=False):
        self.ctx.save(x, axis, keepdims)
        xp = get_xp(device=define_device(x))
        return xp.mean(x, axis=axis, keepdims=keepdims)
    def backward(self, grad):
        x, axis, keepdims = self.ctx.release
        xp = get_xp(device=define_device(x))
        N = xp.prod(xp.array(x.shape if axis is None else xp.array(x.shape)[axis]))
        if axis is not None and not keepdims:
            grad = xp.expand_dims(grad, axis=axis)
        return xp.ones_like(x) * grad / N


class max_op(Policy):
    def forward(self, x, axis=None, keepdims=False):
        xp = get_xp(device=define_device(x))
        out = xp.max(x, axis=axis, keepdims=keepdims)
        self.ctx.save(x, out, axis, keepdims)
        return out
    def backward(self, grad):
        x, out, axis, keepdims = self.ctx.release
        xp = get_xp(device=define_device(x))
        if not keepdims and axis is not None:
            out = xp.expand_dims(out, axis=axis)
            grad = xp.expand_dims(grad, axis=axis)
        mask = x == out
        return grad * mask
