from nexnet._src.autograd.FUNCTION_REGISTER import Tracelet
import torch


def max(x, dim=None, keepdim=False):
    out = x.amax(dim=dim, keepdim=keepdim)

    def max_backward(grad):
        # global max (no dim)
        if dim is None:
            mask = (x == x.max())
            num_max = mask.sum()
            return grad * mask.to(x.dtype) / num_max

        # reduce along dim(s)
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        nd = x.dim()
        dims = tuple(d if d >= 0 else d + nd for d in dims)

        max_vals = x.amax(dim=dims, keepdim=True)
        mask = (x == max_vals)

        if not keepdim:
            for d in sorted(dims):
                grad = grad.unsqueeze(d)

        num_max = mask.sum(dim=dims, keepdim=True)
        return (grad * mask.to(x.dtype) / num_max).to(x.dtype)

    with Tracelet() as t:
        t.register(out, (x,), max_backward)

    return out


def mean(x, dim=None, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)

    def mean_backward(grad):
        if dim is None:
            denom = x.numel()
            return grad * torch.ones_like(x) / denom

        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        nd = x.dim()
        dims = tuple(d if d >= 0 else d + nd for d in dims)

        count = 1
        for d in dims:
            count *= x.shape[d]

        if not keepdim:
            for d in sorted(dims):
                grad = grad.unsqueeze(d)

        return grad.expand_as(x) / float(count)

    with Tracelet() as t:
        t.register(out, (x,), mean_backward)

    return out


def sum(x, dim=None, keepdim=False):
    out = x.sum(dim=dim, keepdim=keepdim)

    def sum_backward(grad):
        if dim is None:
            return grad * torch.ones_like(x)

        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        nd = x.dim()
        dims = tuple(d if d >= 0 else d + nd for d in dims)

        if not keepdim:
            for d in sorted(dims):
                grad = grad.unsqueeze(d)

        return grad.expand_as(x)

    with Tracelet() as t:
        t.register(out, (x,), sum_backward)

    return out
