from neo._src.autograd import Policy
from ..math import neolib

class sum_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        self.ctx.save(x, dim, keepdim)
        return neolib.sum(x, dim=dim, keepdim=keepdim)

    def backward(self, grad):
        x, dim, keepdim = self.ctx.release

        if dim is None:
            # grad is scalar, expand to x's shape
            dx = grad.expand_as(x)
        else:
            # if dim is int, convert to tuple for uniform processing
            dims = (dim,) if isinstance(dim, int) else dim
            if not keepdim:
                # unsqueeze grad at all reduced dims
                for d in sorted(dims):
                    grad = neolib.unsqueeze(grad, dim=d)
            dx = grad.expand_as(x)
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
        if dim is None:
            flat_x = neolib.flatten(x)
            max_val = neolib.amax(flat_x)  # torch.amax returns tensor directly
            max_idx = neolib.argmax(flat_x)
            self.ctx.save(x.shape, max_idx)
            return max_val
        else:
            # Use torch.max to get both values and indices
            result = neolib.max(x, dim=dim, keepdim=keepdim)
            values = result.values
            indices = result.indices
            self.ctx.save(x.shape, indices, dim, keepdim)
            return values

    def backward(self, grad):
        ctx_data = self.ctx.release
        
        if len(ctx_data) == 2:
            x_shape, max_idx = ctx_data
            grad_flat = neolib.zeros(x_shape.numel(), dtype=grad.dtype)
            grad_flat[max_idx] = grad
            return grad_flat.reshape(x_shape)

        # Case 2: max along dim
        x_shape, indices, dim, keepdim = ctx_data

        # Ensure grad and indices have same dims for scatter
        if not keepdim:
            grad = neolib.unsqueeze(grad, dim)
            indices = neolib.unsqueeze(indices, dim)

        device = grad.device
        grad_out = neolib.zeros(x_shape, dtype=grad.dtype, device=device)
        indices = indices.to(device)
        grad_out.scatter_(dim, indices, grad)
