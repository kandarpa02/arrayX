from neo._src.autograd import Policy
import torch

class sum_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        self.ctx.save(x, dim, keepdim)
        return torch.sum(x, dim=dim, keepdim=keepdim)

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
                    grad = torch.unsqueeze(grad, dim=d)
            dx = grad.expand_as(x)
        return dx


class mean_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        self.ctx.save(x, dim, keepdim)
        return torch.mean(x, dim=dim, keepdim=keepdim)

    def backward(self, grad):
        x, dim, keepdim = self.ctx.release
        # compute number of elements averaged per output element (as float)
        if dim is None:
            size = float(int(torch.prod(torch.tensor(list(x.shape))).item()))
        else:
            # assume single integer dim for now
            n = int(x.shape[dim])
            size = float(n)
            if not keepdim:
                grad = torch.unsqueeze(grad, dim=dim)
        dx = torch.ones_like(x) * (grad / size)
        return dx



class max_op(Policy):
    def forward(self, x, dim=None, keepdim=False):
        if dim is None:
            flat_x = torch.flatten(x)
            max_val = torch.amax(flat_x)  # torch.amax returns tensor directly
            max_idx = torch.argmax(flat_x)
            self.ctx.save(x.shape, max_idx)
            return max_val
        else:
            # Use torch.max to get both values and indices
            result = torch.max(x, dim=dim, keepdim=keepdim)
            values = result.values
            indices = result.indices
            self.ctx.save(x.shape, indices, dim, keepdim)
            return values

    def backward(self, grad):
        ctx_data = self.ctx.release
        
        if len(ctx_data) == 2:
            x_shape, max_idx = ctx_data
            grad_flat = torch.zeros(x_shape.numel(), dtype=grad.dtype)
            grad_flat[max_idx] = grad
            return grad_flat.reshape(x_shape)

        # Case 2: max along dim
        x_shape, indices, dim, keepdim = ctx_data

        # Ensure grad and indices have same dims for scatter
        if not keepdim:
            grad = torch.unsqueeze(grad, dim)
            indices = torch.unsqueeze(indices, dim)

        device = grad.device
        grad_out = torch.zeros(x_shape, dtype=grad.dtype, device=device)
        indices = indices.to(device)
        grad_out.scatter_(dim, indices, grad)
