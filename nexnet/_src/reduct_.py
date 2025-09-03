from nexnet._src.autograd import Policy
import torch
import math
from typing import Tuple, Union

class sum_op(Policy):
    def forward(self, x: torch.Tensor, dim: Union[int, Tuple[int, ...], None]=None, keepdim: bool=False):
        # store the original x and reduction args for backward
        self.ctx.save(x, dim, keepdim)
        return torch.sum(x, dim=dim, keepdim=keepdim)

    def backward(self, grad: torch.Tensor):
        x, dim, keepdim = self.ctx.release

        # ensure grad is a torch.Tensor on the correct device/dtype
        grad = torch.as_tensor(grad, dtype=x.dtype, device=x.device)

        # reduce over all elems
        if dim is None:
            # broadcast scalar or any grad to x's shape
            return grad * torch.ones_like(x)

        # normalize dims to tuple of positive ints
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        nd = x.dim()
        dims = tuple(d if d >= 0 else d + nd for d in dims)

        # if keepdim==False we need to unsqueeze grad at each reduced dim
        if not keepdim:
            for d in sorted(dims):
                grad = grad.unsqueeze(d)

        # now grad is broadcastable to x
        return grad.expand_as(x)


class mean_op(Policy):
    def forward(self, x: torch.Tensor, dim: Union[int, Tuple[int, ...], None]=None, keepdim: bool=False):
        self.ctx.save(x, dim, keepdim)
        return torch.mean(x, dim=dim, keepdim=keepdim)

    def backward(self, grad: torch.Tensor):
        x, dim, keepdim = self.ctx.release

        grad = torch.as_tensor(grad, dtype=x.dtype, device=x.device)

        if dim is None:
            denom = float(x.numel())
            return (grad * torch.ones_like(x)) / denom

        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        nd = x.dim()
        dims = tuple(d if d >= 0 else d + nd for d in dims)

        # number of elements averaged per output element
        count = 1
        for d in dims:
            count *= x.shape[d]

        if not keepdim:
            for d in sorted(dims):
                grad = grad.unsqueeze(d)

        return (grad.expand_as(x)) / float(count)


class max_op(Policy):
    def forward(self, x: torch.Tensor, dim: Union[int, Tuple[int, ...], None]=None, keepdim: bool=False):
        # store shape/indices for backward
        if dim is None:
            flat_x = torch.flatten(x)
            max_val = torch.amax(flat_x)
            max_idx = torch.argmax(flat_x)
            # save as (shape_tuple, max_idx_tensor)
            self.ctx.save(tuple(x.shape), max_idx)
            return max_val
        else:
            result = torch.max(x, dim=dim, keepdim=keepdim)
            values = result.values
            indices = result.indices
            self.ctx.save(tuple(x.shape), indices, dim, keepdim)
            return values

    def backward(self, grad: torch.Tensor):
        ctx_data = self.ctx.release

        # Case: global max (dim was None)
        if len(ctx_data) == 2:
            x_shape, max_idx = ctx_data
            # ensure grad is a scalar/tensor on correct device/dtype
            grad = torch.as_tensor(grad, dtype=torch.float32 if not hasattr(grad, 'dtype') else grad.dtype)
            device = max_idx.device if isinstance(max_idx, torch.Tensor) else torch.device('cpu')
            dtype = grad.dtype if isinstance(grad, torch.Tensor) else torch.float32

            # Create flat grad and scatter
            numel = 1
            for s in x_shape:
                numel *= s
            grad_flat = torch.zeros(numel, dtype=dtype, device=device)
            # max_idx may be a 0-dim tensor or python int
            if isinstance(max_idx, torch.Tensor):
                idx = int(max_idx.item())
            else:
                idx = int(max_idx)
            grad_flat[idx] = grad
            return grad_flat.reshape(x_shape)

        # Case: max along a dimension
        # ctx_data = (x_shape, indices, dim, keepdim)
        x_shape, indices, dim, keepdim = ctx_data

        # Ensure grad is a tensor on correct device/dtype
        grad = torch.as_tensor(grad, device=indices.device, dtype=grad.dtype if isinstance(grad, torch.Tensor) else torch.float32)

        # If keepdim was False, unsqueeze grad and indices so scatter matches dims
        if not keepdim:
            grad = grad.unsqueeze(dim)
            indices = indices.unsqueeze(dim)

        device = grad.device
        dtype = grad.dtype

        grad_out = torch.zeros(x_shape, dtype=dtype, device=device)
        indices = indices.to(device)
        # scatter grad into grad_out at the positions of indices
        grad_out.scatter_(dim, indices, grad)
        return grad_out
