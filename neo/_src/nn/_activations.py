from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib
from neo._torch.lite_tensor import LiteTensor


def tanh_fwd(x: neolib.Tensor):
    return neolib.tanh(x)

def tanh_bwd(out: neolib.Tensor, grad: neolib.Tensor):
    return grad * (1 - out * out)

def softmax_fwd(x:neolib.Tensor, dim:int):
    return neolib.nn.functional.softmax(x, dim=dim)

def softmax_bwd(out:neolib.Tensor, grad:neolib.Tensor, dim:int):
    dot = (grad * out).sum(dim=dim, keepdim = True)
    return out * (grad - dot)


class _relu(Policy):
    def forward(self, X):
        self.ctx.save(X)
        return X.relu_()

    def backward(self, grad):
        X, = self.ctx.release
        mask = X > 0
        return grad.mul(mask)

def relu(x: LiteTensor):
    """
    Applies the rectified linear unit (ReLU) function element-wise.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor after applying ReLU in-place.

    Example:
        >>> relu(LiteTensor([-2.0, 0.0, 3.0]))
        LiteTensor([0.0, 0.0, 3.0])
    """
    return function(_relu)(x)


class _tanh(Policy):
    def forward(self, x):
        out = tanh_fwd(x)
        self.ctx.save(out)
        return out

    def backward(self, grad):
        out, = self.ctx.release
        return tanh_bwd(out, grad)

def tanh(x: LiteTensor):
    """
    Applies the element-wise hyperbolic tangent function.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor with tanh applied element-wise.

    Example:
        >>> tanh(LiteTensor([0.0, 1.0]))
        LiteTensor([0.0, 0.7616])
    """
    return function(_tanh)(x)


class _softmax(Policy):
    def forward(self, x, dim):
        x = x.to(neolib.float32)
        out = softmax_fwd(x, dim=dim)
        self.ctx.save(out, dim)
        return out

    
    def backward(self, grad):
        out, dim = self.ctx.release
        return softmax_bwd(out, grad, dim=dim), None
    
def softmax(x: LiteTensor, dim: int = None):
    """
    Applies the softmax function along the specified dimension.

    Converts input to float32 before applying softmax.

    Args:
        x (LiteTensor): Input tensor.
        dim (int): The dimension to apply softmax along. Required.

    Returns:
        LiteTensor: Tensor of probabilities summing to 1 along `dim`.

    Example:
        >>> softmax(LiteTensor([1.0, 2.0, 3.0]), dim=0)
        LiteTensor([0.0900, 0.2447, 0.6652])
    """
    return function(_softmax)(x, dim)
