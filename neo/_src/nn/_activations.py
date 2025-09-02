from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy, custom_grad
from neo import neolib
import torch
from neo._torch.lite_tensor import LiteTensor
from typing import Any


def softmax_fwd(x:neolib.Tensor, dim:int):
    return neolib.nn.functional.softmax(x, dim=dim)

def softmax_bwd(out:neolib.Tensor, grad:neolib.Tensor, dim:int):
    dot = (grad * out).sum(dim=dim, keepdim = True)
    return out * (grad - dot)

def sigmoid_fwd(x:neolib.Tensor):
    return x.sigmoid()

def sigmoid_bwd(out:neolib.Tensor, grad:neolib.Tensor):
    return grad * out * (1 - out)

@custom_grad
def _relu(x):
    out =  x.unary_op(lambda x: x.relu_())

    def backward(grad):
        mask = x.data > 0
        return grad.mul(mask)
    
    return out, (x, ), backward

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
    return _relu(x)

@custom_grad
def  _tanh(x):
    out = x.unary_op(lambda x: x.tanh_())

    def backward(self, grad):
        return grad * (1 - (out.data)**2)
    
    return out, (x, ), backward

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
    return _tanh(x)

@custom_grad
def _softmax(x, dim):
    x = x.to_(neolib.float32)
    out = x.unary_op(lambda x, dim: torch.ops.aten.softmax(x, dim=dim), dim=dim)

    def backward(grad, dim=dim, out=out._t):
        return torch.ops.aten._softmax_backward_data(grad, out, dim, grad.dtype)
    
    return out, (x,), backward
    
def softmax(x: LiteTensor, dim: int|Any = -1):
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
    return _softmax(x, dim)

class _sigmoid(Policy):
    def forward(self, x):
        out = sigmoid_fwd(x)
        self.ctx.save(out)
        return out
    
    def backward(self, grad):
        out, = self.ctx.release
        return sigmoid_bwd(out, grad)
    
def sigmoid(x: LiteTensor):
    """
    Applies the sigmoid function

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor of probabilities 0 and 1
    
    """
    return function(_sigmoid)(x)

