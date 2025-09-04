from nexnet.functions import function
from nexnet._src.autograd.FUNCTION_REGISTER import custom_grad
import torch
from typing import Any

__all__ = [
    "relu",
    "relu6",
    "leaky_relu",
    "elu",
    "gelu",
    "silu",
    "swish",
    "softplus",
    "softsign",
    "hardtanh",
    "tanh",
    "sigmoid",
    "logsigmoid",
    "softmax",
    "mish",
]

def softmax_fwd(x: torch.Tensor, dim: int):
    return torch.ops.aten.softmax(x, dim=dim)

def softmax_bwd(out: torch.Tensor, grad: torch.Tensor, dim: int):
    dot = (grad * out).sum(dim=dim, keepdim=True)
    return out * (grad - dot)

def sigmoid_fwd(x: torch.Tensor):
    return torch.ops.aten.sigmoid(x)

def sigmoid_bwd(out: torch.Tensor, grad: torch.Tensor):
    return grad * out * (1 - out)

@custom_grad
def _relu(x: torch.Tensor):
    out = torch.ops.aten.relu(x)

    def backward(grad: torch.Tensor):
        mask = x > 0
        return grad * mask
    
    return out, (x,), backward

def relu(x: torch.Tensor):
    """
    Applies the rectified linear unit (ReLU) function element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying ReLU.

    Example:
        >>> relu(torch.tensor([-2.0, 0.0, 3.0]))
        tensor([0.0, 0.0, 3.0])
    """
    return _relu(x)

@custom_grad
def _tanh(x: torch.Tensor):
    out = torch.ops.aten.tanh(x)

    def backward(grad: torch.Tensor):
        return grad * (1 - out**2)
    
    return out, (x,), backward

def tanh(x: torch.Tensor):
    """
    Applies the element-wise hyperbolic tangent function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with tanh applied element-wise.

    Example:
        >>> tanh(torch.tensor([0.0, 1.0]))
        tensor([0.0, 0.7616])
    """
    return _tanh(x)

@custom_grad
def _softmax(x: torch.Tensor, dim: int):
    x = x.to(torch.float32)
    out = torch.ops.aten.softmax(x, dim=dim)

    def backward(grad: torch.Tensor):
        return torch.ops.aten._softmax_backward_data(grad, out, dim, grad.dtype)
    
    return out, (x,), backward
    
def softmax(x: torch.Tensor, dim: int | Any = -1):
    """
    Applies the softmax function along the specified dimension.

    Converts input to float32 before applying softmax.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension to apply softmax along. Required.

    Returns:
        torch.Tensor: Tensor of probabilities summing to 1 along `dim`.

    Example:
        >>> softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
        tensor([0.0900, 0.2447, 0.6652])
    """
    return _softmax(x, dim)

@custom_grad
def _sigmoid(x: torch.Tensor):
    out = sigmoid_fwd(x)
    
    def backward(grad: torch.Tensor):
        return sigmoid_bwd(out, grad)
    return out, (x,), backward
    
def sigmoid(x: torch.Tensor):
    """
    Applies the sigmoid function

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor of probabilities between 0 and 1
    
    """
    return _sigmoid(x)

@custom_grad
def _leaky_relu(x: torch.Tensor, negative_slope: float = 0.01):
    out = torch.ops.aten.leaky_relu(x, negative_slope)

    def backward(grad: torch.Tensor):
        mask_pos = (x > 0).to(grad.dtype)
        mask_neg = (x <= 0).to(grad.dtype) * negative_slope
        return grad * (mask_pos + mask_neg)

    return out, (x,), backward

def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01):
    """
    Applies the LeakyReLU activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.
        negative_slope (float, optional): Slope for negative inputs. Default: 0.01.

    Returns:
        torch.Tensor: Tensor after applying LeakyReLU.
    """
    return _leaky_relu(x, negative_slope)

@custom_grad
def _elu(x: torch.Tensor, alpha: float = 1.0):
    out = torch.ops.aten.elu(x, alpha)

    def backward(grad: torch.Tensor):
        dx = torch.where(out > 0, torch.ones_like(out), out + alpha)
        return grad * dx

    return out, (x,), backward

def elu(x: torch.Tensor, alpha: float = 1.0):
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    Args:
        x (torch.Tensor): Input tensor.
        alpha (float, optional): Scaling for negative inputs. Default: 1.0.

    Returns:
        torch.Tensor: Tensor after applying ELU.
    """
    return _elu(x, alpha)

@custom_grad
def _gelu(x: torch.Tensor, approximate: str = "none"):
    out = torch.ops.aten.gelu(x, approximate=approximate)

    def backward(grad: torch.Tensor):
        return torch.ops.aten.gelu_backward(grad, x, approximate=approximate)

    return out, (x,), backward

def gelu(x: torch.Tensor, approximate: str = "none"):
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    Args:
        x (torch.Tensor): Input tensor.
        approximate (str, optional): Approximation mode ("none" or "tanh"). Default: "none".

    Returns:
        torch.Tensor: Tensor after applying GELU.
    """
    return _gelu(x, approximate)

@custom_grad
def _silu(x: torch.Tensor):
    out = torch.ops.aten.silu(x)

    def backward(grad: torch.Tensor):
        return torch.ops.aten.silu_backward(grad, x)

    return out, (x,), backward

def silu(x: torch.Tensor):
    """
    Applies the Sigmoid Linear Unit (SiLU/Swish) activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying SiLU.
    """
    return _silu(x)

# alias
swish = silu

@custom_grad
def _softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    out = torch.ops.aten.softplus(x, beta, threshold)

    def backward(grad: torch.Tensor):
        return torch.ops.aten.softplus_backward(grad, x, beta, threshold)

    return out, (x,), backward

def softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    """
    Applies the Softplus activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.
        beta (float, optional): Slope parameter. Default: 1.0.
        threshold (float, optional): Numerical stability threshold. Default: 20.0.

    Returns:
        torch.Tensor: Tensor after applying Softplus.
    """
    return _softplus(x, beta, threshold)

@custom_grad
def _softsign(x: torch.Tensor):
    out = x / (1 + torch.ops.aten.abs(x))

    def backward(grad: torch.Tensor):
        denom = 1 + torch.abs(x)
        grad_x = grad / (denom * denom)
        return grad_x

    return out, (x,), backward

def softsign(x: torch.Tensor):
    """
    Applies the Softsign activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying Softsign.
    """
    return _softsign(x)

@custom_grad
def _hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    out = torch.ops.aten.hardtanh(x, min_val, max_val)

    def backward(grad: torch.Tensor):
        mask = (x >= min_val) & (x <= max_val)
        return grad * mask.to(grad.dtype)

    return out, (x,), backward

def hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    """
    Applies the HardTanh activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.
        min_val (float, optional): Minimum threshold. Default: -1.0.
        max_val (float, optional): Maximum threshold. Default: 1.0.

    Returns:
        torch.Tensor: Tensor after applying HardTanh.
    """
    return _hardtanh(x, min_val, max_val)

@custom_grad
def _relu6(x: torch.Tensor):
    out = torch.ops.aten.hardtanh(x, 0.0, 6.0)

    def backward(grad: torch.Tensor):
        mask = (x >= 0.0) & (x <= 6.0)
        return grad * mask.to(grad.dtype)

    return out, (x,), backward

def relu6(x: torch.Tensor):
    """
    Applies the ReLU6 activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying ReLU6 (clamped between 0 and 6).
    """
    return _relu6(x)

@custom_grad
def _logsigmoid(x: torch.Tensor):
    out = torch.ops.aten.log_sigmoid(x)

    def backward(grad: torch.Tensor):
        return grad * torch.ops.aten.sigmoid(-x)

    return out, (x,), backward

def logsigmoid(x: torch.Tensor):
    """
    Applies the Log-Sigmoid activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying Log-Sigmoid.
    """
    return _logsigmoid(x)

@custom_grad
def _mish(x: torch.Tensor):
    out = torch.ops.aten.mish(x)

    def backward(grad: torch.Tensor):
        return torch.ops.aten.mish_backward(grad, x)

    return out, (x,), backward

def mish(x: torch.Tensor):
    """
    Applies the Mish activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying Mish.
    """
    return _mish(x)