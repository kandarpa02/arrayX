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


@custom_grad
def _leaky_relu(x, negative_slope: float = 0.01):
    out = x.unary_op(lambda t: torch.ops.aten.leaky_relu(t, negative_slope, inplace=False))

    def backward(grad, x=x.data, negative_slope=negative_slope):
        mask_pos = (x > 0).to(grad.dtype)
        mask_neg = (x <= 0).to(grad.dtype) * negative_slope
        return grad * (mask_pos + mask_neg)

    return out, (x,), backward

def leaky_relu(x: LiteTensor, negative_slope: float = 0.01):
    """
    Applies the LeakyReLU activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.
        negative_slope (float, optional): Slope for negative inputs. Default: 0.01.

    Returns:
        LiteTensor: Tensor after applying LeakyReLU.
    """
    return _leaky_relu(x, negative_slope)


@custom_grad
def _elu(x, alpha: float = 1.0):
    out = x.unary_op(lambda t: torch.ops.aten.elu(t, alpha, inplace=False))

    def backward(grad, out=out.data, alpha=alpha):
        dx = torch.where(out > 0, torch.ones_like(out), out + alpha)
        return grad * dx

    return out, (x,), backward

def elu(x: LiteTensor, alpha: float = 1.0):
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    Args:
        x (LiteTensor): Input tensor.
        alpha (float, optional): Scaling for negative inputs. Default: 1.0.

    Returns:
        LiteTensor: Tensor after applying ELU.
    """
    return _elu(x, alpha)


@custom_grad
def _gelu(x, approximate: str = "none"):
    out = x.unary_op(lambda t: torch.ops.aten.gelu(t, approximate))

    def backward(grad, x=x.data, approximate=approximate):
        return torch.ops.aten.gelu_backward(grad, x, approximate)

    return out, (x,), backward

def gelu(x: LiteTensor, approximate: str = "none"):
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    Args:
        x (LiteTensor): Input tensor.
        approximate (str, optional): Approximation mode ("none" or "tanh"). Default: "none".

    Returns:
        LiteTensor: Tensor after applying GELU.
    """
    return _gelu(x, approximate)


@custom_grad
def _silu(x):
    out = x.unary_op(lambda t: torch.ops.aten.silu(t))

    def backward(grad, out=out.data, x=x.data):
        return torch.ops.aten.silu_backward(grad, out, x)

    return out, (x,), backward

def silu(x: LiteTensor):
    """
    Applies the Sigmoid Linear Unit (SiLU/Swish) activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor after applying SiLU.
    """
    return _silu(x)

# alias
swish = silu


@custom_grad
def _softplus(x, beta: float = 1.0, threshold: float = 20.0):
    out = x.unary_op(lambda t: torch.ops.aten.softplus(t, beta, threshold))

    def backward(grad, x=x.data, beta=beta, threshold=threshold):
        return torch.ops.aten.softplus_backward(grad, x, beta, threshold, out)

    return out, (x,), backward

def softplus(x: LiteTensor, beta: float = 1.0, threshold: float = 20.0):
    """
    Applies the Softplus activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.
        beta (float, optional): Slope parameter. Default: 1.0.
        threshold (float, optional): Numerical stability threshold. Default: 20.0.

    Returns:
        LiteTensor: Tensor after applying Softplus.
    """
    return _softplus(x, beta, threshold)


@custom_grad
def _softsign(x):
    out = x.unary_op(lambda t: torch.ops.aten.softsign(t))

    def backward(grad, out=out.data):
        return torch.ops.aten.softsign_backward(grad, out)

    return out, (x,), backward

def softsign(x: LiteTensor):
    """
    Applies the Softsign activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor after applying Softsign.
    """
    return _softsign(x)


@custom_grad
def _hardtanh(x, min_val: float = -1.0, max_val: float = 1.0):
    out = x.unary_op(lambda t: torch.ops.aten.hardtanh(t, min_val, max_val))

    def backward(grad, x=x.data, min_val=min_val, max_val=max_val):
        mask = (x >= min_val) & (x <= max_val)
        return grad * mask.to(grad.dtype)

    return out, (x,), backward

def hardtanh(x: LiteTensor, min_val: float = -1.0, max_val: float = 1.0):
    """
    Applies the HardTanh activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.
        min_val (float, optional): Minimum threshold. Default: -1.0.
        max_val (float, optional): Maximum threshold. Default: 1.0.

    Returns:
        LiteTensor: Tensor after applying HardTanh.
    """
    return _hardtanh(x, min_val, max_val)


@custom_grad
def _relu6(x):
    out = x.unary_op(lambda t: torch.ops.aten.hardtanh(t, 0.0, 6.0))

    def backward(grad, x=x.data):
        mask = (x >= 0.0) & (x <= 6.0)
        return grad * mask.to(grad.dtype)

    return out, (x,), backward

def relu6(x: LiteTensor):
    """
    Applies the ReLU6 activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor after applying ReLU6 (clamped between 0 and 6).
    """
    return _relu6(x)


@custom_grad
def _logsigmoid(x):
    out = x.unary_op(lambda t: torch.ops.aten.log_sigmoid(t))

    def backward(grad, x=x.data):
        return torch.ops.aten.log_sigmoid_backward(grad, x, out)

    return out, (x,), backward

def logsigmoid(x: LiteTensor):
    """
    Applies the Log-Sigmoid activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor after applying Log-Sigmoid.
    """
    return _logsigmoid(x)


@custom_grad
def _mish(x):
    out = x.unary_op(lambda t: torch.ops.aten.mish(t))

    def backward(grad, x=x.data):
        return torch.ops.aten.mish_backward(grad, x)

    return out, (x,), backward

def mish(x: LiteTensor):
    """
    Applies the Mish activation function element-wise.

    Args:
        x (LiteTensor): Input tensor.

    Returns:
        LiteTensor: Tensor after applying Mish.
    """
    return _mish(x)
