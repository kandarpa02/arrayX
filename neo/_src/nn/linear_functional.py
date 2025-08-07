from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib
from neo._torch.lite_tensor import LiteTensor

def linear_fwd(X: neolib.Tensor, w: neolib.Tensor, b: neolib.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b), X, w

def linear_bwd(grad: neolib.Tensor, X: neolib.Tensor, w: neolib.Tensor):
    return grad @ w, grad.T @ X, grad.sum(0)

class _linear(Policy):
    """
    Autograd policy for the `linear` operation.

    Implements the forward and backward logic for a fully-connected (dense) linear layer:
    ( y = xW^T + b )

    - Forward pass: Performs matrix multiplication followed by bias addition.
    - Backward pass: Computes gradients w.r.t. input, weight, and bias using efficient matrix operations.

    Context saved:
        - Input tensor `x` (after possible reshaping)
        - Weight tensor `w`

    Inputs:
        x (Tensor): Input tensor of shape `(N, in_features)` or `(in_features,)`
        w (Tensor): Weight tensor of shape `(out_features, in_features)`
        b (Tensor): Bias tensor of shape `(out_features,)`

    Output:
        Tensor: Result of the linear transformation of shape `(N, out_features)`

    Gradient outputs:
        - dx: Gradient of the input tensor, shape `(N, in_features)`
        - dw: Gradient of the weight tensor, shape `(out_features, in_features)`
        - db: Gradient of the bias tensor, shape `(out_features,)`
    """

    def forward(self, X, w, b):
        out, X, w = linear_fwd(X, w, b)
        self.ctx.save(X, w)
        return out

    def backward(self, grad):
        X, w = self.ctx.release
        del self.ctx
        return linear_bwd(grad, X, w)
    
    

def linear(x: LiteTensor, w: LiteTensor, b: LiteTensor):
    """
    Applies a linear transformation to the incoming data: ( y = xW^T + b )

    This operation is equivalent to a fully-connected layer without activation.
    Automatically supports autograd via Neo's function tracing mechanism.

    Args:
        x (LiteTensor): Input tensor of shape `(N, in_features)` or `(in_features,)`.
        w (LiteTensor): Weight tensor of shape `(out_features, in_features)`.
        b (LiteTensor): Bias tensor of shape `(out_features,)`.

    Returns:
        LiteTensor: Output tensor of shape `(N, out_features)`, where:
            - `N` is the batch size if `x` is 2D
            - `1` if `x` is 1D (internally reshaped)

    Notes:
        - Automatically reshapes 1D input to 2D for compatibility.
        - Supports gradient propagation through all inputs.
        - In-place bias addition is performed for performance.

    Example:
        >>> x = neo.lite([[1.0, 2.0]])
        >>> w = neo.lite([[0.5, 0.5]])
        >>> b = neo.lite([0.1])
        >>> y = linear(x, w, b)  # returns [[1.6]]
    """
    return function(_linear)(x, w, b)
