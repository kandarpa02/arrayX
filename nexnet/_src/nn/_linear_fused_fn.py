import torch

# raw linear
def linear_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b), X, w

def linear_bwd(grad: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    return grad @ w, grad.T @ X, grad.sum(0)

# linear relu
def linear_relu_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b).relu_(), X, w

def linear_relu_bwd(grad: torch.Tensor, out:torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    mask = out > 0
    grad = grad.mul_(mask)
    return grad @ w, grad.T @ X, grad.sum(0)

# linear tanh
def linear_tanh_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    return (X @ w.T).add_(b).tanh_(), X, w

def linear_tanh_bwd(grad: torch.Tensor, out:torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad = grad.mul_(1-out**2)
    return grad @ w, grad.T @ X, grad.sum(0)

# linear sigmoid
def linear_sigmoid_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    out = (X @ w.T).add_(b).sigmoid_()
    return out, out, X, w

def linear_sigmoid_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad = grad.mul_(out * (1 - out))
    return grad @ w, grad.T @ X, grad.sum(0)


# linear softmax
def linear_softmax_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor, dim=-1):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    logits = (X @ w.T).add_(b)
    out = torch.softmax(logits, dim=dim)
    return out, out, X, w

def linear_softmax_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor, dim=-1):
    # Jacobian-vector product trick
    dot = (grad * out).sum(dim=dim, keepdim=True)
    grad = out * (grad - dot)
    return grad @ w, grad.T @ X, grad.sum(0)


# linear leakyrelu
def linear_leakyrelu_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor, negative_slope=0.01):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    out = torch.nn.functional.leaky_relu((X @ w.T).add_(b), negative_slope)
    return out, out, X, w, negative_slope

def linear_leakyrelu_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor, negative_slope=0.01):
    mask_pos = out > 0
    grad = grad * (mask_pos + (~mask_pos) * negative_slope)
    return grad @ w, grad.T @ X, grad.sum(0)


# linear relu6
def linear_relu6_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    if X.ndim == 1:
        X = X.unsqueeze(0)
    out = torch.clamp((X @ w.T).add_(b), 0, 6)
    return out, out, X, w

def linear_relu6_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    mask = (out > 0) & (out < 6)
    grad = grad * mask
    return grad @ w, grad.T @ X, grad.sum(0)