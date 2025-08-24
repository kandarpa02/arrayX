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