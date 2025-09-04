import torch
import torch.nn.functional as F


def flatten_batch(X: torch.Tensor):
    """Flatten all dims except last feature dim"""
    orig_shape = X.shape[:-1]
    X_flat = X.reshape(-1, X.shape[-1])
    return X_flat, orig_shape

def unflatten_batch(X_flat: torch.Tensor, orig_shape, out_features):
    """Reshape back to original batch dims + out_features"""
    return X_flat.reshape(*orig_shape, out_features)

# raw linear
def linear_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    X_flat, orig_shape = flatten_batch(X)
    out_flat = X_flat @ w.T + b
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, X, w

def linear_bwd(grad: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad_flat, _ = flatten_batch(grad)
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db


# linear relu
def linear_relu_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    X_flat, orig_shape = flatten_batch(X)
    out_flat = (X_flat @ w.T + b).relu_()
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, X, w

def linear_relu_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad_flat, _ = flatten_batch(grad)
    out_flat, _ = flatten_batch(out)
    mask = out_flat > 0
    grad_flat = grad_flat * mask
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db

def linear_tanh_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    X_flat, orig_shape = flatten_batch(X)
    out_flat = (X_flat @ w.T + b).tanh_()
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, X, w

def linear_tanh_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad_flat, _ = flatten_batch(grad)
    out_flat, _ = flatten_batch(out)
    grad_flat = grad_flat * (1 - out_flat ** 2)
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db

def linear_sigmoid_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    X_flat, orig_shape = flatten_batch(X)
    out_flat = (X_flat @ w.T + b).sigmoid_()
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, out, X, w

def linear_sigmoid_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad_flat, _ = flatten_batch(grad)
    out_flat, _ = flatten_batch(out)
    grad_flat = grad_flat * (out_flat * (1 - out_flat))
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db


def linear_softmax_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor, dim=-1):
    X_flat, orig_shape = flatten_batch(X)
    logits_flat = X_flat @ w.T + b
    out_flat = torch.softmax(logits_flat, dim=dim)
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, out, X, w

def linear_softmax_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor, dim=-1):
    grad_flat, _ = flatten_batch(grad)
    out_flat, _ = flatten_batch(out)
    dot = (grad_flat * out_flat).sum(dim=dim, keepdim=True)
    grad_flat = out_flat * (grad_flat - dot)
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db


def linear_leakyrelu_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor, negative_slope=0.01):
    X_flat, orig_shape = flatten_batch(X)
    out_flat = F.leaky_relu(X_flat @ w.T + b, negative_slope)
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, out, X, w, negative_slope

def linear_leakyrelu_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor, negative_slope=0.01):
    grad_flat, _ = flatten_batch(grad)
    out_flat, _ = flatten_batch(out)
    mask_pos = out_flat > 0
    grad_flat = grad_flat * (mask_pos + (~mask_pos) * negative_slope)
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db

def linear_relu6_fwd(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    X_flat, orig_shape = flatten_batch(X)
    out_flat = torch.clamp(X_flat @ w.T + b, 0, 6)
    out = unflatten_batch(out_flat, orig_shape, w.shape[0])
    return out, out, X, w

def linear_relu6_bwd(grad: torch.Tensor, out: torch.Tensor, X: torch.Tensor, w: torch.Tensor):
    grad_flat, _ = flatten_batch(grad)
    out_flat, _ = flatten_batch(out)
    mask = (out_flat > 0) & (out_flat < 6)
    grad_flat = grad_flat * mask
    X_flat, _ = flatten_batch(X)
    dw = grad_flat.T @ X_flat
    db = grad_flat.sum(0)
    dx = grad_flat @ w
    dx = unflatten_batch(dx, X.shape[:-1], X.shape[-1])
    return dx, dw, db
