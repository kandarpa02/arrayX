import neo
import torch
from neo._torch.lite_tensor import LiteTensor
from neo._src.autograd.FUNCTION_REGISTER import custom_grad


@custom_grad
def _cce(x, target, reduction="mean", label_smoothing=0.0, sparse=True):
    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    reduction_enum = reduction_map[reduction]

    if sparse:
        # Sparse target: integer class indices
        loss = torch.ops.aten.cross_entropy_loss(
            x._t, target._t, None, reduction_enum, -100, label_smoothing
        )

        def backward(grad_out, x=x._t, target=target._t):
            # grad = softmax(x) - one_hot(target)
            probs = torch.ops.aten.softmax(x, dim=-1)
            target_onehot = torch.nn.functional.one_hot(target, num_classes=x.size(-1)).to(x.dtype)
            grad_input = probs - target_onehot
            if reduction == "mean":
                grad_input = grad_input * (grad_out / x.size(0))
            elif reduction == "sum":
                grad_input = grad_input * grad_out
            else:
                grad_input = grad_input * grad_out.unsqueeze(-1)
            return grad_input, None

    else:
        # Dense target: one-hot
        log_probs = torch.ops.aten.log_softmax(x._t, dim=-1)
        loss = -(target._t * log_probs).sum(dim=-1)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        def backward(grad_out, x=x._t, target=target._t):
            grad_input = torch.ops.aten.softmax(x, dim=-1) - target
            if reduction == "mean":
                grad_input = grad_input * (grad_out / x.size(0))
            elif reduction == "sum":
                grad_input = grad_input * grad_out
            else:
                grad_input = grad_input * grad_out.unsqueeze(-1)
            return grad_input, None

    return LiteTensor(loss), (x, target), backward

def cce(x: LiteTensor, target: LiteTensor, reduction="mean", label_smoothing=0.0, sparse=True):
    return _cce(x, target, reduction, label_smoothing, sparse)


@custom_grad
def _softmax_cce(x, target, reduction="mean", sparse=True):
    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    reduction_enum = reduction_map[reduction]

    if sparse:
        # Sparse targets: integer class indices
        loss = torch.ops.aten.cross_entropy_loss(x._t, target._t, None, reduction_enum, -100, 0.0)

        def backward(grad_out, x=x._t, target=target._t):
            probs = torch.ops.aten.softmax(x, dim=-1)
            target_onehot = torch.nn.functional.one_hot(target, num_classes=x.size(-1)).to(x.dtype)
            grad_input = probs - target_onehot
            if reduction == "mean":
                grad_input = grad_input * (grad_out / x.size(0))
            elif reduction == "sum":
                grad_input = grad_input * grad_out
            else:
                grad_input = grad_input * grad_out.unsqueeze(-1)
            return grad_input, None

    else:
        # Dense targets: one-hot
        probs = torch.ops.aten.softmax(x._t, dim=-1)
        loss = -(target._t * probs.clamp_min(1e-12).log()).sum(dim=-1)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        def backward(grad_out, probs=probs, target=target._t):
            grad_input = probs - target
            if reduction == "mean":
                grad_input = grad_input * (grad_out / probs.size(0))
            elif reduction == "sum":
                grad_input = grad_input * grad_out
            else:
                grad_input = grad_input * grad_out.unsqueeze(-1)
            return grad_input, None

    return LiteTensor(loss), (x, target), backward

def softmax_cce(x: LiteTensor, target: LiteTensor, reduction="mean", sparse=True):
    return _softmax_cce(x, target, reduction, sparse)
