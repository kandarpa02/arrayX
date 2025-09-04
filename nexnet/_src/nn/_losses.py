import nexnet
import torch
from nexnet._src.autograd.FUNCTION_REGISTER import custom_grad


@custom_grad
def _cce(x: torch.Tensor, target: torch.Tensor, reduction="mean", label_smoothing=0.0, sparse=True):
    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    reduction_enum = reduction_map[reduction]

    if sparse:
        # Sparse target: integer class indices
        loss = torch.ops.aten.cross_entropy_loss(
            x, target, None, reduction_enum, -100, label_smoothing
        )

        def backward(grad_out, x=x, target=target):
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
        log_probs = torch.ops.aten.log_softmax(x, dim=-1)
        loss = -(target * log_probs).sum(dim=-1)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        def backward(grad_out, x=x, target=target):
            grad_input = torch.ops.aten.softmax(x, dim=-1) - target
            if reduction == "mean":
                grad_input = grad_input * (grad_out / x.size(0))
            elif reduction == "sum":
                grad_input = grad_input * grad_out
            else:
                grad_input = grad_input * grad_out.unsqueeze(-1)
            return grad_input, None

    return loss, (x, target), backward


def cce(x: torch.Tensor, target: torch.Tensor, reduction="mean", label_smoothing=0.0, sparse=True):
    return _cce(x, target, reduction, label_smoothing, sparse)


@custom_grad
def _softmax_cce(x: torch.Tensor, target: torch.Tensor, reduction="mean", sparse=True):
    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    reduction_enum = reduction_map[reduction]

    if sparse:
        loss = torch.ops.aten.cross_entropy_loss(x, target, None, reduction_enum, -100, 0.0)

        def backward(grad_out, x=x, target=target):
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
        probs = torch.ops.aten.softmax(x, dim=-1)
        loss = -(target * probs.clamp_min(1e-12).log()).sum(dim=-1)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        def backward(grad_out, probs=probs, target=target):
            grad_input = probs - target
            if reduction == "mean":
                grad_input = grad_input * (grad_out / probs.size(0))
            elif reduction == "sum":
                grad_input = grad_input * grad_out
            else:
                grad_input = grad_input * grad_out.unsqueeze(-1)
            return grad_input, None

    return loss, (x, target), backward


def softmax_cce(x: torch.Tensor, target: torch.Tensor, reduction="mean", sparse=True):
    return _softmax_cce(x, target, reduction, sparse)
