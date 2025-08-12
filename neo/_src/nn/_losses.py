import neo
import torch

def cce_fwd(logits:torch.Tensor, labels:torch.Tensor):
    shifted = logits.sub_(torch.max(logits, dim=1, keepdim=True).values)
    exps = shifted.exp()
    probs = exps.div_(exps.sum(dim=1, keepdim=True))
    log_probs = torch.log(probs.clamp(min=1e-9))
    loss = torch.sum(labels * log_probs, dim=1).mean()
    return loss, probs

# @neo.function
class _softmax_cross_entropy(neo.Policy):
    def forward(self, logits, labels):
        loss, probs = cce_fwd(logits, labels)
        self.ctx.save(probs, labels)
        return loss

    def backward(self, grad):
        probs, labels = self.ctx.release
        batch_size = labels.shape[0]
        d_logits = (probs - labels)
        return d_logits.mul_(grad.view(1, 1).div(batch_size)), torch.zeros_like(labels)

def softmax_cross_entropy(logits, labels):
    return neo.function(_softmax_cross_entropy)(logits, labels)