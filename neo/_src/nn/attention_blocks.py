from neo.functions import function
from neo._src.autograd.FUNCTION_REGISTER import Policy
from neo import neolib


@neolib.jit.script
def multihead_attention_forward_fn(query, key, value, mask, num_heads: int, dropout_p: float, train:bool):
    B, T, D = query.shape
    S = key.shape[1]
    d = D // num_heads
    scale = d ** -0.5

    q = query.view(B, T, num_heads, d).permute(0, 2, 1, 3)
    k = key.view(B, S, num_heads, d).permute(0, 2, 1, 3)
    v = value.view(B, S, num_heads, d).permute(0, 2, 1, 3)

    scores = neolib.matmul(q, k.transpose(-2, -1))
    scores.mul_(scale)

    if mask is not None:
        scores.masked_fill_(mask.unsqueeze(1) == 0, float('-inf'))

    attn = neolib.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = neolib.dropout(attn, p=dropout_p, train=train)

    out = neolib.matmul(attn, v)
    out = out.permute(0, 2, 1, 3).reshape(B, T, D)

    return out, q, k, v, attn


@neolib.jit.script
def multihead_attention_backward_fn(grad_out, q, k, v, attn, mask, num_heads: int):
    B, H, T, d = q.shape
    D = H * d

    grad_out = grad_out.view(B, T, H, d).permute(0, 2, 1, 3)

    grad_v = neolib.matmul(attn.transpose(-2, -1), grad_out)
    grad_attn = neolib.matmul(grad_out, v.transpose(-2, -1))

    tmp = grad_attn * attn
    grad_scores = tmp - attn * neolib.sum(tmp, dim=-1, keepdim=True)
    scale = d ** -0.5
    grad_scores.mul_(scale)

    if mask is not None:
        grad_scores.masked_fill_(mask.unsqueeze(1) == 0, 0.0)

    grad_q = neolib.matmul(grad_scores, k)
    grad_k = neolib.matmul(grad_scores.transpose(-2, -1), q)

    grad_query = grad_q.permute(0, 2, 1, 3).reshape(B, T, D)
    grad_key   = grad_k.permute(0, 2, 1, 3).reshape(B, k.shape[2], D)
    grad_value = grad_v.permute(0, 2, 1, 3).reshape(B, v.shape[2], D)

    return grad_query, grad_key, grad_value

@function
class MultiHeadAttention(Policy):

    def forward(self, query, key, value, num_heads, dropout_p, mask=None):
        out, q, k, v, attn = multihead_attention_forward_fn(
            query, key, value, mask, num_heads, dropout_p
        )
        self.ctx.save(q, k, v, attn, mask, num_heads)
        return out

    def backward(self, grad_out):
        q, k, v, attn, mask, num_heads = self.ctx.release
        return multihead_attention_backward_fn(
            grad_out, q, k, v, attn, mask, num_heads
        )
