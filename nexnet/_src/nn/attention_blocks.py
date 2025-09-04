from .layers import Module
from typing import Any, Optional
import torch
from torch import Tensor
from nexnet._src.autograd.GRAPH_MANAGER import Node, TapeContext
import nexnet as nx

@torch.jit.script
def attn_func_fwd(
    q_vector: Tensor, 
    k_vector: Tensor, 
    v_vector: Tensor, 
    qw: Tensor, kw: Tensor, vw: Tensor, ow: Tensor, 
    num_heads: int, use_mask: bool = True,
    dropout_p: float = 0.0, training: bool = True
):
    B, seq, dim = q_vector.shape
    head_dim = dim // num_heads
    scale = head_dim ** -0.5

    # Projections
    _q = q_vector @ qw
    _k = k_vector @ kw
    _v = v_vector @ vw

    # Split heads -> (B, H, S, Dh)
    query = _q.reshape(B, seq, num_heads, head_dim).transpose(1, 2)
    key   = _k.reshape(B, seq, num_heads, head_dim).transpose(1, 2)
    value = _v.reshape(B, seq, num_heads, head_dim).transpose(1, 2)

    # Scores: (B, H, S, S)
    scores = (query @ key.transpose(-1, -2)) * scale

    if use_mask:
        _NEG_INF_SUB = -1e9
        mask = torch.tril(torch.ones((seq, seq), device=q_vector.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, torch.as_tensor(_NEG_INF_SUB))

    weights = scores.softmax(dim=-1)  # (B, H, S, S)

    # --- Dropout here ---
    dropout_mask: Optional[Tensor] = None
    if training and dropout_p > 0.0:
        dropout_mask = (torch.rand_like(weights) > dropout_p).to(weights.dtype)
        weights = weights * dropout_mask / (1.0 - dropout_p)

    attended = weights @ value  # (B, H, S, Dh)

    # Merge heads -> (B, S, H*Dh)
    attended.transpose_(1, 2)                      
    attended_heads_merged = attended.reshape(B, seq, dim)

    out = attended_heads_merged @ ow               # (B, S, D)

    return (out, query, key, value, weights, attended_heads_merged, dropout_mask)


@torch.jit.script
def attn_func_bwd(
    grad: Tensor, 
    q_vector: Tensor, 
    k_vector: Tensor, 
    v_vector: Tensor,
    qw: Tensor, kw: Tensor, vw: Tensor, ow: Tensor,
    query: Tensor, key: Tensor, value: Tensor, weights: Tensor, 
    attended_heads_merged: Tensor,
    dropout_mask: Optional[Tensor],
    num_heads: int,
    dropout_p: float = 0.0, training: bool = True
):
    B, seq, dim = q_vector.shape
    head_dim = dim // num_heads
    scale = head_dim ** -0.5

    # d(out = A @ ow)
    grad_attended = grad @ ow.t()
    grad_ow = attended_heads_merged.reshape(-1, dim).t() @ grad.reshape(-1, dim)

    grad_attended = grad_attended.reshape(B, seq, num_heads, head_dim).transpose(1, 2)

    # attended = weights @ value
    grad_weights = grad_attended @ value.transpose(-1, -2)
    grad_value   = weights.transpose(-1, -2) @ grad_attended

    # Dropout backward 
    if training and dropout_p > 0.0 and dropout_mask is not None:
        grad_weights = grad_weights * dropout_mask / (1.0 - dropout_p)

    # Softmax backward
    dot = (grad_weights * weights).sum(dim=-1, keepdim=True)
    grad_scores = (grad_weights - dot) * weights

    # Scores = (Q @ K^T) * scale
    grad_query = (grad_scores @ key) * scale
    grad_key   = (grad_scores.transpose(-1, -2) @ query) * scale

    # Back to (B, S, D)
    grad_query = grad_query.transpose(1, 2).reshape(B, seq, dim)
    grad_key   = grad_key.transpose(1, 2).reshape(B, seq, dim)
    grad_value = grad_value.transpose(1, 2).reshape(B, seq, dim)

    # Through input projections
    grad_q_vector = grad_query @ qw.t()
    grad_k_vector = grad_key   @ kw.t()
    grad_v_vector = grad_value @ vw.t()

    grad_qw = q_vector.reshape(-1, dim).t() @ grad_query.reshape(-1, dim)
    grad_kw = k_vector.reshape(-1, dim).t() @ grad_key.reshape(-1, dim)
    grad_vw = v_vector.reshape(-1, dim).t() @ grad_value.reshape(-1, dim)

    return (
        grad_q_vector, grad_k_vector, grad_v_vector,
        grad_qw, grad_kw, grad_vw, grad_ow
    )


def multi_head_attention(
    q_vector: Tensor,
    k_vector: Tensor,
    v_vector: Tensor,
    qw: Tensor, 
    kw: Tensor, 
    vw: Tensor, 
    ow: Tensor, 
    num_heads: int,
    use_mask: bool = True,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tensor:
    out, query, key, value, weights, attended_heads_merged, dropout_mask = attn_func_fwd(
        q_vector,
        k_vector,
        v_vector,
        qw,
        kw,
        vw,
        ow,
        num_heads,
        use_mask,
        dropout_p,
        training
    )

    def backward(
        grad,
        q_vector=q_vector, k_vector=k_vector, v_vector=v_vector,
        qw=qw, kw=kw, vw=vw, ow=ow,
        query=query, key=key, value=value, weights=weights,
        attended_heads_merged=attended_heads_merged,
        dropout_mask=dropout_mask,
        num_heads=num_heads,
        dropout_p=dropout_p,
        training=training
    ):
        return attn_func_bwd(
            grad,
            q_vector, k_vector, v_vector,
            qw, kw, vw, ow,
            query, key, value, weights,
            attended_heads_merged,
            dropout_mask,
            num_heads,
            dropout_p,
            training
        )
    
    if nx.tape_enabled():
        TapeContext.add_node(
            Node(
                out,
                (q_vector, k_vector, v_vector, qw, kw, vw, ow),
                backward
            )
        )

    return out
