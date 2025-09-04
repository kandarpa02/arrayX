from .layers import Module
from typing import Callable, Any
import nexnet as nx
from .initializers import *
from .attention_blocks import multi_head_attention
import torch


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, dropout: float, initializer: Callable | Any = None, name: str = ""):
        super().__init__(name)
        self.heads = num_heads
        self.dropout = dropout
        self.init_fn = initializer if initializer is not None else xavier_uniform

    def __call__(
        self,
        q_vector: torch.Tensor,
        k_vector: torch.Tensor,
        v_vector: torch.Tensor,
        rng: Callable,
        mask: bool = False,
        deterministic: bool = False
    ) -> torch.Tensor:

        shape = [q_vector.shape[-2], q_vector.shape[-1]]
        dtype = q_vector.dtype

        with self.name_context():

            qw = self.param(
                'query',
                shape=shape,
                dtype=dtype,
                init_fn=self.init_fn,
                rng=rng
            )

            kw = self.param(
                'key',
                shape=shape,
                dtype=dtype,
                init_fn=self.init_fn,
                rng=rng
            )

            vw = self.param(
                'value',
                shape=shape,
                dtype=dtype,
                init_fn=self.init_fn,
                rng=rng
            )

            ow = self.param(
                'output',
                shape=shape,
                dtype=dtype,
                init_fn=self.init_fn,
                rng=rng
            )

        return multi_head_attention(
            q_vector=q_vector,
            k_vector=k_vector,
            v_vector=v_vector,
            qw=qw,
            kw=kw,
            vw=vw,
            ow=ow,
            num_heads=self.heads,
            use_mask=mask,
            dropout_p=self.dropout,
            training=deterministic
        )
