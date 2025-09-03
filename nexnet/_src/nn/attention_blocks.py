from nexnet.functions import function
from nexnet._src.autograd.FUNCTION_REGISTER import Policy
from nexnet._src.nn._linear_fused_fn import *
import torch
from typing import Any

class _fused_attn_dropout_layernorm(Policy):
    def forward(self, *args):
        return NotImplementedError
    