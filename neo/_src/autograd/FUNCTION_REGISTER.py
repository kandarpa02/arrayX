from typing import Any
from dataclasses import dataclass
from neo.backend import get_xp

def define_device(x):
    import numpy as np
    device = 'cpu'
    if not isinstance(x, np.ndarray):
        device = 'cuda'
    return device

# @dataclass
class context:
    def __init__(self):
        self.fingerprint = self.data[0]

    def save(self, *args):
        self.data = args

    @property
    def release(self):
        return self.data

class Policy:
    def __init__(self):
        self.ctx = context()
        self.lib = get_xp(define_device(self.ctx.fingerprint))
    
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

