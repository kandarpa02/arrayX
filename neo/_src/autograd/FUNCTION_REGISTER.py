from typing import Any
from dataclasses import dataclass

def define_device(x):
    import numpy as np
    device = 'cpu'
    if not isinstance(x, np.ndarray):
        device = 'cuda'
    return device

@dataclass
class context:
    def save(self, *args):
        self.data = args
    @property
    def release(self):
        return self.data

class Policy:
    def __init__(self, device='cpu'):
        self.ctx = context()
        self.device = device

    @property
    def lib(self):
        from neo.backend import get_xp
        return get_xp(self.device)
    
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
