from typing import Any
from dataclasses import dataclass

@dataclass
class context:
    def save(self, *args):
        self.data = args
    @property
    def release(self):
        return self.data

class Policy:
    def __init__(self):
        self.ctx = context()
    
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

