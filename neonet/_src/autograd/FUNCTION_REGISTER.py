from typing import Any
from .._object import context
from dataclasses import dataclass


class Policy:
    def __init__(self):
        self.ctx = context()
    
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    
    
