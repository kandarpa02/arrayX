# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.

from typing import Any
from dataclasses import dataclass


@dataclass
class context:
    def save(self, *args):
        self.data = args
    @property
    def release(self):
        return self.data

from neo._torch import neolib

class Policy:
    def __init__(self):
        self.ctx = context()
    
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
