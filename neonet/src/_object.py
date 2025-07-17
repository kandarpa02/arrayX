from typing import Any, NamedTuple, Self as Self
from dataclasses import dataclass
import numpy as np

class Tensor(NamedTuple):
    value : np.ndarray

    def __repr__(self) -> str:
        return f"{self.value}"
    
    def shape(self):
        return self.value.shape
    
    def get(self):
        return self.value
    
    @property
    def numpy(self):
        return self.value

    def _repl(self, new_value:(int | float)) -> Self:
        return self._replace(value = new_value)

@dataclass
class context:
    value : tuple = ()
    def save(self, *args):
        self.value = args
    
    @property
    def release(self):
        return self.value
