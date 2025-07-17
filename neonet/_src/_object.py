from typing import Any, NamedTuple, Self as Self
from dataclasses import dataclass
from neonet.backend import get_xp
import numpy as np

class Tensor(NamedTuple):
    value : Any

    def __repr__(self) -> str:
        return f"{self.value}"
    
    def shape(self):
        return self.value.shape
    
    def get(self):
        return self.value
    
    @property
    def numpy(self):
        return np.asarray(self.value)

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
