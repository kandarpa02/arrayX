from typing import Protocol
from neonet.src.struc import tensor
import numpy as np

class TensorObj(Protocol):
    # @property
    # def shape(self) -> tuple: ...

    # @property
    # def dtype(self) -> type: ...
    def __init__(self, data, _ctx=()) -> None: ...

    def numpy(self) -> np.ndarray: ...
    
    def __add__(self, other) -> tensor: ...


        

