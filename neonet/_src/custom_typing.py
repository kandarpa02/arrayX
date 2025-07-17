from typing import Protocol
from neonet._src.struc import tensor
import numpy as np

class TensorObj(Protocol):
    def __init__(self, data, _ctx) -> None: ...

    def numpy(self) -> np.ndarray: ...
    
    def __add__(self, other) -> tensor: ...


        

