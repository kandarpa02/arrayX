from ._object import Tensor, context
import numpy as np
from neonet.backend import get_xp

def safe_input(x):
    if not isinstance(x, tensor):
        try:
            x = tensor(x)
        except:
            raise ValueError("f{x} is not a valid dtype to be converted to Tensor" \
            ", maybe you passed a list or tuple or other unsupported data types, " \
            "use NumPy arrays instead for CPU usage")
    return x
        
        
class tensor:
    def __init__(self, data, _ctx=(), device:str='cpu'):
        self.xp = get_xp(device)
        self.data = Tensor(self.xp.asarray(data)) if not isinstance(data, tensor) else data
        self.ctx = context(_ctx)
        self.device  = device
        self.id = id(self)
        self.xp = get_xp(device)

    def numpy(self):
        return self.data.numpy
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, tensor) and self.id == other.id

    @property
    def shape(self):
        if not isinstance(self.data.get(), np.ndarray):
            return ()
        else:
            return self.data.shape()
        
    def __add__(self, other):
        self = safe_input(self)
        other = safe_input(other)
        
        a, b = self.data.get(), other.data.get()
        out = tensor((a + b), (self, other))
        return out
    
    def __mul__(self, other):
        self = safe_input(self)
        other = safe_input(other)

        a, b = self.data.get(), other.data.get()
        out = tensor((a * b), (self, other))
        return out
    
    def __radd__(self, other):
        self = safe_input(self)
        other = safe_input(other)
        return self.__add__(other)
    
    def __rmul__(self, other):
        self = safe_input(self)
        other = safe_input(other)
        return self.__mul__(other)

    def __repr__(self):
        """
        String representation of the deriv.array object.
        """
        
        prefix = " " * len("Tensor(")
        arr_str = self.xp.array2string(
            self.data.value,
            precision=4,
            suppress_small=True,
            threshold=6,        
            edgeitems=3,       
            max_line_width=80, 
            separator=', ',     
            prefix=prefix     
        )
        return f"Tensor({arr_str})"
