from ._object import Tensor

class tensor:
    def __init__(self, data):
        self.data = Tensor(data)

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    def __add__(self, other):
        out = tensor((self.data.get() + other.data.get()))
        return out
    
