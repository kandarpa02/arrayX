from typing import NamedTuple, Any

class parameter(NamedTuple):
    param: Any

    def get_params(self):
        return self.param

class Layer:
    def __init__(self, name:str='') -> None:
        self.name=name

    def parameters(self):
        return NotImplementedError
    
    def call(self):
        return NotImplementedError

    def __call__(self, *args):
        return self.call(*args)
    
    