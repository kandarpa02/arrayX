from typing import Any
from ..Tensor.base import placeholder, vector, matrix
from ..Tensor.utils import *

class internal:
    def __init__(self, name='', signature=()) -> None:
        self.name = name
        self.expr = ''
        self.sig = f'({", ".join(signature)})'

    def __repr__(self) -> str:
        return self.expr
    
    def call(self, *args):
        raise NotImplementedError
    
    def __call__(self) -> str:
        self.expr = f'{self.name}{self.sig}'
        return self.expr

    class _apply:
        def __get__(self, instance, owner):
            if instance is None:
                # class-level access; auto-instantiate and call
                def wrapper(*args, **kwargs):
                    obj = owner()
                    return obj.call(*args, **kwargs)
                return wrapper
            else:
                # instance-level access; just delegate
                def wrapper(*args, **kwargs):
                    return instance.call(*args, **kwargs)
                return wrapper
    apply = _apply()
