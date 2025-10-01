from ..Tensor.base import placeholder
from typing import List
from dataclasses import dataclass

class TraceMeta(type):
    def __new__(cls, name, bases, namespace):
        annotations = namespace.get('__annotations__', {})
        keys = list(annotations.keys())
        if '__init__' not in namespace:
            def __init__(self, *args, **kwargs):
                # Assign positional arguments first
                for key, value in zip(keys, args):
                    setattr(self, key, value)
                # Assign remaining keyword arguments
                for key in keys[len(args):]:
                    setattr(self, key, kwargs.get(key, None))
            namespace['__init__'] = __init__
        return super().__new__(cls, name, bases, namespace)


class Trace(metaclass=TraceMeta):
    variables: list = None #type:ignore
    argumentes: list = None #type:ignore
    ctx: tuple = ()

    def _make_parents(self):
        parents = [v for v in self.__dict__.values() if isinstance(v, placeholder)]
        args = [v for v in self.__dict__.values()]
        self.variables = parents
        self.argumentes = args

    def ctx_save(self, *args):
        self.ctx = args

    def __dir__(self):
        return [k for k in self.__dict__ if not k.startswith("_")]

    def add_node(self):
        raise NotImplementedError

    def grad_fn(self, grad):
        raise NotImplementedError

    def apply(self):
        self._make_parents()
        out = self.add_node()
        out.parents = self.variables  # type: ignore
        out.grad_fn = self.grad_fn   # type: ignore
        return out
