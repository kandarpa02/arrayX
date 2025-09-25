from ..Tensor.base import scalar, vector, matrix, placeholder
from ..Tensor import arithmetic, logarithmic
from .graph import Function
from typing import Union
from dataclasses import dataclass
import inspect


def build_func_dict(*modules):
    func_dict = {}
    for mod in modules:
        funcs = {name: func for name, func in inspect.getmembers(mod, inspect.isfunction)}
        func_dict.update(funcs)
    return func_dict

_OP_MAP = build_func_dict(arithmetic, logarithmic)
TensorLike = Union[scalar, vector, matrix]


class EmptyNodeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FlashGraph:
    _nodes: dict = None #type:ignore
    _history: dict = None #type:ignore
    _end_node: TensorLike = None #type:ignore

    @property
    def nodes(self):
        return self._nodes
    
    def out_node(self):
        return self._end_node

    def Init(self, shape:list=[], node:str=None): #type:ignore
        place = placeholder.place(*shape, name=node)
        self._end_node = place
        self._nodes = {'init':place}
        self._history = {place.name:place}

    def Init_Intermediate(self, graph):
        if not isinstance(graph, FlashGraph):
            raise RuntimeError(f"graph must be a {FlashGraph} instance, but found {type(graph)}")
        self._end_node = graph._end_node
        self._nodes = graph._nodes
        self._history = graph._history

    def Add(self, operation:str='', shape:list=[], node:str=None): #type:ignore
        try:
            _OP_MAP.get(operation, None)
        
        except KeyError:
            raise KeyError(f'{operation} is not a valid operation')

        finally:
            op = _OP_MAP.get(operation, None)
            place = placeholder.place(*shape, name=node)
            if not operation == '':
                if not op is None:
                    if not self._end_node is None:
                        self._end_node = op(self._end_node, place) #type:ignore
                        self._nodes[place.name] = self._end_node
                        self._history[place.name] = place
                    else:
                        raise EmptyNodeError(f'Graph is not initialized')
                else:
                    raise KeyError(f'Invalid operation [{operation}] is given')
                
            else:
                raise ValueError(f'Intermediate nodes must have an operation')
            
    def Residual(self, operation:str='', shape:list=[], res_node:str=None, new_node:str=None): #type:ignore
        if res_node is None:
            raise EmptyNodeError(f"res_node can't be None in Residual method")
        
        try:
            self._nodes.get(res_node, None)
        
        except KeyError:
            raise KeyError(f'[{res_node}] is not a resious node')
        
        else:
            res_node = self._history.get(res_node, None) #type:ignore
        
        try:
            _OP_MAP.get(operation, None)
        
        except KeyError:
            raise KeyError(f'{operation} is not a valid operation')

        finally:
            op = _OP_MAP.get(operation, None)
            place = placeholder.place(*shape, name=new_node)
            if not operation == '':
                if not op is None:
                    self._end_node = op(self._end_node, res_node) #type:ignore
                    self._nodes[place.name] = self._end_node
                    
                else:
                    raise KeyError(f'Invalid operation [{operation}] is given')
                
            else:
                raise ValueError(f'Intermediate nodes must have an operation')

        
    def Compile(self):
        graph = Function(self._end_node, list(self._history.values()))
        self.fw, self.bw = graph.forward(), graph.backward()

    
    def apply(self, *args):
        return self.fw(*args)
    
    def grad(self, *args):
        return self.bw(*args)