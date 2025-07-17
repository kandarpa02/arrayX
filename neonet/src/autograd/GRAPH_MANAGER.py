from typing import NamedTuple, Callable, Any
from neonet.src.custom_typing import TensorObj

class Node:
    def __init__(self, output:TensorObj, parents:tuple, bwd_fn:Callable):
        self.output = output
        self.parents = parents
        self.bwd_fn = bwd_fn


class Tape:
    def __init__(self):
        self.nodes = []
    def add(self, node):
        self.nodes.append(node)

class TapeContext:
    current = None

    @classmethod
    def push(cls, tape:list):
        cls.current = tape
    
    @classmethod
    def pop(cls):
        cls.current=None

    @classmethod
    def add_node(cls, node:Node):
        if cls.current:
            cls.current.append(node)


