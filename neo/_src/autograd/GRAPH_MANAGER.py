from typing import NamedTuple, Callable, Any

class Node:
    def __init__(self, output, parents:tuple|list, bwd_fn:Callable):
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
    def push(cls, tape):
        cls.current = tape
    
    @classmethod
    def pop(cls):
        cls.current=None

    @classmethod
    def add_node(cls, node:Node):
        if cls.current is not None:
            cls.current.append(node)



