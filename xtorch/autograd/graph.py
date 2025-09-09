from typing import Callable, Any, Tuple

class Node:
    def __init__(self, output: Any, parents: Tuple[Any], bwd_fn: Callable):
        self.output = output
        self.parents = parents
        self.bwd_fn = bwd_fn


class Tape:
    def __init__(self):
        self.stored = []

    def add(self, node: Node):
        self.stored.append(node)

    def __getitem__(self, index):
        return self.stored[index]

    def __len__(self):
        return len(self.stored)

    def clear(self):
        self.stored.clear()


class TapeContext:
    context = None

    @staticmethod
    def add(node: Node):
        if TapeContext.context is not None:
            TapeContext.context.add(node)

    @staticmethod
    def push(tape: Tape):
        TapeContext.context = tape

    @staticmethod
    def pop():
        TapeContext.context = None
