from typing import Any

class Node:
    def __init__(
        self,
        op_name: str,
        inputs: list,
        output: Any,
        meta: dict = {},  # Optional: store shape/strides/dtype etc
    ):
        self.op_name = op_name
        self.inputs = inputs  # references to other Node or raw input
        self.output = output  # actual CuPy array
        self.meta = meta or {}

    def __repr__(self):
        return f"Node({self.op_name}, {self.inputs}, out={self.output.shape})"

class Tape:
    def __init__(self):
        self.nodes = []
    
    def add(self, node: Node):
        self.nodes.append(node)
    
    def dump_ir(self):
        for idx, node in enumerate(self.nodes):
            print(f"[{idx}] {node.op_name}({node.inputs}) -> shape={node.output.shape}")



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


def trace_op(op_name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            node = Node(op_name=op_name, inputs=list(args), output=out)
            if TapeContext.current:
                TapeContext.current.add(node)
            return out
        return wrapper
    return decorator


import numpy as np

# def matmul(a, b):
#     out = np.matmul(a, b)
#     node = Node(
#         op_name='matmul',
#         inputs=[a, b],
#         output=out,
#         meta={'shape': out.shape, 'dtype': out.dtype}
#     )
#     if TapeContext.current:
#         TapeContext.current.add(node)
#     return out

@trace_op("matmul")
def matmul(a, b): return np.matmul(a, b)

@trace_op("add")
def add(a, b): return np.add(a, b)

@trace_op("relu")
def relu(x): return np.maximum(0, x)



with_tape = Tape()
TapeContext.push(with_tape)

x = np.random.randn(32, 64)
w = np.random.randn(64, 128)
b = np.random.randn(128)

out = relu(add(matmul(x, w), b))

TapeContext.pop()
with_tape.dump_ir()

