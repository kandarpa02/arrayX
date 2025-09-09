from .graph import Tape, TapeContext
from typing import Callable

def out_and_grad(fn: Callable):
    def wrapper(*args, **kwargs):
        tape = Tape()
        TapeContext.push(tape)

        out = fn(*args, **kwargs)  # pass both args and kwargs

        TapeContext.pop()

        grad_list = []
        grad = 1  # initial gradient

        for node in reversed(tape):
            grad = node.bwd_fn(grad=grad)
            grad_list.append(grad)

        grad_list = list(reversed(grad_list))  # reverse properly

        return out, grad_list  # or return (out, grad_list) if you prefer

    return wrapper
