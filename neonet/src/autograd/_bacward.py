from neonet.src.autograd import Node, Tape, TapeContext
from typing import Callable
from neonet.src.TEMPORARY import xp

def value_and_grad(fn: Callable):
    def wrapped_function(*args):
        tape = Tape()
        TapeContext.push(tape.nodes)

        out = fn(*args)

        TapeContext.pop()
        print("tape: ", tape.nodes)
        
        grads = {id(arg): 0 for arg in args}
        grads_out = xp.ones_like(out.numpy())

        for node in reversed(tape.nodes):
            grad_inputs = node.bwd(grad=grads_out)
            if grad_inputs is None:
                continue
            for parent, grad in zip(node.parents, grad_inputs):
                grads[id(parent)] += grad
            grads_out = grad 

        return out, {arg: grads[id(arg)] for arg in args}

    return wrapped_function




            