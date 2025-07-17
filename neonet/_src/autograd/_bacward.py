from neonet._src.autograd import Node, Tape, TapeContext
from typing import Callable
from neonet._src.TEMPORARY import xp

def value_and_grad(fn: Callable):
    def wrapped_function(*args):
        tape = Tape()
        TapeContext.push(tape.nodes)
        out = fn(*args)
        TapeContext.pop()
        
        grad_dict = {}
        grad_dict[id(out)] = xp.ones_like(out.numpy()) 
        
        for node in reversed(tape.nodes):
            node_out_grad = grad_dict.get(id(node.output))
            if node_out_grad is None:
                continue
                
            grad_inputs = node.bwd_fn(grad=node_out_grad)
            if grad_inputs is None:
                continue
                
            for parent, grad in zip(node.parents, grad_inputs):
                pid = id(parent)
                if pid in grad_dict:
                    grad_dict[pid] += grad
                else:
                    grad_dict[pid] = grad
        
        arg_grads = {arg: grad_dict.get(id(arg), 0) for arg in args}
        return out, arg_grads

    return wrapped_function



