# ================================================================
# NeoNet Core Autograd primitives
# WARNING: DO NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING
# The stuff below is HIGHLY PERFORMANCE SENSITIVE and literally
# just wires tensors to nodes with minimal Python overhead.
# You break it → you die (maybe not literally but GPU tears guaranteed)
# ================================================================

from typing import Any, Callable
from dataclasses import dataclass

# CONTEXT: stupid simple ctx thingy, stores whatever you want.
# inspired by PyTorch ctx, but this one is dumb as a brick.
# Only saves what you give it, no magic, no sanity checks, deal with it.

@dataclass
class context:
    def save(self, *args):
        # lol just store it, will explode if you store weird stuff
        self.data = args

    @property
    def release(self):
        # returns the stuff you stored, no copies, no guarantees
        return self.data

# POLICY: base class for higher level user-facing ops
# forward / backward must be implemented by subclasses
# Kinda like an abstract class but we trust you not to be stupid

class Policy:
    def __init__(self):
        # ctx is your little black box, shove tensors in, get them back later
        self.ctx = context()
    
    def forward(self, *args):
        raise NotImplementedError("implement this or cry later")

    def backward(self, grad):
        raise NotImplementedError("implement this or cry later")

# ----------------------------------------------------------------
# TRACELET: low-level, minimal overhead, pure functional, ugly AF
# Works like: enter context -> register node -> exit -> nothing remains
# Think JAX-style, think speed, think pain for humans reading this
# WARNING: do not store stuff in self.out if you like memory
# ----------------------------------------------------------------
class Tracelet:
    def __init__(self) -> None:
        # placeholders, will be overwritten by register(), do not touch
        self.out = None
        self.parents = None
        self.backward = None

    def __enter__(self):
        # yay enter, nothing special here
        return self

    def register(self, out, parents, backward):
        from nexnet._src.autograd import Node, TapeContext
        from nexnet import record_tape as rt
        # literally shove node into the tape
        self.out = out
        self.parents = parents
        self.backward = backward

        # Node creation is messy, deals with raw tensors, closures, and tears
        if rt.is_enabled():
            TapeContext.add_node(
                Node(
                    output=self.out,     # yeah output, could be anything
                    parents=self.parents, # tuple of inputs, should be LiteTensors ideally
                    bwd_fn=self.backward # lambda / function to call for backward
                )
            )
        # after this, Python overhead = basically zero, GPU does the work

    def __exit__(self, val1, val2, val3):
        # bye bye references, help GC, prevent leaks, optional cleanup, do not touch
        self.out = None
        self.parents = None
        self.backward = None


from torch import Tensor
def custom_grad(fn:Callable):
    def wrapper(*args, **kwargs) -> Tensor:
        from nexnet._src.autograd import Node, TapeContext
        from nexnet import record_tape as rt
        out, parents, grad_fn = fn(*args, **kwargs)
        if rt.is_enabled():
            TapeContext.add_node(
                Node(
                    output = out,
                    parents = parents, 
                    bwd_fn = grad_fn
                )
            )
        return out
    return wrapper
