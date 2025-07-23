# distutils: language = c++
import torch

cdef class FunctionalSGD:
    cdef float lr
    cdef float momentum
    cdef bint nesterov

    def __cinit__(self, float lr, float momentum=0.0, bint nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self, object param, object grad, dict state):
        velocity = state.get("velocity", torch.zeros_like(param))
        velocity.mul_(self.momentum).add_(grad)
        if self.nesterov:
            update = grad + self.momentum * velocity
        else:
            update = velocity
        return param - self.lr * update, {"velocity": velocity}
