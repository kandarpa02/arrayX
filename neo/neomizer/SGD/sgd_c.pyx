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
        cdef object velocity = state.get("velocity", torch.zeros_like(param))

        velocity = velocity.mul_(self.momentum).add_(grad)

        if self.nesterov:
            update = grad.add(velocity, alpha=self.momentum)
        else:
            update = velocity
            
        return param - self.lr * update, {"velocity": velocity}