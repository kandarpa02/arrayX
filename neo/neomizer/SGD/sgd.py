from neo.neomizer.SGD import sgd_c  # compiled cython module
import torch

class SGD:
    def __init__(self, lr=1e-2, momentum=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self._backend = sgd_c.FunctionalSGD(lr, momentum, nesterov)
        self.state = {}

    def step(self, params: dict, grads: dict) -> dict:
        new_params = {}
        for name, param in params.items():
            grad = grads.get(param)
            if grad is not None:
                state = self.state.get(id(param), {})
                updated, new_state = self._backend.step(param, grad, state)
                self.state[id(param)] = new_state
                new_params[name] = updated
            else:
                new_params[name] = param 
        return new_params
