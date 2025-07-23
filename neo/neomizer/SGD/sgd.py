from neo.neomizer.SGD import sgd_c  # compiled cython module
import torch

class SGD:
    def __init__(self, lr=1e-2, momentum=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self._backend = sgd_c.FunctionalSGD(lr, momentum, nesterov)
        self.state = {}  # Stores state by parameter name

    def step(self, params: dict, grads: dict) -> dict:
        new_params = {}
        for name, param in params.items():

            grad = grads.get(param)
            if grad is not None:

                param_state = self.state.get(name, {})

                updated_param, new_state = self._backend.step(param, grad, param_state)
                
                self.state[name] = new_state
                new_params[name] = updated_param
            else:
                # If no gradient, keep parameter unchanged
                new_params[name] = param
        return new_params