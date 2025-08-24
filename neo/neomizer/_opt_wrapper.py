import torch
from typing import Any, Callable, Dict
from torch.optim import (
    SGD as _SGD,
    Adam as _Adam,
    AdamW as _AdamW,
    Adadelta as _Adadelta,
    Adagrad as _Adagrad,
    Adamax as _Adamax,
    ASGD as _ASGD,
    LBFGS as _LBFGS,
    NAdam as _NAdam,
    RAdam as _RAdam,
    RMSprop as _RMSprop,
    Rprop as _Rprop
)

class NeoOptimizer:
    def __init__(self, params: Dict[str, Any], torch_opt_cls: Callable, **kwargs):
        """
        Neo optimizer wrapper around PyTorch optimizers.
        
        Args:
            params: dict[str, LiteTensor]  # Neo params
            torch_opt_cls: torch.optim class (SGD, Adam, etc.)
            **kwargs: optimizer hyperparams
        """
        self.params = params

        # Use shared storage → no clones, no double allocation
        self.torch_params = [
            torch.nn.Parameter(p._tensor.detach().requires_grad_())
            for p in params.values()
        ]
        
        # Map Neo param names → torch Parameter
        self._param_map = dict(zip(params.keys(), self.torch_params))

        # Torch optimizer instance
        self.optimizer = torch_opt_cls(self.torch_params, **kwargs)

    def step(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply gradient update.
        
        Args:
            grads: dict[str, LiteTensor] with same keys as params
        """
        # Assign gradients directly
        for key, grad in grads.items():
            if key not in self._param_map:
                continue
            torch_p = self._param_map[key]
            torch_p.grad = grad._tensor  # LiteTensor wraps a torch.Tensor

        # Step + clear grads
        self.optimizer.step()
        self.optimizer.zero_grad()

        # No need to copy back → shared storage keeps Neo params updated
        return self.params

    def state_dict(self):
        """Return optimizer + params state for checkpointing"""
        return {
            "torch_opt": self.optimizer.state_dict(),
            "params": {k: v._tensor.clone() for k, v in self.params.items()}
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load optimizer + params state"""
        self.optimizer.load_state_dict(state["torch_opt"])
        for k, v in state["params"].items():
            self.params[k]._tensor.copy_(v)


class SGD(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _SGD, **kwargs)

class Adam(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _Adam, **kwargs)

class AdamW(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _AdamW, **kwargs)

class Adadelta(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _Adadelta, **kwargs)

class Adagrad(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _Adagrad, **kwargs)

class Adamax(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _Adamax, **kwargs)

class ASGD(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _ASGD, **kwargs)

class LBFGS(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _LBFGS, **kwargs)

class NAdam(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _NAdam, **kwargs)

class RAdam(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _RAdam, **kwargs)

class RMSprop(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _RMSprop, **kwargs)

class Rprop(NeoOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, _Rprop, **kwargs)
