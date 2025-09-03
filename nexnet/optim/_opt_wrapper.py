import torch
from typing import Any, Callable, Dict, List
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
        self.param_keys = list(params.keys())  # keep fixed order

        # Use shared storage → no clones, no double allocation
        self.torch_params = [
            torch.nn.Parameter(p.data.detach().requires_grad_())
            for p in params.values()
        ]

        # Map Neo param name → torch.Parameter
        self._param_map = dict(zip(self.param_keys, self.torch_params))

        # Torch optimizer instance
        self.optimizer = torch_opt_cls(self.torch_params, **kwargs)

    def step(self, grads: List[Any]) -> Dict[str, Any]:
        """
        Apply gradient update.

        Args:
            grads: list[LiteTensor] in same order as self.param_keys
        """
        # Assign gradients by index
        for key, grad in zip(self.param_keys, grads):
            torch_p = self._param_map[key]
            torch_p.grad = grad.data  # LiteTensor wraps torch.Tensor

        # Step + clear grads
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Shared storage → Neo params auto updated
        return self.params

    def state_dict(self):
        return {
            "torch_opt": self.optimizer.state_dict(),
            "params": {k: v.data.detach().clone() for k, v in self.params.items()}
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.optimizer.load_state_dict(state["torch_opt"])
        for k, v in state["params"].items():
            self.params[k].data.copy_(v.to(self.params[k].device))



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
