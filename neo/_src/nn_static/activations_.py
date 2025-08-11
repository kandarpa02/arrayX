from neo._src.autograd.define_then_run import Symbol
from neo._src.nn import _activations as act

# Map nice names -> Policy classes
_ACTIVATION_POLICIES = {
    "relu": act._relu,
    "tanh": act._tanh,
}

# Auto-create wrapper functions
def _make_activation(name, policy_cls):
    def _fn(x: Symbol):
        return x._unary_op(policy_cls)
    _fn.__name__ = name
    _fn.__doc__ = f"{name}(x) — applies {policy_cls.__name__} activation."
    return _fn

# Exported functions
globals().update({
    name: _make_activation(name, cls)
    for name, cls in _ACTIVATION_POLICIES.items()
})
