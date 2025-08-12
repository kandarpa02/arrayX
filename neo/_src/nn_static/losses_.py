from neo._src.autograd.define_then_run import Symbol
from neo._src.nn import _losses as loss

# Map nice names -> Policy classes
_LOSS_POLICIES = {
    "softmax_cross_entropy": loss._softmax_cross_entropy,
}

# Auto-create wrapper functions
def _make_loss_fn(name, policy_cls):
    def _fn(x: Symbol, y:Symbol):
        return x._binary_op(y, policy_cls)
    _fn.__name__ = name
    _fn.__doc__ = f"{name}(x) (y) — applies {policy_cls.__name__} loss."
    return _fn

# Exported functions
globals().update({
    name: _make_loss_fn(name, cls)
    for name, cls in _LOSS_POLICIES.items()
})
