VARIABLE_REGISTRY = {}
SCOPE_STACK = []

from ..utils import ParamDict

class variable_scope:
    def __init__(self, name, reuse=False, reset=False):
        self.name = name
        self.reuse = reuse
        self.reset = reset

    def __enter__(self):
        # Determine effective reuse: inherit reuse from outer scopes
        effective_reuse = self.reuse or any(r for _, r in SCOPE_STACK)
        SCOPE_STACK.append((self.name, effective_reuse))

        if self.reset:
            prefix = "_".join(scope for scope, _ in SCOPE_STACK)
            keys_to_remove = [k for k in VARIABLE_REGISTRY if k.startswith(prefix)]
            for k in keys_to_remove:
                del VARIABLE_REGISTRY[k]

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        SCOPE_STACK.pop()


def variables_in_scope(scope_name):
    return ParamDict({k: v for k, v in VARIABLE_REGISTRY.items() if k.startswith(scope_name + "_")})


def get_variable(name: str, shape=None, initializer=lambda shape, seed=None: None, rng=None):
    from ...user_api.basic import Variable

    shape = shape or []

    # Build full variable name
    full_scope = "_".join(scope for scope, _ in SCOPE_STACK) if SCOPE_STACK else ""
    full_name = f"{full_scope}_{name}" if full_scope else name

    # Determine if reuse is allowed from current scope stack
    reuse_allowed = any(r for _, r in SCOPE_STACK)

    # Variable already exists
    if full_name in VARIABLE_REGISTRY:
        if not reuse_allowed:
            raise ValueError(f"Variable {full_name} already exists, but reuse is False")
        return VARIABLE_REGISTRY[full_name]

    # Variable does not exist but reuse is requested
    if reuse_allowed:
        raise ValueError(f"Variable {full_name} does not exist, cannot reuse")

    # Otherwise, create a new variable
    try:
        data = initializer(shape, key=rng)
    except TypeError:
        data = initializer(shape)

    out = Variable(data, name=full_name)
    VARIABLE_REGISTRY[full_name] = out
    return out
