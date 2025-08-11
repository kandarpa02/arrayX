
def function(fn):
    from neo._src.autograd.define_then_run import Symbol, Variable, Constant
    def wrapper(*args, **kwargs):
        # Convert all args to Symbol if not already
        sym_args = [a if isinstance(a, Symbol) else Symbol(a) for a in args]

        # Convert kwargs values to Symbol only if they're Variable/Constant
        sym_kwargs = {
            k: (v if isinstance(v, Symbol) else Symbol(v))
            if isinstance(v, (Variable, Constant, Symbol)) else v
            for k, v in kwargs.items()
        }

        return fn(*sym_args, **sym_kwargs)
    return wrapper
