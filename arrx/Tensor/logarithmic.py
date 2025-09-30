from .base import placeholder, scalar

def log(x:placeholder):
    out = placeholder.place(*x.shape, name=f"lib.log({x.name})")
    out.parents = (x, )
    def _bwd_log(grad):
        one = scalar(name='1')
        g = grad * (one / x)
        return (g,)
    
    out.grad_fn = _bwd_log
    return out

