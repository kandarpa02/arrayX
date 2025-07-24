from neo import neolib

@neolib.jit.script
def fast_softmax_fwd(x:neolib.Tensor, dim:int):
    return neolib.nn.functional.softmax(x, dim=dim)

@neolib.jit.script
def fast_softmax_bwd(out:neolib.Tensor, grad:neolib.Tensor, dim:int):
    dot = (grad * out).sum(dim=dim, keepdim = True)
    return out * (grad - dot)


@neolib.jit.script
def fast_tanh_fwd(x: neolib.Tensor):
    return neolib.tanh(x)

@neolib.jit.script
def fast_tanh_bwd(out: neolib.Tensor, grad: neolib.Tensor):
    return grad * (1 - out * out)
