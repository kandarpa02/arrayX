from nexnet import neolib

def softmax_fwd(x:neolib.Tensor, dim:int):
    return neolib.nn.functional.softmax(x, dim=dim)

def softmax_bwd(out:neolib.Tensor, grad:neolib.Tensor, dim:int):
    dot = (grad * out).sum(dim=dim, keepdim = True)
    return out * (grad - dot)

def tanh_fwd(x: neolib.Tensor):
    return neolib.tanh(x)

def tanh_bwd(out: neolib.Tensor, grad: neolib.Tensor):
    return grad * (1 - out * out)
