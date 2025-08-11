from neo._src.nn._activations import _relu, _tanh
from neo._src.autograd.define_then_run import Symbol

def relu(x:Symbol):
    return x._unary_op(_relu)

def tanh(x:Symbol):
    return x._unary_op(_tanh)