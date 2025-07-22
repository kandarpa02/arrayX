from ..math.unary_policy import *
from neo.functions import function

def abs(x):
    return function(absolute_op)(x)

def sign(x):
    return function(signum_op)(x)

def exp(x):
    return function(exponential_op)(x) 

def sqrt(x):
    return function(sqrt_op)(x)
