from ..math.reductions_policy import *
from neo.functions import function

def max(x, dim=0, keepdim=False):
    return function(max_op)(x, dim, keepdim)
def mean(x, dim=None, keepdim=False):
    return function(mean_op)(x, dim, keepdim)
def sum(x, dim=None, keepdim=False):
    return function(sum_op)(x, dim, keepdim)