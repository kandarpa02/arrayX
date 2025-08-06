from ..math.reductions_policy import *
from neo.functions import function

def max(x):
    return function(max_op)(x)
def mean(x):
    return function(mean_op)(x)
def sum(x):
    return function(sum_op)(x)