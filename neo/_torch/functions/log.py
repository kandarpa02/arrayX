from ..math.log_policy import *
from neo.functions import function

def log(x):
    return function(log_e)(x)

def log10(x):
    return function(log_10)(x)