from ..math.arithmetic_policy import *
from neo.functions import function

def add(x, y):
    return function(addition)(x, y)

def mul(x, y):
    return function(multiplication)(x, y)

def sub(x, y):
    return function(subtraction)(x, y)

def div(x, y):
    return function(division)(x, y)

def power(x, y):
    return function(power_op)(x, y)

def neg(x):
    return function(negative)(x)

def matmul(x, y):
    return function(matmul_op)(x, y)