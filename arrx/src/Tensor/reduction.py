from .base import placeholder, vector, matrix

def sum(a: vector | matrix, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)

def mean(a: vector | matrix, axis=None, keepdims=False):
    return a.mean(axis=axis, keepdims=keepdims)

def prod(a: vector | matrix, axis=None, keepdims=False):
    return a.prod(axis=axis, keepdims=keepdims)

def max(a: vector | matrix, axis=None, keepdims=False):
    return a.max(axis=axis, keepdims=keepdims)

def min(a: vector | matrix, axis=None, keepdims=False):
    return a.min(axis=axis, keepdims=keepdims)

