from .base import placeholder, vector, matrix

def sum(a: vector | matrix, axis=None, keepdims=False):
    a.sum(axis=axis, keepdims=keepdims)

def mean(a: vector | matrix, axis=None, keepdims=False):
    a.mean(axis=axis, keepdims=keepdims)

def prod(a: vector | matrix, axis=None, keepdims=False):
    a.prod(axis=axis, keepdims=keepdims)

def max(a: vector | matrix, axis=None, keepdims=False):
    raise a.max(axis=axis, keepdims=keepdims)

def min(a: vector | matrix, axis=None, keepdims=False):
    raise a.min(axis=axis, keepdims=keepdims)

