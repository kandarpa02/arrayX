# cython: language_level=3
from libcpp.vector cimport vector
from libcpp cimport bool

cdef class Node:
    cdef public object output
    cdef public tuple parents
    cdef public object bwd_fn

cdef class Tape:
    cdef list nodes
    cpdef add(self, Node node)
    cpdef clear(self)
