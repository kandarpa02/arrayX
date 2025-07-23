from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

cdef class Node:
    cdef public object output
    cdef public tuple parents
    cdef public object bwd_fn

    def __cinit__(self, object output, tuple parents, object bwd_fn):
        self.output = output
        self.parents = parents
        self.bwd_fn = bwd_fn


cdef class Tape:
    cdef list nodes 

    def __cinit__(self):
        self.nodes = []

    def add(self, Node node):
        self.nodes.append(node)

    def __getitem__(self, int i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    def clear(self):
        self.nodes.clear()


class TapeContext:
    current = None 

    @staticmethod
    def push(Tape tape):
        TapeContext.current = tape

    @staticmethod
    def pop():
        TapeContext.current = None

    @staticmethod
    def add_node(Node node):
        if TapeContext.current is not None:
            TapeContext.current.add(node)
