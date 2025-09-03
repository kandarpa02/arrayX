# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.
cimport cython
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool
from nexnet._src.autograd.GRAPH_MANAGER cimport Node, Tape

cdef class Node:
    def __cinit__(self, object output, tuple parents, object bwd_fn):
        self.output = output
        self.parents = parents
        self.bwd_fn = bwd_fn


cdef class Tape:
    def __cinit__(self):
        self.nodes = []

    cpdef add(self, Node node):
        self.nodes.append(node)

    def __getitem__(self, int i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    cpdef clear(self):
        self.nodes.clear()


# Keep this pure Python, not declared in .pxd
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
