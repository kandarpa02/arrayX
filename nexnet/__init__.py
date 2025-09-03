# from neo._src import autograd
from nexnet.functions import function, neo_function
from nexnet._torch.functions import *

from nexnet._torch.user_functions import *
from nexnet._torch.random import *
from nexnet._torch import neolib
from nexnet._src._extras import save, load

from nexnet._src import nn
from nexnet._src.nn import (
    dense_layer, 
    conv_layer,

    avg_pool1d,
    avg_pool2d,
    avg_pool3d,

    max_pool1d,
    max_pool2d,
    max_pool3d,

    batchnorm1d,
    batchnorm2d,
    batchnorm3d,

)

# Layers
Dense = dense_layer.Dense
Conv1D = conv_layer.Conv1D
Conv2D = conv_layer.Conv2D
Conv3D = conv_layer.Conv3D

# Activations
from nexnet._src.nn._activations import *

from nexnet._src.nn.layers import (
    Module as _module
)
Module = _module


# Autograd stuff
from nexnet._src.autograd.backward_utility import (
            build_computation_graph as _bcg,
            value_and_grad as _vag,
            grad as _g,
            GraphContext as _gctx
        )

from nexnet._src.autograd._backward_utility_static import StaticGraphBuilder as _sgb
from nexnet._src.autograd.define_then_run import Variable as var, Constant as cnst, Symbol as sym, run_graph as r_g, eval_graph as e_g
from nexnet._src.autograd.FUNCTION_REGISTER import Policy as _policy, Tracelet as _tracelet, custom_grad as _cstm_grad


build_computation_graph = _bcg
value_and_grad = _vag
grad = _g
GraphContext = _gctx
# StaticGraphBuilder = _sgb

Variable = var
Constant = cnst
Symbol = sym
run_graph = r_g
eval_graph = e_g

Policy = _policy
Tracelet = _tracelet
custom_grad = _cstm_grad


from contextlib import contextmanager

class record_tape:
    enabled = True

    @staticmethod
    def is_enabled():
        return record_tape.enabled

    @staticmethod
    def set(mode: bool):
        record_tape.enabled = mode

    @contextmanager
    def no_tape():
        old = record_tape.enabled
        record_tape.enabled = False
        try:
            yield
        finally:
            record_tape.enabled = old
