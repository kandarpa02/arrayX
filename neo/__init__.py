# from neo._src import autograd
from neo.functions import function, neo_function
from neo._torch.functions import *

from neo._torch.user_functions import *
from neo._torch.random import *
from neo._torch import neolib
from neo._src._extras import save, load
from neo._src import nn, nn_static
from neo._src.autograd._backward_utility import build_computation_graph as _bcg
from neo._src.autograd._backward_utility_static import StaticGraphBuilder as _sgb
from neo._src.autograd.define_then_run import Variable as var, Constant as cnst, Symbol as sym, run_graph as r_g, eval_graph as e_g
from neo._src.autograd.FUNCTION_REGISTER import Policy as _policy


build_computation_graph = _bcg
# StaticGraphBuilder = _sgb

Variable = var
Constant = cnst
Symbol = sym
run_graph = r_g
eval_graph = e_g

Policy = _policy