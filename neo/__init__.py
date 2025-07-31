# from neo._src import autograd
from neo.functions import function, neo_function
from neo._torch.functions import *

from neo._torch.user_functions import *
from neo._torch.random import *
from neo._torch import neolib
from neo._src._extras import save, load
from neo._src import nn
from neo._src.autograd.SESSION import value_and_grad as _vag
from neo._src.autograd.FUNCTION_REGISTER import Policy as _policy

value_and_grad = _vag
Policy = _policy