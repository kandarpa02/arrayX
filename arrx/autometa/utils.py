from typing import NamedTuple, Callable, Tuple
from itertools import chain
from ..Core.Array import ArrayImpl

AI = ArrayImpl
_is = isinstance


class custom_grad:
    __slots__ = ['fn', 'patch']
    
    def __init__(self, fn=None, patch_nodes=False):
        self.fn = fn
        self.patch = patch_nodes

    def __call__(self, *args, **kwargs):
        if self.fn is not None:
            return self.decorate(self.fn)(*args, **kwargs)
        
        fn = args[0]
        return self.decorate(fn)

    def decorate(self, fn):
        patch = self.patch
        
        def wrapper(*args, **kwargs):
            
            fn_out = fn(*args, **kwargs)

            if not patch:
                try:
                    out, bwd = fn_out
                except:
                    if len(fn_out) != 2:
                        raise ValueError(
                            f"input function {fn} is expected to return a tuple of two "
                            "objects: \n(output, backward_function) when patch_node==False,\n "
                            f"but got \n{fn_out}"
                        )
                    
                parent_inputs = tuple(
                    i for i in chain(args, kwargs.values()) if _is(i, AI)
                )

                out.parents = parent_inputs
                out.bwd_fn = bwd
                return out
            
            else:
                try:
                    out, _parents, bwd = fn_out
                except:
                    if len(fn_out) != 3:
                        raise ValueError(
                            f"input function {fn} is expected to return a tuple of three "
                            "objects: \n(output, parents, backward_function),\n but got \n{fn_out}"
                        )
                parents = _parents if _is(_parents, (tuple, list)) else (_parents,)
                out.parents = parents
                out.bwd_fn = bwd
                return out
            
        return wrapper
