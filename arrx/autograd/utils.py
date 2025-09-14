from typing import NamedTuple, Callable, Tuple
from ..core.Array import ArrayImpl

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
                            "objects: (output, backward_function) when patch_node==False, "
                            f"but got {fn_out}"
                        )
                    
                parent_inputs = None
                if not out.parents == ():
                    parent_inputs = out.parents
                else:
                    parent_inputs = tuple([i for i in [*args, *kwargs.values()] if isinstance(i, ArrayImpl)])

                print(f"parents:\n {parent_inputs}")
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
                            "objects: (output, parents, backward_function), but got {fn_out}"
                        )
                parents = _parents if isinstance(_parents, (tuple, list)) else (_parents,)
                print(f"parents:\n {parents}")
                out.parents = parents
                out.bwd_fn = bwd
                return out
            
        return wrapper
