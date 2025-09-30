from typing import Sequence, List
from ..Tensor.base import placeholder
from ..errors import CompilationError


class _compiler:
    def __init__(self, out: placeholder, var: List[placeholder], debug=False):
        self.out = out
        self.var = var
        self.code = None
        self.debug = debug

        self._forward = self._fwd_fn(debug)
        self._backward = self._grad_fn_stack(debug)

        
    def _fwd_fn(self, debug=False):
        var_list = self.var
        arg_list = ', '.join(v.expr for v in var_list) if var_list else '' #type:ignore
        out_expr = self.out.expr.replace('init_grad', '1') #type:ignore
        fun_code = f"def func({arg_list}, **kwargs):\n"
        fun_code += f"    from arrx.internal_ops import OPS\n"
        fun_code += f"    from arrx.Tensor.utils import lib\n"
        fun_code += f"    return {out_expr}"
        namespace = {}
        if debug:
            if self.code is None:
                self.code = {}
            self.code['forward'] = fun_code
        try:
            exec(fun_code, namespace)
        except:
            raise CompilationError(
                f"Some fatal issue is occured that the compiler could not figure out. Better exception handling will be added in future."
            )
        
        return namespace["func"]


    def _grad_fn_stack(self, debug=False):
        # build symbolic grads
        def topological_sort(node):
            visited, order = set(), []
            def visit(n):
                if n not in visited:
                    visited.add(n)
                    for p in n.parents:
                        visit(p)
                    order.append(n)
            visit(node)
            return order

        def _backward(out):
            # Reset grads in graph (important if called multiple times)
            for n in topological_sort(out):
                n.grad = None

            out.grad = placeholder.as_place(self.out, 'init_grad')
            for node in reversed(topological_sort(out)):
                if node.grad_fn:
                    grads = node.grad_fn(node.grad)
                    for parent, g in zip(node.parents, grads):
                        if parent.grad is None:
                            parent.grad = g
                        else:
                            parent.grad = parent.grad + g

        _backward(self.out)

        # arg list for the back function: (init_grad, x, y, z, ...)
        arg_names = [v.expr for v in self.var]
        arg_str = ', '.join(arg_names) #type:ignore

        # collect the grad placeholders for each variable (these are placeholders)
        grad_placeholders = [v.grad for v in self.var]

        # Defensive: ensure grads exist
        for i, g in enumerate(grad_placeholders):
            if g is None: 
                raise RuntimeError(f"gradient for var {self.var[i].expr} is None")

        
        body_lines = []
        for i, g in enumerate(grad_placeholders):
            # g.expr is a string representation of the symbolic expression
            # we will directly place it into the function body
            body_lines.append(f"    g{i} = {g.expr}") #type:ignore

        body_lines.append(f"    return ({', '.join('g'+str(i) for i in range(len(grad_placeholders)))})")
        fun_code = f"def back_fn({arg_str}):\n" 
        fun_code += f"    from arrx.Tensor.utils import lib\n"
        fun_code += f"    from arrx.internal_ops import OPS\n"
        fun_code += f"    init_grad = lib.numpy.ones({self.out.shape})\n"
        fun_code += "\n".join(body_lines)

        namespace = {}
        # exec in a fresh namespace so symbols x,y,z,init_grad will be function args
        if debug:
            if self.code == None:
                self.code = {}
            self.code['backward'] = fun_code
        exec(fun_code, namespace)
        return namespace["back_fn"]


class Function:
    def __init__(self, out: placeholder, var: List[placeholder], debug=False):
        self._compiler = _compiler(out, var, debug)
        self.fwd = None
        self.bwd = None

    def _fwdjit_normalized(self, *args):
        import jax
        jit_fn = jax.jit(self._compiler._forward) #type:ignore
        out =  jit_fn(*args)
        self.fwd = jit_fn
        return out

    def _bwdjit_normalized(self, *args):
        import jax
        jit_fn = jax.jit(self._compiler._backward) #type:ignore
        out =  jit_fn(*args)
        self.bwd = jit_fn
        return out

    def apply(self, *args):
        if self.fwd is None:
            return self._fwdjit_normalized(*args)
        else:
            return self.fwd(*args)

    def grad(self, *args):
        if self.bwd is None:
            return self._bwdjit_normalized(*args)
        else:
            return self.bwd(*args)
