# Copyright (c) 2025 Kandarpa Sarkar
# This file is part of the NeoNet project and is licensed under the MIT License.
# See the LICENSE file in the root directory for more information.

from neo._src.autograd import Node, Tape, TapeContext
from typing import Callable, List, Any
from neo._torch import neolib
from neo._torch.lite_tensor import LiteTensor

def rectify_shapes(val):
    return val.reshape(1) if val.ndim < 1 else val

def unpack_tuple(tup):
    return {f'x{i+1}': value for i, value in enumerate(tup)}

def if_xnary(grads):
    def _fix(g):
        if g.ndim == 0:
            return g.reshape(1)
        elif g.ndim == 1:
            return g[None, :]
        return g

    if isinstance(grads, tuple):
        return tuple(_fix(g) for g in grads)
    else:
        return _fix(grads)
 

# import ast
# import inspect
# import textwrap

# def make_function(method):
#     src = textwrap.dedent(inspect.getsource(method))
#     tree = ast.parse(src)

#     func_def = tree.body[0]  # def backward(...) function

#     # self.ctx.release unpack line
#     release_vars = []
#     new_body = []
#     for node in func_def.body:
#         if isinstance(node, ast.Assign):
#             if (isinstance(node.value, ast.Attribute) and
#                 isinstance(node.value.value, ast.Attribute) and
#                 isinstance(node.value.value.value, ast.Name) and
#                 node.value.value.value.id == "self" and
#                 node.value.value.attr == "ctx" and
#                 node.value.attr == "release"):
#                 # This is the ctx.release line
#                 if isinstance(node.targets[0], ast.Tuple):
#                     release_vars = [elt.id for elt in node.targets[0].elts]
#                 continue 
#         new_body.append(node)

#     new_args = release_vars + ['grad']
#     new_func = ast.FunctionDef(
#         name="extracted_backward",
#         args=ast.arguments(
#             posonlyargs=[],
#             args=[ast.arg(arg=a) for a in new_args],
#             vararg=None,
#             kwonlyargs=[],
#             kw_defaults=[],
#             kwarg=None,
#             defaults=[],
#         ),
#         body=new_body,
#         decorator_list=[],
#         returns=None,        
#         type_comment=None,      
#         type_params=[]
#     )

#     new_module = ast.Module(body=[new_func], type_ignores=[])
#     ast.fix_missing_locations(new_module)

#     compiled = compile(new_module, filename="<ast>", mode="exec")
#     namespace = {}
#     exec(compiled, globals(), namespace)
#     return namespace["extracted_backward"]


# def _grad_function(fn:Callable):
#     def wrapped(*args):

#         tape = Tape()
#         TapeContext.push(tape)
#         out = fn(*args)

#         TapeContext.pop()
#         return out, tape
#     return wrapped


# def _gradient(fn:Callable, args:list, safe=False):
#     import torch
#     out, tape = _grad_function(fn)(*args)
#     out_grad =  neolib.ones_like(out.data)

#     grad_dict = {id(out): out_grad}
#     any_cuda = out_grad.is_cuda 

#     for node in reversed(tape):
#         node_out_id = id(node.output)
#         node_out_grad = grad_dict.pop(node_out_id, None)
#         if node_out_grad is None:
#             continue
        
#         grads = node.bwd_fn(grad=node_out_grad)

#         if grads is None:
#             node.parents = None
#             continue

#         if not isinstance(grads, tuple):
#             grads = (grads,)
#         if len(grads) < len(node.parents):
#             grads = grads + (None,) * (len(node.parents) - len(grads))

#         for parent, grad in zip(node.parents, grads):
#             if grad is None:
#                 continue

#             if grad.is_cuda:
#                 any_cuda = True

#             pid = id(parent)
#             if pid in grad_dict:
#                 grad_dict[pid].add_(grad.clone() if safe else grad)
#             else:
#                 grad_dict[pid] = grad.clone() if safe else grad

#             del grad  
    
#     input_grads = {}
#     for arg in args:
#         grad = grad_dict.get(id(arg))
#         if grad is not None:
#             input_grads[arg] = LiteTensor(grad)

#     if any_cuda:
#         torch.cuda.empty_cache()

#     grads_list = list(input_grads.values())
#     grad_out = grads_list[0] if len(grads_list) == 1 else grads_list

#     return out, grad_out, tape


# def grad(fn:Callable|List[Callable], args:list, graph=False, safe=False):
#     if not isinstance(fn, list):
#         out, out_grad, tape = _gradient(fn, args, safe=safe)
#     else:
#         fn_stack = []
#         input_stack = []
#         for node in fn:
#             _dummy_grad = neolib.ones_like(node.output.data)
#             extracted_fn = make_function(node.bwd)
#             fn_stack.append(extracted_fn)
#             parents = node.parents
#             parents.append(_dummy_grad)
#             input_stack.append(parents)

#         _seq = Callable

#         out, out_grad, tape = _gradient(_seq, parents, safe=safe)

#     if graph:
#         return out, out_grad, tape
#     else:
#         return out, out_grad


def value_and_grad(fn: Callable, safe=False):
    def wrapped_function(args:list):
        import torch
        torch.set_grad_enabled(False)

        tape = Tape()
        TapeContext.push(tape)
        out = fn(*args)
        if not hasattr(out, 'data'):
            print(out)
            raise TypeError(
                f"value_and_grad expected `fn` to return a scalar-like LiteTensor, "
                f"but got {type(out)}: {out}"
        )
        TapeContext.pop()

        out_grad = neolib.ones_like(out.data)
        grad_dict = {id(out): out_grad}

        any_cuda = out_grad.is_cuda  

        for node in reversed(tape):
            node_out_id = id(node.output)
            node_out_grad = grad_dict.pop(node_out_id, None)
            if node_out_grad is None:
                continue

            grads = node.bwd_fn(grad=node_out_grad)

            node.output = None
            node.bwd_fn = None

            if grads is None:
                node.parents = None
                continue

            if not isinstance(grads, tuple):
                grads = (grads,)
            if len(grads) < len(node.parents):
                grads = grads + (None,) * (len(node.parents) - len(grads))

            for parent, grad in zip(node.parents, grads):
                if grad is None:
                    continue

                if grad.is_cuda:
                    any_cuda = True

                pid = id(parent)
                if pid in grad_dict:
                    grad_dict[pid].add_(grad.clone() if safe else grad)
                else:
                    grad_dict[pid] = grad.clone() if safe else grad

                del grad  

            node.parents = None 
            del node  

        input_grads = {}
        for arg in args:
            grad = grad_dict.get(id(arg))
            if grad is not None:
                input_grads[arg] = LiteTensor(grad)

        if any_cuda:
            torch.cuda.empty_cache()

        grads_list = list(input_grads.values())
        grad_out = grads_list[0] if len(grads_list) == 1 else grads_list

        return out, grad_out

    return wrapped_function

