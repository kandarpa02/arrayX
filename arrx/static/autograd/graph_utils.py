from ..Tensor.base import scalar, vector, matrix, placeholder
from ..Tensor import arithmetic, logarithmic
from .graph import Function
from typing import Union, Any, List
from dataclasses import dataclass
import inspect


def build_func_dict(*modules):
    func_dict = {}
    for mod in modules:
        funcs = {name: func for name, func in inspect.getmembers(mod, inspect.isfunction)}
        func_dict.update(funcs)
    return func_dict

_OP_MAP = build_func_dict(arithmetic, logarithmic)

# Typings
TensorLike = Union[scalar, vector, matrix]
from arrx import lib
NDarray = lib.ndarray


class EmptyNodeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FlashGraph:
    ___nodes: dict = None #type:ignore
    ___history: dict = None #type:ignore
    ___end_node: TensorLike = None #type:ignore

    """
    FlashGraph
    ----------
    A directed computation graph builder for ArrayX placeholders (scalar, vector, matrix).
    FlashGraph provides a minimal yet flexible API to construct symbolic graphs for forward
    computation and automatic differentiation. 
    
    This module is inspired by static graph frameworks (Theano, TensorFlow 1.x) but 
    redesigned to be lightweight, explicit, and compatible with modern scientific workflows.
    
    Attributes
    ----------
    nodes : dict
        A dictionary mapping node names to their computed graph nodes (endpoints after ops).
    history : dict
        A dictionary mapping placeholder names to their original placeholder objects.
    """
    
    @property
    def nodes(self):
        return self.___nodes
    
    @property
    def history(self):
        return self.___history
    
    def out_node(self):
        return self.___end_node

    def Init(self, prev=None, *, shape:list|tuple=[], node:Any|str=None): #type:ignore 

        """
        Init(prev=None, *, shape=[], node=None)
        ---------------------------------------
        Initialize the graph. This method sets the *starting point* of a FlashGraph
        by creating the very first placeholder node. 
        
        Arguments
        ---------
        prev : FlashGraph, optional
            Another FlashGraph instance to bootstrap from. If provided, the new graph 
            starts where the previous one left off (useful for model chaining).
        shape : list | tuple, optional
            Shape of the initial placeholder node. Defaults to scalar if left empty.
        node : str, optional
            Name of the initial node. Required if residual connections are intended later.

        Notes
        -----
        - Exactly one `Init` call is required before adding further operations.
        - If `prev` is passed, shape/node are ignored, and the new graph shares the 
          existing structure of the previous graph.
        """

        if not prev is None:
            node = prev
            if isinstance(node, FlashGraph):
                self.___end_node = node.___end_node
                self.___nodes = node.___nodes
                self.___history = node.___history
            
            else:
                raise ValueError(f"prev must be a {FlashGraph} object. But found {type(prev)}")
        else:
            place = placeholder.place(*shape, name=node)
            self.___end_node = place
            self.___nodes = {'init':place}
            self.___history = {place.name:place}

    def Synapse(self, operation:str='', out_node:placeholder=None, variables:list=[placeholder]): #type:ignore

        """
        synapse(operation='', out_node=None, variables=[])
        --------------------------------------------------
        Connect an *external placeholder expression* into the current FlashGraph.
        This allows fusing pre-defined symbolic expressions with FlashGraph nodes,
        enabling hybrid or advanced constructions.

        Arguments
        ---------
        operation : str
            The name of the operation (must exist in `_OP_MAP`).
        out_node : placeholder
            A placeholder or expression (e.g., result of custom scalar/matrix ops).
        variables : list of placeholders
            All placeholders involved in `out_node`. These will be registered 
            into the graph history if not already present.

        Notes
        -----
        - This is designed for advanced users who want to insert pre-built symbolic 
          expressions directly into a graph pipeline.
        - If operation only takes one input, it automatically falls back to unary mode.
        """

        try:
            _OP_MAP.get(operation, None)
        
        except KeyError:
            raise KeyError(f'{operation} is not a valid operation')
        
        finally:
            op = _OP_MAP.get(operation, None)


        try:
            self.___end_node = op(self.___end_node, out_node) #type:ignore
        except TypeError:
            self.___end_node = op(self.___end_node) #type:ignore

        for var in variables:
            if var.name not in self.___nodes:
                self.___nodes[var.name] = self.___end_node

            if var.name not in self.___history:
                self.___history[var.name] = var


    def Add(self, operation:str='', shape:list=[], node:str=None): #type:ignore

        """
        Add(operation='', shape=[], node=None)
        --------------------------------------
        Add a new intermediate node to the graph by applying an operation between 
        the current end-node and a newly created placeholder.

        Arguments
        ---------
        operation : str
            Operation name (e.g., 'add', 'mul', etc.) from `_OP_MAP`.
        shape : list, optional
            Shape of the new placeholder being added.
        node : str, optional
            Name of the new placeholder node. Needed for residual or debugging.

        Raises
        ------
        EmptyNodeError
            If called before Init().
        KeyError
            If an invalid operation is given.

        Notes
        -----
        - Add expands the graph sequentially. Each call mutates the current end-node.
        - Operation may fall back to unary if the op signature allows.
        """

        try:
            _OP_MAP.get(operation, None)
        
        except KeyError:
            raise KeyError(f'{operation} is not a valid operation')

        finally:
            op = _OP_MAP.get(operation, None)
            place = placeholder.place(*shape, name=node)
            if not operation == '':
                if not op is None:
                    if not self.___end_node is None:
                        try:
                            self.___end_node = op(self.___end_node, place) #type:ignore
                        except TypeError:
                            self.___end_node = op(self.___end_node)
                        # finally:
                        #     self.___end_node = op(self.___end_node, place)

                        self.___nodes[place.name] = self.___end_node
                        self.___history[place.name] = place
                    else:
                        raise EmptyNodeError(f'Graph is not initialized')
                else:
                    raise KeyError(f'Invalid operation [{operation}] is given')
                
            else:
                raise ValueError(f'Intermediate nodes must have an operation')
            
            
    def Add_Const(self, value:NDarray, operation:str=''):

        """
        Add_Const(value, operation='')
        ------------------------------
        Add a constant tensor into the graph and apply the given operation with 
        the current end-node.

        Arguments
        ---------
        value : NDarray
            Constant numerical value (NumPy/CuPy/etc.).
        operation : str
            Operation name from `_OP_MAP`.

        Notes
        -----
        - Unlike Add(), this does not store the constant into history, as constants
          are not treated as differentiable placeholders.
        - Use this for bias terms or fixed scalars in a model definition.
        """

        try:
            _OP_MAP.get(operation, None)
        
        except KeyError:
            raise KeyError(f'{operation} is not a valid operation')
        
        finally:
            op = _OP_MAP.get(operation, None)
            place = placeholder.place(*value.shape, name=f"{value}")
            if not operation == '':
                if not op is None:
                    if not self.___end_node is None:
                        try:
                            self.___end_node = op(self.___end_node, place) #type:ignore
                        except TypeError:
                            self.___end_node = op(self.___end_node)

                        self.___nodes[place.name] = self.___end_node
                        # self.___history[place.name] = place
                    else:
                        raise EmptyNodeError(f'Graph is not initialized')
                else:
                    raise KeyError(f'Invalid operation [{operation}] is given')
                
            else:
                raise ValueError(f'Intermediate nodes must have an operation')

            
    def Residual(self, operation:str='', shape:list=[], res_node:str=None, new_node:str=None): #type:ignore

        """
        Residual(operation='', shape=[], res_node=None, new_node=None)
        --------------------------------------------------------------
        Create a *residual (skip) connection* by combining the current end-node 
        with a previously defined placeholder node.

        Arguments
        ---------
        operation : str
            Operation to apply between the current end-node and the residual node.
        shape : list
            Shape of the new placeholder being added.
        res_node : str
            Name of the node from history to connect as residual.
        new_node : str
            Name of the resulting new node.

        Raises
        ------
        EmptyNodeError
            If res_node is None or missing.
        KeyError
            If operation is invalid.

        Notes
        -----
        - This mimics neural network residual/skip connections (like ResNets).
        - Residual nodes must already exist in history.
        """

        if res_node is None:
            raise EmptyNodeError(f"res_node can't be None in Residual method")
        
        try:
            self.___nodes.get(res_node, None)
        
        except KeyError:
            raise KeyError(f'[{res_node}] is not a resious node')
        
        else:
            res_node = self.___history.get(res_node, None) #type:ignore
        
        try:
            _OP_MAP.get(operation, None)
        
        except KeyError:
            raise KeyError(f'{operation} is not a valid operation')

        finally:
            op = _OP_MAP.get(operation, None)
            place = placeholder.place(*shape, name=new_node)
            if not operation == '':
                if not op is None:
                    try:
                        self.___end_node = op(self.___end_node, res_node) #type:ignore
                    except TypeError:
                        self.___end_node = op(self.___end_node)
                    # finally:
                    #     self.___end_node = op(self.___end_node, res_node) #type:ignore

                    self.___nodes[place.name] = self.___end_node
                    
                else:
                    raise KeyError(f'Invalid operation [{operation}] is given')
                
            else:
                raise ValueError(f'Intermediate nodes must have an operation')

        
    def Compile(self):

        """
        Compile()
        ---------
        Finalize the FlashGraph into an executable Function object. 
        This produces callable forward (`fw`) and backward (`bw`) 
        computation functions.

        Notes
        -----
        - Must be called after building the full graph.
        - Internally constructs a `Function` wrapper with the end-node and 
          all placeholders from history.
        """

        graph = Function(self.___end_node, list(self.___history.values()))
        self.fw, self.bw = graph.forward(), graph.backward()

    
    def apply(self, *args):

        """
        apply(*args)
        ------------
        Execute the compiled forward function on the provided input values.

        Arguments
        ---------
        *args : list
            Numerical values (NumPy/CuPy arrays) to feed into placeholders,
            in the order of their definition.

        Returns
        -------
        result : numeric
            The computed output value from the graph.
        """
        try:
            out = self.fw(*args)
        
        except AttributeError:
            raise AttributeError(f"Compile() method has to be called before")
        
        return self.fw(*args)
    
    def grad(self, *args):

        """
        grad(*args)
        -----------
        Execute the compiled backward function to compute gradients with respect 
        to all placeholders.

        Arguments
        ---------
        *args : list
            Numerical values (NumPy/CuPy arrays) to feed into placeholders.

        Returns
        -------
        grads : tuple
            Gradients for each input placeholder in order.
        """

        try:
            out = self.bw(*args)
        
        except AttributeError:
            raise AttributeError(f"Compile() method has to be called before")
        
        return self.bw(*args)