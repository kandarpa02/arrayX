from collections import defaultdict
from ..utils import ParamDict
from ...src.Tensor.base import placeholder


class Parameter:
    def __init__(self, init_fn, **kwargs):
        self.weights = init_fn(**kwargs)

class LayerMeta(type):
    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)

class Layer(metaclass=LayerMeta):
    def __init__(self, name: str = ""):
        self.name = name
        self._counters = defaultdict(int) 

    def _next_name(self, base_name: str) -> str:
        count = self._counters[base_name]
        self._counters[base_name] += 1
        return f"{base_name}{count}"

    def __setattr__(self, key, value):
        if isinstance(value, Layer) and key not in ("name", "_counters"):
            base = value.name or value.__class__.__name__
            if hasattr(self, "_counters"):
                assigned = f"{base}{self._counters[base]}"
                self._counters[base] += 1
            else:
                assigned = base
            value.name = assigned
            object.__setattr__(self, key, value)
            return
        object.__setattr__(self, key, value)

    def add_param(self, name: placeholder, init_fn, **kwargs):
        """
        Create placeholder-based param internally, but use readable string
        keys when exporting via params(). Keeps internal storage unchanged.
        """
        # Use layer's own name (if set) otherwise use class name (no numeric model prefix)
        layer_name = self.name if self.name else self.__class__.__name__
        full_name = f"{layer_name}_{name.expr}"
        place = placeholder.place(*name.shape, name=full_name)
        param = ParamDict({place: init_fn(**kwargs)})
        # attach ParamDict to this layer instance under the placeholder expression
        setattr(self, place.expr, param)
        return place

    def _initiate(self):
        return [k for k in self.__dict__ if k not in ("name", "_counters")]


    def _collect_params(self): 
        params = ParamDict() 
        for attr in self._initiate(): 
            v = getattr(self, attr) 
            if isinstance(v, ParamDict): 
                params.update(v) 
            elif isinstance(v, Layer):
                params[v.name] = v._collect_params() 
        return params

    def params(self):
        # Return children/params directly — no top-level model wrapper or count.
        return ParamDict({self.name : self._collect_params()})

    @property
    def param_vals(self):
        return self.params().to_list()

    def load(self, new_params):
        self._loaded_params = new_params

    def call(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.call(*args)


def compile(self, *args):
    out = self.__call__(*args)
    from ...src.autograd.graph import Function
    va = [*args]
    va.extend(self.params().variables())
    graph = Function(out, va)
    return graph.apply, graph.grad
