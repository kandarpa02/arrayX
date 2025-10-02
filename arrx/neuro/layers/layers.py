from collections import defaultdict
from ..utils import ParamDict
from ...src.Tensor.base import placeholder


class Parameter:
    def __init__(self, init_fn, **kwargs):
        self.weights = init_fn(**kwargs)


class LayerMeta(type):
    _global_counters = defaultdict(int)  
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        # generate unique hierarchical name
        base_name = instance.name if instance.name else cls.__name__
        count = LayerMeta._global_counters[base_name]
        LayerMeta._global_counters[base_name] += 1
        instance.name = f"{base_name}{count}"
        return instance


class Layer(metaclass=LayerMeta):
    def __init__(self, name=""):
        self.name = name
        # each Layer instance gets its own local counter for children
        self._counters = defaultdict(int)

    def _next_name(self, base_name: str) -> str:
        count = self._counters[base_name]
        self._counters[base_name] += 1
        return f"{base_name}{count}"

    def add_param(self, name: placeholder, init_fn, **kwargs):
        # unchanged — uses hierarchical placeholder name
        full_name = f"{self.name}/{name.expr}"
        place = placeholder.place(*name.shape, name=full_name)
        param = ParamDict({place: init_fn(**kwargs)})
        setattr(self, place.expr, param)
        return place

    def _initiate(self):
        return [k for k in self.__dict__ if k not in ("name", "_counters")]

    def _collect_params(self):
        params = ParamDict()
        for attr in self._initiate():
            v = getattr(self, attr)
            if isinstance(v, ParamDict):
                # param dict already keyed by placeholder
                params.update(v)
            elif isinstance(v, Layer):
                params[v.name] = v._collect_params()
        return params

    def params(self):
        return ParamDict({self.name: self._collect_params()})

    def load(self, new_params):
        # TODO: implement recursive loading properly
        self._loaded_params = new_params

    def call(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.call(*args)
