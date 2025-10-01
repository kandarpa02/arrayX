from collections import defaultdict

class Parameter:
    def __init__(self, init_fn, **kwargs):
        self.weights = init_fn(**kwargs)


class LayerMeta(type):
    _counters = defaultdict(int)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        base_name = instance.name if instance.name else cls.__name__  # user name or class
        count = LayerMeta._counters[base_name]
        instance.name = f"{base_name}{count}"  # always append count
        LayerMeta._counters[base_name] += 1
        return instance


class Layer(metaclass=LayerMeta):
    def __init__(self, name=""):
        self.name = name 

    def add_param(self, name, init_fn, **kwargs):
        setattr(self, name, Parameter(init_fn, **kwargs))
        return getattr(self, name)

    def _initiate(self):
        return [k for k in self.__dict__ if k != "name"]

    def _collect_params(self):
        params = {}
        for attr in self._initiate():
            v = getattr(self, attr)
            if isinstance(v, Parameter):
                params[attr] = v.weights
            elif isinstance(v, Layer):
                params[v.name] = v._collect_params()
        return params

    def params(self):
        return {self.name: self._collect_params()}
