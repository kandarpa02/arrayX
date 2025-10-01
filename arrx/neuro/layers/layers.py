from collections import defaultdict

class Parameter:
    def __init__(self, init_fn, **kwargs):
        self.weights = init_fn(**kwargs)


class LayerMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        base_name = instance.name if instance.name else cls.__name__
        instance.name = instance._next_name(base_name)
        return instance


class Layer(metaclass=LayerMeta):
    _root = None

    def __init__(self, name=""):
        self.name = name
        if Layer._root is None:
            Layer._root = self
            self._counters = defaultdict(int)

    def _next_name(self, base_name):
        counters = Layer._root._counters #type:ignore
        count = counters[base_name]
        counters[base_name] += 1
        return f"{base_name}{count}"

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

    def load(self, new_params):
        # TODO: implement recursive loading properly
        self._loaded_params = new_params

    def call(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        for attr in self._initiate():
            v = getattr(self, attr)
            if isinstance(v, Layer):
                args = v.call(*args)
        return args
