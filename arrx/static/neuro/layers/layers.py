class parameters:
    def __init__(self, shape, init_fn, rng=None, dtype=None, name=''):
        """
        Wraps a parameter tensor with a name for hierarchical param management.
        """
        self.name = name
        self.value = {name: self._init_value(shape, init_fn, rng, dtype)}

    def _init_value(self, shape, init_fn, rng, dtype):
        try:
            return init_fn(shape, rng, dtype)
        except TypeError:
            try:
                return init_fn(shape, dtype)
            except TypeError:
                return init_fn(shape)

class Layer:
    def __init__(self, name=''):
        self.base_name = name or self.__class__.__name__.lower()
        self.name = None
        self._params = {}  # maps surface names to parameters
        self._counter = None

    def add_param(self, shape, init_fn, rng=None, dtype=None, param_name=None):
        """
        Add a parameter with a semantic name like 'weight' or 'bias'.
        param_name is the surface-level name used in flattened dict.
        """
        if param_name is None:
            param_name = f"param{len(self._params)}"
        param = parameters(shape, init_fn, rng, dtype, param_name)
        self._params[param_name] = param
        return param

    def assign_names(self, name_counters=None):
        if name_counters is None:
            name_counters = {}
        count = name_counters.get(self.base_name, 0)
        self.name = f"{self.base_name}{count}"
        name_counters[self.base_name] = count + 1

        # recursively assign names to sublayers
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, Layer):
                value.assign_names(name_counters)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Layer):
                        v.assign_names(name_counters)

    def params_dict(self, prefix=''):
        """
        Flatten all parameters into a dict with hierarchical names.
        Uses semantic names for surface parameters like 'weight', 'bias'.
        """
        param_dict = {}
        prefix = f"{prefix}{self.name}/" if prefix else f"{self.name}/"

        # Add own parameters
        for pname, param in self._params.items():
            param_dict[f"{prefix}{pname}"] = param.value[pname]

        # Add nested layer parameters
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, Layer):
                param_dict.update(value.params_dict(prefix=prefix))
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, Layer):
                        param_dict.update(v.params_dict(prefix=f"{prefix}{attr_name}/{i}/"))
        return param_dict
