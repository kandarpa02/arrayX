from .core.Array import ArrayImpl, NumericObject
from typing import Protocol, runtime_checkable, Union


Array = Union[NumericObject, ArrayImpl]
