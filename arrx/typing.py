from .core.Array import ArrayImpl, NumericObject
from typing import Protocol, runtime_checkable

class Array(Protocol):
    def __init__(self, data: NumericObject) -> None:
        ...

    @property
    def _rawbuffer(self) -> NumericObject:
        ...

    def __add__(self, other) -> ArrayImpl:
        ...