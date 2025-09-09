from .core.tensor import DeviceBuffer, NumericObject
from typing import Protocol, runtime_checkable

class Tensor(Protocol):
    def __init__(self, data: NumericObject) -> None:
        ...

    @property
    def _rawbuffer(self) -> NumericObject:
        ...

    def __add__(self, other) -> DeviceBuffer:
        ...