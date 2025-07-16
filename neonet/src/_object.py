from typing import Any, NamedTuple, Self as Self

class Tensor(NamedTuple):
    value : int | float

    def __repr__(self) -> str:
        return f"{self.value}"
    
    def get(self):
        return self.value

    def _repl(self, new_value:(int | float)) -> Self:
        return self._replace(value = new_value)

