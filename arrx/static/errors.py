class ShapeError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)

class CompilationError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)