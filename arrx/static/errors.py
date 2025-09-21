class ShapeError(Exception):
    def __init__(self, true_dim, name) -> None:
        super().__init__(f"given dim {true_dim} is not correct for class [{name}]")