import torch

def _device(arg):
    if arg is None:
        return None
    if isinstance(arg, torch.device):
        return arg
    if isinstance(arg, str):
        return torch.device(arg)
    raise TypeError(f"Invalid device: {arg}")

def _dtype(arg):
    if arg is None:
        return None
    if isinstance(arg, torch.dtype):
        return arg
    if isinstance(arg, str):
        try:
            return getattr(torch, arg)
        except AttributeError:
            raise TypeError(f"Invalid dtype string: '{arg}'")
    raise TypeError(f"Invalid dtype: {arg}")
